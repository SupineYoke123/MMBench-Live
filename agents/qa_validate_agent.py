import asyncio
import json
import time
from pathlib import Path
from typing import Any
import requests
from mcp.server.fastmcp import FastMCP
from project_config import DEPTH_MCP, QA_DIR, QA_VALIDATE_AGENT_MCP_PORT, SEGMENTATION_MCP, VISION_MCP, WRONG_QA_PATH


mcp = FastMCP("qa_validate_tools", port=QA_VALIDATE_AGENT_MCP_PORT)
REQUEST_TIMEOUT = 300
_QA_ITERATORS: dict[str, dict[str, Any]] = {}


def _safe_task_name(task_name: str) -> str:
    return "_".join(task_name.split())


def _task_name_from_qa_path(path: Path) -> str:
    name = path.stem
    return name[:-3] if name.endswith("_qa") else name


def _qa_path_for_task(task_name: str) -> Path:
    safe_name = _safe_task_name(task_name)
    candidates = [
        Path(QA_DIR) / f"{task_name}_qa.json",
        Path(QA_DIR) / f"{safe_name}_qa.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[-1]


def _load_qa_items(qa_path: Path) -> list[tuple[str, dict[str, Any]]]:
    if not qa_path.exists():
        raise FileNotFoundError(f"QA file not found: {qa_path}")
    with open(qa_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"QA file must be a JSON object: {qa_path}")
    return [(qid, qa) for qid, qa in data.items() if isinstance(qa, dict)]


def _dump_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    tmp_path.replace(path)


def _post_json(url: str, payload: dict[str, Any]) -> Any:
    response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()["result"]


@mcp.tool()
async def get_all_task_qa_file_paths():
    """
    Get all task QA file paths.

    Returns:
    - A list of task QA files under QA_DIR. Each item contains task_name, qa_path,
      and qa_count.
    """
    qa_dir = Path(QA_DIR)
    if not qa_dir.exists():
        return []

    results = []
    for qa_path in sorted(qa_dir.glob("*_qa.json")):
        qa_count = 0
        try:
            with open(qa_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                qa_count = len(data)
        except Exception:
            qa_count = -1

        results.append(
            {
                "task_name": _task_name_from_qa_path(qa_path),
                "qa_path": qa_path.as_posix(),
                "qa_count": qa_count,
            }
        )
    return results


@mcp.tool()
async def request_task_qa_pair(task_name: str, reset: bool = False):
    """
    Request the next QA pair for one task.

    Parameters:
    - task_name: Task name, usually the name returned by get_all_task_qa_file_paths.
    - reset: If true, restart iteration for this task from the first QA pair.

    Returns:
    - The next QA pair for the task on each call.
    - When all QA pairs have been returned, returns a done message.
    """
    qa_path = _qa_path_for_task(task_name)
    iterator_key = qa_path.as_posix()

    if reset or iterator_key not in _QA_ITERATORS:
        _QA_ITERATORS[iterator_key] = {
            "task_name": _task_name_from_qa_path(qa_path),
            "qa_path": iterator_key,
            "items": _load_qa_items(qa_path),
            "index": 0,
        }

    state = _QA_ITERATORS[iterator_key]
    items = state["items"]
    index = state["index"]
    total = len(items)

    if index >= total:
        return {
            "status": "done",
            "task_name": state["task_name"],
            "qa_path": state["qa_path"],
            "index": index,
            "total": total,
            "message": "Completed traversal",
        }

    question_id, qa_pair = items[index]
    state["index"] = index + 1
    return {
        "status": "ok",
        "task_name": state["task_name"],
        "qa_path": state["qa_path"],
        "index": index + 1,
        "total": total,
        "question_id": question_id,
        "qa_pair": qa_pair,
    }


@mcp.tool()
async def record_wrong_qa_pair(
    task_name: str,
    question_id: str,
    qa_pair: dict[str, Any],
    error_reason: str,
    corrected_answer: str | None = None,
    validation_evidence: dict[str, Any] | None = None,
):
    """
    Record a QA pair that is found to be wrong during validation.

    Parameters:
    - task_name: Task name of the QA pair.
    - question_id: QA/question id in the task QA file.
    - qa_pair: The original QA pair object.
    - error_reason: Why this QA pair is considered wrong.
    - corrected_answer: Optional corrected answer.
    - validation_evidence: Optional evidence collected by validation tools.

    Returns:
    - A summary containing the saved record path and record key.
    """
    if not question_id:
        raise ValueError("question_id is required.")
    if not isinstance(qa_pair, dict):
        raise ValueError("qa_pair must be a JSON object.")

    qa_path = _qa_path_for_task(task_name)
    record_key = f"{_task_name_from_qa_path(qa_path)}::{question_id}"

    if WRONG_QA_PATH.exists():
        try:
            with open(WRONG_QA_PATH, "r", encoding="utf-8") as f:
                wrong_records = json.load(f)
            if not isinstance(wrong_records, dict):
                wrong_records = {}
        except Exception:
            wrong_records = {}
    else:
        wrong_records = {}

    wrong_records[record_key] = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task_name": _task_name_from_qa_path(qa_path),
        "qa_path": qa_path.as_posix(),
        "question_id": question_id,
        "qa_pair": qa_pair,
        "error_reason": error_reason,
        "corrected_answer": corrected_answer,
        "validation_evidence": validation_evidence or {},
    }

    _dump_json_atomic(WRONG_QA_PATH, wrong_records)
    return {
        "status": "recorded",
        "record_key": record_key,
        "wrong_qa_path": WRONG_QA_PATH.as_posix(),
    }


@mcp.tool()
async def recognize(image_path: str):
    """
    Recognize the main visual content in an image.

    Parameters:
    - image_path: Absolute path to the image.

    Returns:
    - A global description of the image and the main objects present in it.
    """
    return await asyncio.to_thread(
        _post_json,
        f"{VISION_MCP}/recognize",
        {"image_path": image_path},
    )


@mcp.tool()
async def detect_and_seg(query: str, image_path: str):
    """
    Detect and segment target objects in an image.

    Parameters:
    - query: Object or region to detect.
    - image_path: Absolute path to the image.

    Returns:
    - Bounding boxes or segmentation results for the requested target.
    """
    return await asyncio.to_thread(
        _post_json,
        SEGMENTATION_MCP,
        {"query": query, "image_path": image_path},
    )


@mcp.tool()
async def ocr(image_path: str, bbox):
    """
    Read text from an image or specified image regions.

    Parameters:
    - image_path: Absolute path to the image.
    - bbox: Bounding boxes that need OCR.

    Returns:
    - OCR results for text in the selected regions.
    """
    return await asyncio.to_thread(
        _post_json,
        f"{VISION_MCP}/ocr_base64",
        {"image_path": image_path, "bbox": bbox},
    )


@mcp.tool()
async def see_depth(image_path: str):
    """
    Estimate relative depth relationships in an image.

    Parameters:
    - image_path: Absolute path to the image.

    Returns:
    - A text description of relative object depth and camera-distance relationships.
    """
    depth_base64 = await asyncio.to_thread(
        _post_json,
        DEPTH_MCP,
        {"image_path": image_path},
    )
    return await asyncio.to_thread(
        _post_json,
        f"{VISION_MCP}/see_depth",
        {"image_path": image_path, "base64": depth_base64},
    )


@mcp.tool()
async def see_attribute(image_path: str, bbox):
    """
    Inspect visual attributes for specified image regions.

    Parameters:
    - image_path: Absolute path to the image.
    - bbox: Bounding boxes whose attributes should be inspected.

    Returns:
    - Attribute descriptions such as color, material, pose, state, or shape.
    """
    return await asyncio.to_thread(
        _post_json,
        f"{VISION_MCP}/see_attribute",
        {"image_path": image_path, "bbox": bbox},
    )


if __name__ == "__main__":
    mcp.run(transport="sse")
