import asyncio
import base64
import itertools
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Tuple

from mcp.server.fastmcp import FastMCP
from openai import AsyncOpenAI
from PIL import Image
from tqdm import tqdm
from project_config import GEMINI_API_KEY, IMAGE_INFO_DIR, IMAGE_SVAE_DIR, OPENAI_API_KEY, QA_DIR, DEFAULT_GEMINI_GENERATION_MODEL, DEFAULT_GPT_GENERATION_MODEL, QA_GENERATION_AGENT_MCP_PORT


mcp = FastMCP("qa_generation_tools", port=QA_GENERATION_AGENT_MCP_PORT)

AVAILABLE_VISUAL_TOOLS = ["recognize", "detect_and_seg", "see_depth", "see_attribute", "ocr"]


class JSONParseError(ValueError):
    pass


def _load_json(json_path: str) -> Any:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON file: {e}") from e


def _dump_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def _tasks_root(data: Dict[str, Any]) -> Dict[str, Any]:
    if "finest_granularity_tasks" in data:
        return data["finest_granularity_tasks"]
    if "raw_tool_output" in data and "finest_granularity_tasks" in data["raw_tool_output"]:
        return data["raw_tool_output"]["finest_granularity_tasks"]
    raise KeyError('Summary JSON must contain "finest_granularity_tasks".')


def _question_type(data: Dict[str, Any]) -> str:
    value = data.get("question_types")
    if value is None:
        value = data.get("raw_tool_output", {}).get("question_types")
    if isinstance(value, list):
        return value[0] if value else "multiple-choice"
    return value or "multiple-choice"


def _safe_task_name(task_name: str) -> str:
    return "_".join(task_name.split())


def _task_info(data: Dict[str, Any], task_name: str) -> Tuple[str, Dict[str, Any]]:
    tasks = _tasks_root(data)
    if task_name in tasks:
        return task_name, tasks[task_name]
    safe_name = _safe_task_name(task_name)
    for key, value in tasks.items():
        if _safe_task_name(key) == safe_name:
            return key, value
    raise KeyError(f"Task not found in summary JSON: {task_name}")


def _task_image_info_path(task_name: str) -> Path:
    safe_name = _safe_task_name(task_name)
    candidates = [
        Path(IMAGE_INFO_DIR) / task_name / "img_info.json",
        Path(IMAGE_INFO_DIR) / safe_name / "img_info.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[-1]


def _task_image_folder(task_name: str) -> Path:
    safe_name = _safe_task_name(task_name)
    candidates = [Path(IMAGE_SVAE_DIR) / task_name, Path(IMAGE_SVAE_DIR) / safe_name]
    for path in candidates:
        if path.exists():
            return path
    return candidates[-1]


def _task_qa_path(task_name: str) -> Path:
    safe_name = _safe_task_name(task_name)
    return Path(QA_DIR) / f"{safe_name}_qa.json"


def parse_model_json(text: str):
    if text is None:
        raise JSONParseError("Input text is None")

    s = text.strip().lstrip("\ufeff")
    try:
        return json.loads(s)
    except Exception:
        pass

    fence_pat = re.compile(r"```(?:json|JSON)?\s*([\s\S]*?)\s*```", re.MULTILINE)
    m = fence_pat.search(s)
    last_err = None
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception as e:
            last_err = e

    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\{\[]", s):
        try:
            obj, _ = decoder.raw_decode(s[match.start():].lstrip())
            if isinstance(obj, (dict, list)):
                return obj
        except Exception:
            continue

    hint = "Could not parse JSON from the response."
    if last_err is not None:
        hint += f" Last fenced-block parse error: {repr(last_err)}"
    raise JSONParseError(hint)


def load_resize_image_path_max1024(image_path: str, max_side: int = 1024) -> tuple[bytes, str]:
    with Image.open(image_path) as im:
        if im.mode in ("P", "LA"):
            im = im.convert("RGBA")

        w, h = im.size
        scale = min(max_side / max(w, h), 1.0)
        if scale < 1.0:
            im = im.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

        has_alpha = (im.mode in ("RGBA", "LA")) or ("transparency" in im.info)
        out = io.BytesIO()
        if has_alpha:
            im.save(out, format="PNG", optimize=True)
            return out.getvalue(), "image/png"

        if im.mode != "RGB":
            im = im.convert("RGB")
        im.save(out, format="JPEG", quality=92, optimize=True)
        return out.getvalue(), "image/jpeg"


def _build_generation_clients() -> list[tuple[str, AsyncOpenAI, str]]:
    clients: list[tuple[str, AsyncOpenAI, str]] = []
    if GEMINI_API_KEY:
        clients.append(
            (
                "gemini",
                AsyncOpenAI(
                    api_key=GEMINI_API_KEY,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                ),
                DEFAULT_GEMINI_GENERATION_MODEL,
            )
        )
    if OPENAI_API_KEY:
        clients.append(("gpt", AsyncOpenAI(api_key=OPENAI_API_KEY), DEFAULT_GPT_GENERATION_MODEL))

    if not clients:
        raise RuntimeError("OPENAI_API_KEY or GOOGLE_AI_STUDIO_KEY is required for QA generation.")
    if len(clients) < 2:
        raise RuntimeError(
            "Both OPENAI_API_KEY and GOOGLE_AI_STUDIO_KEY are required to alternate GPT and Gemini generation."
        )
    return clients


QA_GENERATION_SYSTEM_PROMPT = """
You are a Multimodal Evaluation QA Generation Expert.

Generate exactly ONE high-quality QA pair based on the given image and metadata.

Inputs:
- task_description: the capability to evaluate
- qa_example: examples used only for style and difficulty alignment
- question_type: required question format
- image: the authoritative evidence source
- title and search_query: auxiliary clues only; they may be wrong
- available_tools: allowed visual tools for verification planning

Requirements:
1. The question must strictly evaluate task_description.
2. The answer must be uniquely determined from the image.
3. Do not rely on title or search_query as ground truth.
4. Do not fabricate visual evidence.
5. The preferred_tool_sequence must only use tools in available_tools.
6. Tool dependencies must be respected:
   - detect_and_seg after recognize
   - see_attribute after detect_and_seg
   - ocr after detect_and_seg

Output exactly one JSON object and no extra text:
{
  "question_type": "<given question_type>",
  "question": "<question text>",
  "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
  "answer": "<A/B/C/D, True/False, or short text>",
  "solution_plan": ["Step 1: ...", "Step 2: ..."],
  "preferred_tool_sequence": [
    {
      "tool": "string",
      "purpose": "string",
      "inputs": "string",
      "expected_outputs": "string"
    }
  ]
}

If question_type is not multiple-choice, set options to null.
"""


async def _generate_one_qa(
    client_entry: tuple[str, AsyncOpenAI, str],
    image_info: Dict[str, Any],
    task: Dict[str, Any],
    task_name: str,
    question_type: str,
) -> Dict[str, Any]:
    provider, client, model = client_entry
    task_description = task.get("task_description", "")
    qa_example = task.get("task_qa_example") or task.get("qa_example") or ""
    title = image_info.get("img_title") or image_info.get("title") or ""
    search_query = image_info.get("search_query") or image_info.get("query") or ""
    image_path = image_info.get("img_path") or image_info.get("path")
    if not image_path:
        raise ValueError("image_info is missing img_path/path.")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image_bytes, mime_type = await asyncio.to_thread(load_resize_image_path_max1024, image_path, 1024)
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    user_text = json.dumps(
        {
            "task_name": task_name,
            "task_description": task_description,
            "question_type": question_type,
            "qa_example": qa_example,
            "title": title,
            "search_query": search_query,
            "available_tools": AVAILABLE_VISUAL_TOOLS,
        },
        ensure_ascii=False,
    )

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": QA_GENERATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
                    {"type": "text", "text": user_text},
                ],
            },
        ],
    }
    if provider == "gemini":
        kwargs["reasoning_effort"] = "low"
    else:
        kwargs["response_format"] = {"type": "json_object"}

    resp = await client.chat.completions.create(**kwargs)
    result = parse_model_json(resp.choices[0].message.content)
    if not isinstance(result, dict):
        raise ValueError("QA generation result must be a JSON object.")
    result["img_path"] = image_path
    result["generator_provider"] = provider
    result["generator_model"] = model
    return result


async def _safe_generate_one_qa(
    client_entry: tuple[str, AsyncOpenAI, str],
    img_name: str,
    image_info: Dict[str, Any],
    question_type: str,
    task: Dict[str, Any],
    task_name: str,
    sem: asyncio.Semaphore,
) -> Tuple[str, Dict[str, Any]]:
    async with sem:
        try:
            return img_name, await _generate_one_qa(client_entry, image_info, task, task_name, question_type)
        except Exception as e:
            return img_name, {"error": repr(e), "img_path": image_info.get("img_path") or image_info.get("path")}


@mcp.tool()
async def generate_qa_pairs(task_name: str, json_path: str, img_concurrency: int = 5):
    """
    Generate QA pairs for one task.

    Parameters:
    - task_name: Task name in the benchmark summary JSON.
    - json_path: Path to the benchmark summary JSON.
    - img_concurrency: Maximum concurrent image-level generation calls.

    Returns:
    - dict: generation summary with output QA JSON path.
    """
    data = _load_json(json_path)
    resolved_task_name, task = _task_info(data, task_name)
    safe_name = _safe_task_name(resolved_task_name)
    question_type = _question_type(data)

    img_info_path = _task_image_info_path(resolved_task_name)
    if not img_info_path.exists():
        raise FileNotFoundError(f"Image metadata not found: {img_info_path}")

    image_info_json = _load_json(img_info_path.as_posix())
    if not isinstance(image_info_json, dict):
        raise ValueError(f"Image metadata must be a JSON object: {img_info_path}")

    task_img_folder = _task_image_folder(resolved_task_name)
    if task_img_folder.exists():
        img_names = [n for n in os.listdir(task_img_folder) if n in image_info_json]
    else:
        img_names = list(image_info_json.keys())

    qa_save_path = _task_qa_path(resolved_task_name)
    clients = _build_generation_clients()
    client_cycle = itertools.cycle(clients)
    sem = asyncio.Semaphore(max(1, img_concurrency))

    coros = [
        _safe_generate_one_qa(
            client_entry=next(client_cycle),
            img_name=img_name,
            image_info=image_info_json[img_name],
            question_type=question_type,
            task=task,
            task_name=resolved_task_name,
            sem=sem,
        )
        for img_name in img_names
    ]

    qa_dir: Dict[str, Dict[str, Any]] = {}
    pbar = tqdm(total=len(coros), desc=f"[QA] {safe_name}", leave=False, dynamic_ncols=True)
    for fut in asyncio.as_completed(coros):
        img_name, qa_result = await fut
        qa_dir[img_name] = qa_result
        pbar.update(1)
    pbar.close()

    _dump_json(qa_save_path, qa_dir)
    errors = sum(1 for item in qa_dir.values() if item.get("error"))
    return {
        "task_name": resolved_task_name,
        "image_metadata_path": img_info_path.as_posix(),
        "qa_path": qa_save_path.as_posix(),
        "total": len(qa_dir),
        "success": len(qa_dir) - errors,
        "errors": errors,
    }


@mcp.tool()
def get_task_list(json_path: str):
    """
    Get Task List from Summary JSON

    Description:
    - Read the summary JSON file from the specified path.
    - Extract and return the list of task names under "finest_granularity_tasks".

    Parameters:
    - json_path (str): Path to the summary JSON file.

    Returns:
    - list: List of task names in the summary's finest_granularity_tasks.
    """

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            task_list = list(data["finest_granularity_tasks"].keys())
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON file: {e}")

    return task_list


if __name__ == "__main__":
    mcp.run(transport="sse")
