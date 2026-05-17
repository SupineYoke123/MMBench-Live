import os
from tqdm import tqdm
import re
from openai import AsyncOpenAI
import base64
from PIL import Image
import io
from typing import Dict, Any, Tuple
import json
import asyncio
from project_config import BENCHMARK_SUMMARY_PATH, IMAGE_SVAE_DIR, IMAGE_INFO_DIR, QA_DIR, OPENAI_API_KEY, GEMINI_API_KEY
import itertools

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
gemini_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

clients_list = [
    ("gpt", openai_client, "gpt-5o-mini"),
    ("gemini", gemini_client, "gemini-3-flash-preview"),
]
client_cycle = itertools.cycle(clients_list)


class JSONParseError(ValueError):
    pass


def parse_model_json(text: str):
    """
    Parse a model response into JSON (dict or list).

    Handles:
      1) raw JSON
      2) fenced code blocks: ```json ... ``` or ``` ... ```
      3) JSON embedded in surrounding text

    Raises:
      JSONParseError with details if parsing fails.
    """
    if text is None:
        raise JSONParseError("Input text is None")

    s = text.strip()
    s = s.lstrip("\ufeff")

    try:
        return json.loads(s)
    except Exception:
        pass


    fence_pat = re.compile(r"```(?:json|JSON)?\s*([\s\S]*?)\s*```", re.MULTILINE)
    m = fence_pat.search(s)
    if m:
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception as e:
            last_err = e
    else:
        last_err = None

    decoder = json.JSONDecoder()
    starts = [m.start() for m in re.finditer(r"[\{\[]", s)]
    for i in starts:
        substr = s[i:].lstrip()
        try:
            obj, end = decoder.raw_decode(substr)
            if isinstance(obj, (dict, list)):
                return obj
        except Exception:
            continue
    hint = "Could not parse JSON from the response."
    if last_err is not None:
        hint += f" Last fenced-block parse error: {repr(last_err)}"
    raise JSONParseError(hint)

def load_resize_image_path_max1024(image_path: str, max_side: int = 1024) -> tuple[bytes, str]:
    """
    Load image from path and resize so that max(width, height) <= max_side (keep aspect).
    Never upscales. Returns (image_bytes, mime_type).
    Uses PNG if alpha/transparency exists, else JPEG.
    """
    with Image.open(image_path) as im:
        if im.mode in ("P", "LA"):
            im = im.convert("RGBA")

        w, h = im.size
        scale = min(max_side / max(w, h), 1.0)
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            im = im.resize((new_w, new_h), Image.Resampling.LANCZOS)

        has_alpha = (im.mode in ("RGBA", "LA")) or ("transparency" in im.info)

        out = io.BytesIO()
        if has_alpha:
            im.save(out, format="PNG", optimize=True)
            return out.getvalue(), "image/png"
        else:
            if im.mode != "RGB":
                im = im.convert("RGB")
            im.save(out, format="JPEG", quality=92, optimize=True)
            return out.getvalue(), "image/jpeg"


async def qa_generation(image_info, task, question_type):
    system_prompt = """
        You are a Multimodal Evaluation QA Generation Expert.

        Your task is to generate exactly ONE high-quality Question–Answer (QA) pair based on the given image and its metadata, strictly following the task definition.

        You will receive the following inputs:
        - task_description: A description defining the target capability to be evaluated (must be strictly followed)
        - qa_example: One or more example QA pairs, used ONLY to align style, structure, and difficulty
        - question_types: The allowed question formats (e.g., multiple-choice, true/false, open-ended)
        - image: The input image (primary and authoritative information source)
        - title: The image title text (may be noisy or inaccurate)
        - search_query: The search query used to retrieve the image (may be noisy or inaccurate)
        - available_tools: A list of available tool names that can be used for image analysis and verification

        ---
        Available Tools Description

        The caller will provide a list of `available_tools`, where each element is a tool name chosen from the following set:

        - `recognize`  
        Generates a global description of the image and identifies the main objects present in the image.

        - `detect_and_seg`  
        Detects and segments objects in the image, returning bounding boxes and segmentation masks for each identified object.  
        Note: This tool MUST be run after `recognize` to ensure object recognition results are available.

        - `see_depth`  
        Generates depth information for the image, which can be used to reason about relative distances, front–back relationships, or spatial layering.

        - `see_attribute`  
        Extracts attributes (e.g., color, shape, material, pose, state) for each detected object.  
        Note: This tool MUST be run after `detect_and_seg` to ensure object-level regions are available.

        - `ocr`  
        Extracts textual content from the image.  
        Note: This tool MUST be run after `detect_and_seg` to ensure text regions are properly localized.

        You MUST reference ONLY these tool names in any tool planning.  
        You MUST NOT invent, rename, or assume the existence of any other tools.

        ---
        Objective

        Generate one and only one QA pair that:
        - Strictly evaluates the capability defined in task_description
        - Depends on the image as the primary evidence source
        - May reference title and/or search_query as auxiliary cues, but MUST NOT rely on them as ground truth
        - Cannot be correctly answered without analyzing the image

        ---
        Generation Requirements (MUST FOLLOW)

        1. Question Type  
        - Strictly conform to the given question_type.

        2. Strict Task Alignment  
        - The question must assess ONLY the ability described in task_description.

        3. Image Dependency  
        - The answer must be verifiable from visual evidence in the image.
        - If title or search_query conflicts with the image, the image must take precedence.

        4. Uniqueness & Determinism  
        - The question must have exactly one correct answer.

        5. Solution Plan (Reasoning Guidance)  
        - Provide a clear solution_plan explaining how to derive the answer.
        - The solution_plan may:
        - Describe step-by-step visual reasoning based on the image, and/or
        - Reference title or search_query as a potential clue, explicitly noting that they may be inaccurate.
        - If title/search_query is used, the solution_plan MUST explain how the image is used to verify or correct them.
        - Do NOT fabricate visual evidence not present in the image.

        6. Tool-based Verification Planning (MUST FOLLOW)
        - Based on the solution_plan, you MUST propose a tool usage plan that could be used to verify or support the reasoning.
        - You MUST select tools ONLY from the provided `available_tools` list.
        - The proposed tool sequence MUST respect tool dependencies (e.g., `detect_and_seg` after `recognize`).
        - Tool usage is for verification or structured analysis only; tools do NOT override visual evidence from the image.

        7. Reference to qa_example  
        - You MAY follow the structure, phrasing style, and difficulty level demonstrated in qa_example.
        - You MUST NOT copy content verbatim or generate more than one QA pair.

        ---
        Output Format Requirements

        - Output only ONE JSON object
        - Do NOT include any extra text
        - The JSON structure MUST strictly follow the schema below:

        {
            "question_type": "<given question_type>",
            "question": "<the generated question text>",
            "options": ["A. ...", "B. ...", "C. ...", "D. ..."],  // required ONLY if question_type is multiple_choice; otherwise null
            "answer": "<A/B/C/D or True/False or a short text answer>",
            "solution_plan": [
                "Step 1: ...",
                "Step 2: ...",
                "Step 3: ..."
            ],
            "preferred_tool_sequence": [
                {
                "tool": "string, tool name, must be from the available_tools list",
                "purpose": "string, a brief description of why this tool is used in the reasoning or verification process",
                "inputs": "string, a description of the key information passed to the tool (e.g., entire image, specific region, object category, mask, etc.)",
                "expected_outputs": "string, a description of what information or signal is expected from this tool"
                }
            ]
        }

    """

    system_prompt = """
        You are a Multimodal Evaluation QA Generation Expert.

        Your task is to generate exactly ONE high-quality Question–Answer (QA) pair based on the given image and its metadata, strictly following the task definition.

        You will receive the following inputs:
        - task_description: A description defining the target capability to be evaluated (must be strictly followed)
        - qa_example: One or more example QA pairs, used ONLY to align style, structure, and difficulty
        - question_types: The allowed question formats (e.g., multiple-choice, true/false, open-ended)
        - image: The input image (primary and authoritative information source)
        - title: The image title text (may be noisy or inaccurate)
        - search_query: The search query used to retrieve the image (may be noisy or inaccurate)

        
        Objective

        Generate one and only one QA pair that:
        - Strictly evaluates the capability defined in task_description
        - Depends on the image as the primary evidence source
        - May reference title and/or search_query as auxiliary cues, but MUST NOT rely on them as ground truth
        - Cannot be correctly answered without analyzing the image

        ---
        Generation Requirements (MUST FOLLOW)

        1. Question Type  
        - Strictly conform to the given question_type.

        2. Strict Task Alignment  
        - The question must assess ONLY the ability described in task_description.

        3. Image Dependency  
        - The answer must be verifiable from visual evidence in the image.
        - If title or search_query conflicts with the image, the image must take precedence.

        4. Uniqueness & Determinism  
        - The question must have exactly one correct answer.

        5. Reference to qa_example  
        - You MAY follow the structure, phrasing style, and difficulty level demonstrated in qa_example.
        - You MUST NOT copy content verbatim or generate more than one QA pair.

        ---
        Output Format Requirements

        - Output only ONE JSON object
        - Do NOT include any extra text
        - The JSON structure MUST strictly follow the schema below:

        {
            "question_type": "<given question_type>",
            "question": "<the generated question text>",
            "options": ["A. ...", "B. ...", "C. ...", "D. ..."],  // required ONLY if question_type is multiple_choice; otherwise null
            "answer": "<A/B/C/D or True/False or a short text answer>",
        }

    """

    provider, client, model_name = next(client_cycle)
    task_description = task["task_description"]
    qa_example = task["task_qa_example"]
    question_types = question_type
    title = image_info["img_title"]
    search_query = image_info["search_query"]
    image_path = image_info["img_path"]

    image_bytes, mime_type = await asyncio.to_thread(load_resize_image_path_max1024, image_path, 1024)
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    user_text = (
        f"task description: {task_description}\n"
        f"question types: {question_types}\n"
        f"qa example: {qa_example}\n"
        f"title: {title}\n"
        f"search query: {search_query}\n"
    )

    request_kwargs = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": user_text},
                ],
            },
        ],
    }
    if provider == "gemini":
        request_kwargs["reasoning_effort"] = "low"
    else:
        request_kwargs["response_format"] = {"type": "json_object"}

    resp = await client.chat.completions.create(**request_kwargs)

    text = resp.choices[0].message.content
    result = parse_model_json(text)
    if isinstance(result, dict):
        result["generator_provider"] = provider
        result["generator_model"] = model_name
    return result


async def _safe_qa_generation(
    img_name: str,
    image_info: Dict[str, Any],
    question_type: str,
    task: Dict[str, Any],
    sem: asyncio.Semaphore,
) -> Tuple[str, Dict[str, Any]]:
    async with sem:
        try:
            qa_result = await qa_generation(image_info, task, question_type)
            qa_result["img_path"] = image_info.get("img_path")
            return img_name, qa_result
        except Exception as e:
            return img_name, {"error": repr(e), "img_path": image_info.get("img_path")}

async def process_one_task(
    key: str,
    task: Dict[str, Any],
    img_folder_path: str,
    json_folder_path: str,
    qa_folder_path: str,
    question_type: str,
    img_concurrency: int = 16,
):
    task_name = "_".join(key.split())
    task_img_folder = os.path.join(img_folder_path, task_name)
    json_path = f"{json_folder_path}/{task_name}/img_info.json"
    qa_save_path = os.path.join(qa_folder_path, f"{task_name}_qa.json")

    with open(json_path, "r") as f:
        image_info_json = json.load(f)

    img_names = [n for n in os.listdir(task_img_folder) if n in image_info_json]
    sem = asyncio.Semaphore(img_concurrency)

    coros = [
        _safe_qa_generation(img_name, image_info_json[img_name], question_type, task, sem)
        for img_name in img_names
    ]

    qa_dir = {}

    pbar = tqdm(
        total=len(coros),
        desc=f"[IMG] {task_name}",
        leave=False,
        dynamic_ncols=True,
    )

    for fut in asyncio.as_completed(coros):
        img_name, qa_result = await fut
        qa_dir[img_name] = qa_result
        pbar.update(1)

    pbar.close()

    os.makedirs(os.path.dirname(qa_save_path), exist_ok=True)
    with open(qa_save_path, "w") as f:
        json.dump(qa_dir, f, indent=4, ensure_ascii=False)

    return task_name, len(img_names)


async def process_all_tasks_parallel(
    data: Dict[str, Any],
    img_folder_path: str,
    json_folder_path: str,
    qa_folder_path: str,
    question_type: str,
    task_concurrency: int = 4,
    img_concurrency: int = 5,
):
    tasks_dict = data["raw_tool_output"]["finest_granularity_tasks"]

    task_sem = asyncio.Semaphore(task_concurrency)

    async def _wrapped(key: str, task: Dict[str, Any]):
        async with task_sem:
            return await process_one_task(
                key, task,
                img_folder_path=img_folder_path,
                json_folder_path=json_folder_path,
                qa_folder_path=qa_folder_path,
                question_type=question_type,
                img_concurrency=img_concurrency,
            )

    task_items = list(tasks_dict.items())
    task_pbar = tqdm(total=len(task_items), desc="[TASK]", dynamic_ncols=True)

    async def _wrapped_with_pbar(k, t):
        try:
            return await _wrapped(k, t)
        finally:
            task_pbar.update(1)

    results = await asyncio.gather(*[_wrapped_with_pbar(k, t) for k, t in task_items])
    task_pbar.close()
    return results


async def main():
    with open(BENCHMARK_SUMMARY_PATH, 'r') as f:
        data = json.load(f)

    question_type = data["raw_tool_output"]["question_types"]
    result = await process_all_tasks_parallel(data, IMAGE_SVAE_DIR, IMAGE_INFO_DIR, QA_DIR, question_type)

