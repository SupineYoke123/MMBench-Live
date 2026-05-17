import os
import json
from typing import Any
import random
from mcp.server.fastmcp import FastMCP
from openai import AsyncOpenAI
from typing import Any, Dict, List, Optional
import base64
from project_config import BENCHMARK_SUMMARY_AGENT_MCP_PORT, OPENAI_API_KEY

mcp = FastMCP("paper_analyse_tools", port=BENCHMARK_SUMMARY_AGENT_MCP_PORT)


client = AsyncOpenAI(api_key=OPENAI_API_KEY)


@mcp.tool()
async def paper_analyse(paper_path: str, save_path: str) -> str:
    """
    Paper Structural Parsing Tool

    Description:
    - Analyze a research paper to extract its evaluation system.
    - Generate a structured JSON with the following keys:
      evaluation_purpose, question_types, evaluation_dimensions,
      task_hierarchy, finest_granularity_tasks.
    - Save the JSON result to the specified `save_path`.

    Parameters:
    - paper_path (str): Path to the paper file to analyze.
    - save_path (str): Path to save the resulting JSON file.

    Returns:
    - str: The path where the JSON result is saved.
    """

    system_prompt = """
        You are a Paper Structural Parsing Agent.
        Your task is to construct a complete, multi-level, structured representation of the evaluation system described in a research paper.
        The final output must be a JSON object with the following fixed keys:
        {
            "evaluation_purpose": "",
            "question_types": [],
            "evaluation_dimensions": [],
            "task_hierarchy": {},
            "finest_granularity_tasks": {}
        }
        Field Requirements
        1. evaluation_purpose (string)
        The explicitly stated evaluation purpose.

        2. question_types (array of string)
        All listed question types or ability types.

        3. evaluation_dimensions (array of string)
        All Level-1 evaluation dimensions.

        4. task_hierarchy (object, hierarchical tree)
        Reconstruct the full multi-level hierarchy of tasks and subtasks.
        Abstract example (no concrete task names):
        "task_hierarchy": {
            "<Dimension_1>": {
                "<Subtask_Level_2_A>": {
                    "<Subtask_Level_3_A>": [...],
                    "<Subtask_Level_3_B>": { ... }
                },
                "<Subtask_Level_2_B>": [...]
            },
            "<Dimension_2>": {
                "<Subtask_Level_2_C>": [...]
            }
        }
        Requirements:
        1. Must match the paper’s structure exactly
        2. Subtasks may be nested structures or arrays
        3. No invented task names or fictional structures

        5. finest_granularity_tasks (object)
        For every finest-granularity task, create an entry using the fixed schema:
        "finest_granularity_tasks": {
            "<task_name>": {
                "task_description": "",
                "data_construction_method": "",
                "data_sources": "",
                "data_characteristics": "",
                "task_qa_example": "",
                "other_attributes": ""
            }
        }
        Rules:
        1. Only include information explicitly stated in the paper
        2. Leave fields empty if not provided
        3. No inference or hallucination

        Global Requirements
        1. Output must be valid JSON only
        2. No extra commentary or explanation outside the JSON
        3. All fixed keys must appear exactly
        4. No additional keys may be added
    """

    user_query = "Please help me analyze this evaluation-domain paper."

    try:
        with open(paper_path, "rb") as f:
            uploaded_file = await client.files.create(
                file=f,
                purpose="assistants",
            )
    except Exception as e:
        raise e

    file_id = uploaded_file.id

    try:
        resp = await client.responses.create(
            model="gpt-5-mini",
            instructions=system_prompt,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_query},
                        {"type": "input_file", "file_id": file_id},
                    ],
                }
            ],
        )
    except Exception as e:
        raise e

    result_json = None
    if hasattr(resp, "output_text") and resp.output_text:
        try:
            result_json = json.loads(resp.output_text)
        except Exception:
            result_json = resp.output_text
    else:
        try:
            first_item = resp.output[0]
            first_content = first_item.content[0]
            if getattr(first_content, "type", "") == "output_text":
                result_json = json.loads(first_content.text)
            else:
                result_json = resp.model_dump(mode="json")
        except Exception:
            result_json = resp.model_dump(mode="json")

    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result_json, f, indent=4, ensure_ascii=False)
    except Exception as e:
        raise RuntimeError(f"Failed to save JSON to {save_path}: {e}")

    return result_json
  

async def visual_content_constraint(examples):
    system_prompt = """
        You are an implicit constraint analysis expert.
        Given a set of evaluation data samples, analyze the implicit constraints on visual content and scene distribution that are consistently followed by the dataset but not explicitly described in the paper.
        In your analysis, explicitly examine the following dimensions:
        1. subject types appearing in the images (e.g., people, animals, vehicles, everyday objects, document interfaces, charts, etc.);
        2. scene category distributions (e.g., indoor vs. outdoor, natural vs. urban, daily-life scenes vs. professional or specialized environments);
        3. the number of subjects and their common composition patterns (e.g., single-object focus, multi-object relations, subject–tool or subject–background pairings).

        Your output MUST follow these rules:
        - The answer must be a LIST.
        - Each list item corresponds to ONE distinct implicit constraint.
        - Each constraint should describe a single, atomic spatial or geometric assumption.
        - Different constraints MUST NOT overlap or repeat the same assumption in different wording.
        - If two observations refer to the same underlying spatial rule, they must be merged into one constraint.
        - Do NOT include examples inside the constraint text.
        - Contains up to 5 constraints.
        - Include all following dimensions as much as possible.
        For each constraint, clearly indicate whether it is CENTRAL or PERIPHERAL to task resolution.

        Summarize the stable visual content patterns observed across samples as implicit constraints that define the default evaluation scope.
        Base your conclusions strictly on recurring evidence in the samples and do not introduce unsupported assumptions.
    """
    user_input = []
    for item in examples:
        img_path = item['img_path']
        question= item['question']
        choices = item['choices']
        answer = item['answer']
        with open(img_path, "rb") as image_file:
            b64_image = base64.b64encode(image_file.read()).decode("utf-8")
        user_input.append(
            {"type": "input_image", "image_url": f"data:image/png;base64,{b64_image}"})
        user_input.append(
            {"type": "input_text", "text": f"question: {question}\nchoices: {choices}\nanswer: {answer}"}
        )
    
    resp = await client.responses.create(
        model="gpt-5-mini",
        instructions=system_prompt,
        input=[
            {
                "role": "user",
                "content": user_input
            }
        ],
    )
    return str(resp.output_text)


async def visual_style_constraint(examples):
    system_prompt = """
        You are an implicit constraint analysis expert.
        Analyze the implicit constraints on visual style and image source that are assumed by the evaluation dataset but not specified in the paper.
        In particular, examine the following dimensions:
        1. image source types (e.g., real-world photographs, mobile phone images, screenshots, web or application interfaces, posters, illustrations, diagrams);
        2. overall visual style (e.g., photorealistic vs. non-photorealistic, diagrammatic vs. natural images, presence of UI or layout elements);
        3. image quality and composition tendencies (e.g., clarity, blur, cropping patterns, compression artifacts, noise).

        Your output MUST follow these rules:
        - The answer must be a LIST.
        - Each list item corresponds to ONE distinct implicit constraint.
        - Each constraint should describe a single, atomic spatial or geometric assumption.
        - Different constraints MUST NOT overlap or repeat the same assumption in different wording.
        - If two observations refer to the same underlying spatial rule, they must be merged into one constraint.
        - Do NOT include examples inside the constraint text.
        - Contains up to 5 constraints.
        - Include all following dimensions as much as possible.
        For each constraint, clearly indicate whether it is CENTRAL or PERIPHERAL to task resolution.

        Summarize the recurring or implicitly accepted image styles and source characteristics as implicit constraints.
        Avoid subjective judgments and rely only on stable visual patterns consistently present in the samples.
    """
    user_input = []
    for item in examples:
        img_path = item['img_path']
        question= item['question']
        choices = item['choices']
        answer = item['answer']
        with open(img_path, "rb") as image_file:
            b64_image = base64.b64encode(image_file.read()).decode("utf-8")
        user_input.append(
            {"type": "input_image", "image_url": f"data:image/png;base64,{b64_image}"})
        user_input.append(
            {"type": "input_text", "text": f"question: {question}\nchoices: {choices}\nanswer: {answer}"}
        )
    
    resp = await client.responses.create(
        model="gpt-5-mini",
        instructions=system_prompt,
        input=[
            {
                "role": "user",
                "content": user_input
            }
        ],
    )
    return str(resp.output_text)


async def ocr_dependency_constraint(examples):
    system_prompt = """
        You are an implicit constraint analysis expert.
        Analyze the implicit constraints related to text readability and OCR dependency in the evaluation tasks, which may not be explicitly described in the paper.
        In your analysis, focus on the following dimensions:
        1. whether solving the task requires reading text in images (no text required, text as auxiliary information, or text as a necessary condition);
        2. the carriers of textual information (e.g., road signs, menus, packaging, documents, web interfaces, tables, or chart annotations);
        3. the form of textual content (e.g., natural language, numbers, units, symbols, formulas, or mixed forms).
        
        Your output MUST follow these rules:
        - The answer must be a LIST.
        - Each list item corresponds to ONE distinct implicit constraint.
        - Each constraint should describe a single, atomic spatial or geometric assumption.
        - Different constraints MUST NOT overlap or repeat the same assumption in different wording.
        - If two observations refer to the same underlying spatial rule, they must be merged into one constraint.
        - Do NOT include examples inside the constraint text.
        - Contains up to 5 constraints.
        - Include all following dimensions as much as possible.
        For each constraint, clearly indicate whether it is CENTRAL or PERIPHERAL to task resolution.
        
        Determine the role that textual information plays in task completion and summarize stable text dependency patterns as implicit constraints.
        Do not introduce OCR or text understanding assumptions that are not clearly supported by the samples.
    """
    user_input = []
    for item in examples:
        img_path = item['img_path']
        question= item['question']
        choices = item['choices']
        answer = item['answer']
        with open(img_path, "rb") as image_file:
            b64_image = base64.b64encode(image_file.read()).decode("utf-8")
        user_input.append(
            {"type": "input_image", "image_url": f"data:image/png;base64,{b64_image}"})
        user_input.append(
            {"type": "input_text", "text": f"question: {question}\nchoices: {choices}\nanswer: {answer}"}
        )
    
    resp = await client.responses.create(
        model="gpt-5-mini",
        instructions=system_prompt,
        input=[
            {
                "role": "user",
                "content": user_input
            }
        ],
    )
    return str(resp.output_text)


async def spatial_relation_constraint(examples):
    system_prompt = """
        You are an implicit constraint analysis expert.
        Analyze whether the evaluation dataset implicitly requires spatial relation understanding or geometric reasoning, and whether such requirements are omitted in the paper description.
        In particular, examine the following dimensions:
        1. relative spatial relations (e.g., left/right, above/below, front/behind, inside/outside, near/far);
        2. orientation and pose information (e.g., facing direction, rotation angles, viewpoint changes, mirror symmetry);
        3. visibility and occlusion conditions (e.g., partial occlusion, overlap, reflections, transparency).

        Your output MUST follow these rules:
        - The answer must be a LIST.
        - Each list item corresponds to ONE distinct implicit constraint.
        - Each constraint should describe a single, atomic spatial or geometric assumption.
        - Different constraints MUST NOT overlap or repeat the same assumption in different wording.
        - If two observations refer to the same underlying spatial rule, they must be merged into one constraint.
        - Do NOT include examples inside the constraint text.
        - Contains up to 5 constraints.
        - Include all following dimensions as much as possible.
        For each constraint, clearly indicate whether it is CENTRAL or PERIPHERAL to task resolution.

        Determine whether these spatial or geometric factors are central or peripheral to task resolution and summarize recurring spatial assumptions as implicit constraints.
        Base your analysis strictly on spatial patterns that are consistently observable in the samples.
    """
    user_input = []
    for item in examples:
        img_path = item['img_path']
        question= item['question']
        choices = item['choices']
        answer = item['answer']
        with open(img_path, "rb") as image_file:
            b64_image = base64.b64encode(image_file.read()).decode("utf-8")
        user_input.append(
            {"type": "input_image", "image_url": f"data:image/png;base64,{b64_image}"})
        user_input.append(
            {"type": "input_text", "text": f"question: {question}\nchoices: {choices}\nanswer: {answer}"}
        )
    
    resp = await client.responses.create(
        model="gpt-5-mini",
        instructions=system_prompt,
        input=[
            {
                "role": "user",
                "content": user_input
            }
        ],
    )
    return str(resp.output_text)


async def external_knowledge_constraint(examples):
    system_prompt = """
        You are an implicit constraint analysis expert.
        Analyze whether the evaluation tasks implicitly depend on specific domain knowledge or external commonsense, beyond what can be inferred from visual observation alone.
        In your analysis, focus on the following dimensions:
        1. involvement of specialized domains (e.g., science, medicine, engineering, geography, history, culture);
        2. reliance on real-world commonsense or functional knowledge (e.g., object usage, rules, causal relations);
        3. the type of knowledge required (factual knowledge, conceptual knowledge, or reasoning-based knowledge).

        Your output MUST follow these rules:
        - The answer must be a LIST.
        - Each list item corresponds to ONE distinct implicit constraint.
        - Each constraint should describe a single, atomic spatial or geometric assumption.
        - Different constraints MUST NOT overlap or repeat the same assumption in different wording.
        - If two observations refer to the same underlying spatial rule, they must be merged into one constraint.
        - Do NOT include examples inside the constraint text.
        - Contains up to 5 constraints.
        - Include all following dimensions as much as possible.
        For each constraint, clearly indicate whether it is CENTRAL or PERIPHERAL to task resolution.

        Determine whether such knowledge dependencies are assumed as default requirements or optional capabilities, and summarize them as implicit constraints.
        Do not introduce domain or knowledge requirements that are not clearly supported by the samples.
    """
    user_input = []
    for item in examples:
        img_path = item['img_path']
        question= item['question']
        choices = item['choices']
        answer = item['answer']
        with open(img_path, "rb") as image_file:
            b64_image = base64.b64encode(image_file.read()).decode("utf-8")
        user_input.append(
            {"type": "input_image", "image_url": f"data:image/png;base64,{b64_image}"})
        user_input.append(
            {"type": "input_text", "text": f"question: {question}\nchoices: {choices}\nanswer: {answer}"}
        )
    
    resp = await client.responses.create(
        model="gpt-5-mini",
        instructions=system_prompt,
        input=[
            {
                "role": "user",
                "content": user_input
            }
        ],
    )
    return str(resp.output_text)


@mcp.tool()
def read_json_file(json_path: str) -> dict:
    """
    JSON File Reader Tool

    Description:
    - Read a JSON file from the specified path.
    - Parse the JSON content and return it as a Python dictionary.

    Parameters:
    - json_path (str): Path to the JSON file to read.

    Returns:
    - dict: Parsed JSON content.
    """

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON file: {e}")

    return data


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


async def implicit_constraints(dataset: List[Dict[str, Any]],
        n: int = 6,
        *,
        seed: Optional[int] = 5,
    ):
    rng = random.Random(seed)
    n_used = min(n, len(dataset))
    sampled_items = rng.sample(dataset, k=n_used) if n_used < len(dataset) else list(dataset)
    constraints = {}
    constraints["visual_content"] = await visual_content_constraint(sampled_items)
    constraints["visual_style"] = await visual_style_constraint(sampled_items)
    constraints["ocr_dependency"] = await ocr_dependency_constraint(sampled_items)
    constraints["spatial_relation"] = await spatial_relation_constraint(sampled_items)
    constraints["external_knowledge"] = await external_knowledge_constraint(sampled_items)
    return constraints


async def compress_constraints(task_description, task_constraints):
    system_prompt = """
        You are an expert in task-constraint refinement and abstraction.
        Given an original task description (task_description) and a set of detailed constraints organized across multiple categories (including but not limited to visual_content, visual_style, ocr_dependency, spatial_relation, and external_knowledge), your goal is to analyze, merge, and rewrite these constraints into a compact and principled form.
        
        Your core objective is to:
        1. Merge semantically overlapping or redundant constraints;
        2. Distill the most essential and universal requirements for the task;
        3. Preserve critical distinctions and category-specific identifying characteristics;
        4. Output a refined, clear list of “Core Constraints” containing no more than 3 items.

        Execution Guidelines:
        1. Input Analysis:
        Carefully read the original task description (task_description) and all constraint entries across the provided categories. Understand both their semantic intent and their role in supporting the task objective.
        2. Priority by Importance:
        Give precedence to constraints explicitly marked as CENTRAL.
        Constraints marked as PERIPHERAL should only be incorporated if they:
            serve as a necessary supplement to a CENTRAL constraint, or
            represent a critical exception or boundary condition that cannot be omitted.
        3. Merge Commonality Across Categories
        Identify recurring requirements that appear across multiple constraint categories (e.g., “prominent subject,” “simple background,” “no reliance on text”) and synthesize them into concise, higher-level statements rather than listing them separately.
        4. Eliminate Redundancy and Impractical Restrictions
        Remove repetitive descriptions or constraints that differ only in wording.
        Discard overly rigid, absolute, or impractical restrictions that are unnecessary for the core task of image type classification (e.g., strict rules on object count or exact spatial arrangements).
        5. Preserve Key Distinctions and Category-Specific Cues
        While abstracting and compressing the constraints, ensure that key identifying characteristics of specific image categories remain recoverable.
        For example, distinctive imaging geometry for medical scans (e.g., centered axial anatomy and scan-field symmetry) should be explicitly reflected, either within the generalized constraints or as clearly stated category-specific exceptions.

        Output Requirements:
        1. Output must be a numbered list titled “Core Constraints”;
        2. The list must contain no more than 3 items;
        3. Each item should be concise, high-level, and directly usable for downstream tasks such as data filtering, dataset curation, or judge-model evaluation.
    """
    user_input = [
        {"type": "input_text", "text": f"Task description: {task_description}\nTask constraints: {task_constraints}"}
    ]    
    resp = await client.responses.create(
        model="gpt-5-mini",
        instructions=system_prompt,
        input=[
            {
                "role": "user",
                "content": user_input
            }
        ],
    )
    return resp.output_text


@mcp.tool()
def fill_implicit_constraints(
    summary_path: str,
    dataset_path: str,
    summary_task_name: str,
    dataset_task_name: str
) -> str:
    """
    Fill implicit constraints for a specific task.

    Description:
    - Read the summary JSON and dataset JSON from the given paths.
    - For the specified summary_task_name and corresponding dataset_task_name,
      compute the implicit_constraints and insert it into the summary.
    - Save the updated summary JSON back to summary_path.
    - Return a short message indicating the task has been updated.

    Parameters:
    - summary_path (str): Path to the summary JSON file.
    - dataset_path (str): Path to the dataset JSON file.
    - summary_task_name (str): Task name in the summary JSON.
    - dataset_task_name (str): Corresponding task name in the dataset JSON.

    Returns:
    - str: Message confirming constraints have been saved.
    """
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary JSON not found: {summary_path}")
    with open(summary_path, "r", encoding="utf-8") as f:
        summary_result = json.load(f)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset JSON not found: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset_images = json.load(f)

    if summary_task_name not in summary_result["finest_granularity_tasks"]:
        raise KeyError(f"Summary task not found: {summary_task_name}")
    if dataset_task_name not in dataset_images:
        raise KeyError(f"Dataset task not found: {dataset_task_name}")

    constraints = implicit_constraints(dataset_images[dataset_task_name])
    task_description = summary_result["finest_granularity_tasks"][summary_task_name]["task_description"]
    core_constraints = compress_constraints(task_description, constraints)
    summary_result["finest_granularity_tasks"][summary_task_name]["constraints"] = constraints
    summary_result["finest_granularity_tasks"][summary_task_name]["core_constraints"] = core_constraints

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_result, f, indent=4, ensure_ascii=False)

    return f"Implicit constraints for task '{summary_task_name}' have been saved to {summary_path}."


if __name__ == "__main__":
    mcp.run(transport="sse")

    
