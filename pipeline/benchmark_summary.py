import json
from typing import Any
import random
from openai import OpenAI
from typing import Any, Dict, List, Optional
import base64
from project_config import OPENAI_API_KEY, PAPER_PATH, DATASET_PATH, BENCHMARK_SUMMARY_PATH


client = OpenAI(api_key=OPENAI_API_KEY)


def paper_analyse(paper_path: str) -> str:
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

    with open(paper_path, "rb") as f:
        uploaded_file = client.files.create(
            file=f,
            purpose="assistants",
        )


    file_id = uploaded_file.id

    resp = client.responses.create(
        model="gpt-5-mini",
        instructions=system_prompt,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": user_query,
                    },
                    {
                        "type": "input_file",
                        "file_id": file_id,
                    },
                ],
            }
        ],
    )


    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text

    try:   
        first_item = resp.output[0]
        first_content = first_item.content[0]
        if getattr(first_content, "type", "") == "output_text":
            return first_content.text
        
    except Exception:
        return json.dumps(resp.model_dump(mode="json"))
    

def visual_content_constraint(examples):
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
    
    resp = client.responses.create(
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


def visual_style_constraint(examples):
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
    
    resp = client.responses.create(
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


def ocr_dependency_constraint(examples):
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
    
    resp = client.responses.create(
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


def spatial_relation_constraint(examples):
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
    
    resp = client.responses.create(
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


def external_knowledge_constraint(examples):
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
    
    resp = client.responses.create(
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


def implicit_constraints(dataset: List[Dict[str, Any]],
        n: int = 6,
        *,
        seed: Optional[int] = 5,
    ):
    rng = random.Random(seed)
    n_used = min(n, len(dataset))
    sampled_items = rng.sample(dataset, k=n_used) if n_used < len(dataset) else list(dataset)
    constraints = {}
    constraints["visual_content"] = visual_content_constraint(sampled_items)
    constraints["visual_style"] = visual_style_constraint(sampled_items)
    constraints["ocr_dependency"] = ocr_dependency_constraint(sampled_items)
    constraints["spatial_relation"] = spatial_relation_constraint(sampled_items)
    constraints["external_knowledge"] = external_knowledge_constraint(sampled_items)
    return constraints


if __name__ == "__main__":
    with open(DATASET_PATH, 'r') as f:
        dataset_images = json.load(f)
    
    summary_result = json.loads(paper_analyse(PAPER_PATH))
    for task_name in summary_result["finest_granularity_tasks"]:
        dataset_task = dataset_images[task_name]
        summary_result["finest_granularity_tasks"]["constraints"] = implicit_constraints(dataset_task)
    
    with open(BENCHMARK_SUMMARY_PATH, 'w') as f:
        json.dump(summary_result, f, indent=4)
