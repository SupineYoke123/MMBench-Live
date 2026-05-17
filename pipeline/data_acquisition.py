import os
import json
from openai import AsyncOpenAI
import requests
import base64
import asyncio
from tqdm import tqdm
import math
import aiohttp
import re, json
from project_config import IMAGE_SVAE_DIR, IMAGE_INFO_DIR, BENCHMARK_SUMMARY_PATH, OPENAI_API_KEY, SERPER_API_KEY, SERPER_URL, TOTAL_IMAGE_PER_TASK

client = AsyncOpenAI(api_key=OPENAI_API_KEY)


def strip_think_and_extract_json(text: str):
    s = (text or "").strip()
    s = re.sub(r"<think>.*?</think>\s*", "", s, flags=re.DOTALL).strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE).strip()
    try:
        return json.loads(s)
    except Exception:
        return s


async def check_image(task_description: str, task_constraints: str, image_path: str):
    system_prompt = """
        You are an Image Task Suitability Judge.
        Goal:
        Given a task description, a set of task constraints, and an image,
        decide whether the image is suitable for this task overall.
        Inputs: Task Description, Task Constraints, Image.

        Judging Principles:
        1) Evaluate whether the image is relevant to the task objective and broadly consistent with the task constraints.
        2) Apply a majority-satisfaction rule:
        - ACCEPT the image if it satisfies most of the core task constraints.
        - REJECT the image if it clearly deviates from the task objective or violates multiple important constraints.
        3) Minor or non-critical deviations are allowed as long as typical task resolution is not affected.
        4) Base all judgments strictly on visually observable evidence in the image.

        Output Requirements:
        Return a single JSON object ONLY, with the following format:
        {
            "decision": "ACCEPT" | "REJECT",
            "reasons": [
                "1–3 concise, non-overlapping reasons explaining the decision"
            ]
        }

        Additional Rules:
        - Each reason must be atomic and non-redundant.
        - Reasons should refer only to observable visual properties or content.
        - Keep each reason concise (preferably under 20 words).
    """
    user_input = []
    if 'http' in image_path:
        user_input = [
            {"type": "input_image", "image_url": image_path},
            {"type": "input_text", "text": f"task description: {task_description}\ntask constraints: {task_constraints}"}
        ]   
    else:
        with open(image_path, "rb") as image_file:
            b64_image = base64.b64encode(image_file.read()).decode("utf-8")
        user_input = [
            {"type": "input_image", "image_url": f"data:image/png;base64,{b64_image}"},
            {"type": "input_text", "text": f"task description: {task_description}\ntask constraints: {task_constraints}"}
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
    try:
        resp = json.loads(resp.output_text)
    except:
        resp = resp.output_text
    return resp


def is_json_string(x) -> bool:
    if not isinstance(x, str):
        return False
    try:
        json.loads(x)
        return True
    except (ValueError, TypeError):
        return False
    

async def regenerate_search_query(search_query, task_description, task_constraints, reason_list):
    system_prompt = """
        You are a search query refinement agent.

        Inputs:
        - Current search query
        - Task description
        - Task constraints
        - Failure reasons

        Goal:
        Refine the current search query to improve image retrieval quality.

        Key Principle:
        The search query must describe a GENERAL VISUAL CONCEPT,
        not a specific scene, event, action sequence, or narrative.

        Rules (MUST FOLLOW):
        - Output EXACTLY ONE search query
        - Use English only
        - The query must be SHORT (prefer 2–7 keywords)
        - Do NOT output explanations or extra text
    """
    user_input = [
        {"type": "input_text", "text": f"Current search query: {search_query}\nTask description: {task_description}\nTask constraints: {task_constraints}\nFailure reasons: {reason_list}"}
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

async def get_content_constraints_keyword(query, task_description, task_constraints):
    system_prompt = """
        You are a constraint synthesis expert for web image crawling.
        Given:
        1) Retrieval query (keyword)
        2) Task description
        3) Existing task constraints

        Goal:
        Synthesize CONTENT-LEVEL constraints implied by the query, but aligned with the task.
        These constraints should help filter false positives caused by query ambiguity.

        What "content-level constraints" mean here:
        - Constraints about what the image SHOULD depict (semantic content), and what it MUST NOT depict.
        - Prefer visually-checkable statements (things a judge model can verify from the image).
        - Focus on disambiguation: typical near-misses retrieved by the query that do not match the intended content.

        Output requirements (MUST follow):
        - Output MUST be a numbered LIST of 5 or fewer items.
        - Each item is ONE atomic constraint. No overlap/rephrasing across items.
        - Each item must be short (one sentence).
        - Include both positive and negative constraints when helpful.
        - Do NOT mention keywords, search engines, or crawling.
        - Do NOT output JSON. Do NOT output any extra text beyond the list.
    """
    user_input = [
        {"type": "input_text", "text": f"search query: {query}\nTask description: {task_description}\nTask constraints: {task_constraints}"}
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


async def _check_one(img_info, task_description, task_constraints, sem):
    async with sem:
        try:
            img_url = img_info["imageUrl"]
            check_result = await check_image(task_description, task_constraints, img_url)
            decision = (check_result.get("decision") or "").lower()
            reasons = check_result.get("reasons") or []
            ok = "accept" in decision
            return ok, reasons
        except Exception:
            return None, []
        

async def run_parallel(img_results_list, task_description, task_constraints,
                       max_concurrency=10):
    sem = asyncio.Semaphore(max_concurrency)

    tasks = [
        _check_one(img_info, task_description, task_constraints, sem)
        for img_info, _ in img_results_list
    ]

    results = await asyncio.gather(*tasks, return_exceptions=False)
    return results


async def get_correct_search_query(query, task_description, task_constraints):
    real_search_query = query
    for i in range(3):
        url = SERPER_URL
        img_results_list = []
        try:
            payload = json.dumps({
                "q": query,
                "tbs": "qdr:y",
                "page": 1
            })
            headers = {
                'X-API-KEY': SERPER_API_KEY,
                'Content-Type': 'application/json'
            }
            response = requests.request("POST", url, headers=headers, data=payload, timeout=10)
            response_json = json.loads(response.text)
            for img_info in response_json['images']:  
                img_results_list.append((img_info, query))
        except:
            return query
        
        if not img_results_list:
            print('google api has made some mistake.')
            return False
        
        img_results_list = img_results_list[:10]
        right_num = 0
        reason_list = []
        results = await run_parallel(img_results_list, task_description, task_constraints)

        for ok, reasons in results:
            if ok is None:
                continue
            if ok:
                right_num += 1
            else:
                reason_list.extend(reasons)
        
        total_num = len(results)
        if right_num > total_num * 0.6:
            return real_search_query
        
        else:
            real_search_query = await regenerate_search_query(query, task_description, task_constraints, reason_list)
    
    return real_search_query

async def crawler_planer(task_input: str):
    """
        Generate a one-time data collection and crawling plan based on a single task input. Please note that each call can only process one independent task description.

        Parameters
        task_input : str
        A complete input string representing one task, combining the following three components:
        1. Task description (the dataset to be built / task objective)
        2. Question type (e.g., multiple-choice, open-ended QA)
        3. One example QA pair (to help the model understand the expected format)
        4. Task constraints

        Returns
        str
        A model-generated data collection plan for this single task, strictly containing the following four sections:
        1. Information Need Analysis
        2. Crawling Strategy & Keyword Plan
        3. Data Schema (JSON)
        4. Final Data Collection Pipeline
    """

    system_prompt = """
        You are a Data Collection Planning Agent.

        Your objective is to design a complete data-acquisition plan based on the user-provided task description, covering only the stage up to data crawling (excluding data cleaning, annotation, training, or any later steps).

        IMPORTANT GLOBAL CONSTRAINTS (MUST FOLLOW):
        - This task does NOT define any data volume or dataset size targets.
        - When keyword-based crawling is used, the total number of keywords MUST be strictly less than 10.
        - Keyword planning should prioritize minimal yet sufficient semantic coverage, rather than maximal expansion.
        - ALL outputs MUST be written in English.
        - Do NOT output any non-English text (including Chinese).
        Input Format
        The user will provide three components:
            1. A task description
            2. The question type (e.g., multiple-choice, open-ended QA)
            3. One example QA pair

        Your Responsibilities

        1. Identify Which Information Must Be Crawled
        You must infer and explicitly list:
            (1) Required image / text / video content
            (2) Required auxiliary attributes, such as category labels or contextual descriptions

        2. Decide an Appropriate Crawling Strategy
        You must choose one strategy of the following:

        (1) Site-Specific Crawling
        This strategy is selected when the task explicitly mentions specific websites or data sources.
        You should provide:
            - The exact websites or data sources to be crawled
            - Whether these sites have clear hierarchical structures (e.g., category pages, index pages, or archive-style listings)

        (2) Latest-Image Crawling
        This strategy is selected when the task does not impose explicit domain or content constraints.
        You should provide:
            - What kinds of newly published images are considered relevant to the task
            - The basic visual or semantic criteria used to judge task relevance

        (3) Keyword-Based Crawling
        This strategy is selected when the task has clear domain or content requirements.
        You should provide:
            - A concise keyword set targeting the specified domain or content
            - Avoid keyword inflation or exhaustive enumeration
            - Ensure the total number of keywords is strictly less than 10
            - Explain whether category-based or sub-domain separation is necessary

        You must explicitly state:
            - Which strategy you choose
            - Why this strategy fits the task
            - The proposed keyword list and a brief coverage justification

        3. Define the Data Structure for Each Collected Item
        The Data Structure for each collected item is fixed and must not be modified:
            img_url
            img_path
            raw_content
            label_or_answer
            source_url

        4. Output the Final High-Level Data Collection Pipeline
        You must summarize and integrate all previous reasoning into a directly executable data-collection workflow (crawling stage only), including:

        (1) Information Need Analysis
            - What data must be crawled
            - Required fields
            - MUST-have vs OPTIONAL information

        (2) Target Sites & Retrieval Strategy
            - Site types
            - Whether APIs are available
            - Site traversal vs keyword-based retrieval

        (3) Crawling Steps
            - The step-by-step procedure from zero to obtaining raw data and structured JSON records

        (4) Keyword / Category Strategy
            - The final keyword list (if applicable)
            - Keyword grouping or category usage (if any)
            - Justification of why this minimal keyword set is sufficient

        Output Format
        Your final output must strictly follow the four mandatory sections:
        1. Information Need Analysis
        2. Crawling Strategy & Keyword Plan
        3. Data Schema (JSON)
        4. Final Data Collection Pipeline
        All four sections must be present and cannot be omitted.
        
        STRICT OUTPUT RULES:
        - Do NOT output <think>...</think> or any reasoning traces.
        - Do NOT output analysis or explanations outside the required sections.
        - Output MUST be valid JSON only.
        - Output format MUST be a single JSON object with EXACTLY the following four keys:
        "1. Information Need Analysis",
        "2. Crawling Strategy & Keyword Plan",
        "3. Data Schema (JSON)",
        "4. Final Data Collection Pipeline"
        

    """

    user_input = [
        {"type": "input_text", "text": task_input}
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


async def keywords_generator(ori_input: str):
    """
        A single-task keyword generation function that produces a structured set of keywords for web crawling or API-based data acquisition.

        Parameters
        ori_input : str
            A user-provided keyword-generation guideline, which may include:
            - Keyword types
            - Construction rules
            - Semantic constraints
            - Optional category definitions
            The agent must follow this guideline strictly and is not allowed to use any
            built-in or predefined keyword logic.This guideline does not allow "describing keywords in both Chinese and English"; the entire text must be in English.

        Returns
        str
            A model-generated result containing two required parts:
            1. **Structured keyword output**  
            - If no categories: a list of keywords  
                Example: `["keyword1", "keyword2", ...]`
            - If categories exist: a dictionary mapping categories to keyword lists  
                Example:
                ```
                {
                    "CategoryA": ["kw1", "kw2"],
                    "CategoryB": ["kw3", "kw4"]
                }
                ```

            2. **Natural-language explanation**  
            A short paragraph describing:
            - How N is distributed across categories (if any)
            - How many items each keyword is expected to collect

            The function returns only the final formatted output from the agent.  
            It does **not** execute crawling or verify keyword effectiveness.
    """

    system_prompt = """
        You are the keyword planning agent responsible for the data acquisition task.
        You will receive an initial crawling suggestion.
        Your goal is to generate a concise and effective set of keywords for web crawling and API-based image acquisition.
        Important Semantic Definitions (Must Be Followed):
        - The primary goal is to achieve semantic coverage and high recall.
        - The total number of generated keywords must be strictly less than 10.
        You must strictly adhere to the user-provided keyword generation guidelines and must not use any predefined or built-in keyword generation logic.
        Your reasoning and output must follow these rules:
        1. Read the user input, including:
        - Keyword generation guidelines (keyword types, structural rules, semantic constraints, etc.)
        2. Keyword design constraints (crucial):
        - Include every keyword that appears in the initial crawling suggestion (or replace it with a synonym).
        - Each keyword should be concise, specific, and easy to remember.
        - Avoid using overly long phrases, compound descriptions, or overly specific modifiers.
        - Prioritize common visual nouns and widely used search terms.
        - Avoid generating too many keywords; a small number of high-quality keywords is better.
        - The total number of keywords across all categories must be strictly less than 10 but more than 6.
        - Keywords must not be near-repetitive or simple definitions.
        3. Keyword Planning Logic:
        - Do not assume each keyword corresponds to a certain number of images.
        - Do not attempt to balance or distribute the data volume across keywords.
        - Generate only the minimum number of keywords required to cover the semantic space defined in the guidelines.
        4. Output Requirements:
        The output must include the following:
        Structured Keyword Output (
        Output a list:
        ["<keyword_1>", "<keyword_2>", "..."]
        - Do not include intermediate reasoning, analysis steps, or any additional explanations.
        - When considering the keywords in the original crawling suggestions, you can brainstorm up to three synonyms, and ultimately select up to ten, ensuring that at least one of all keywords (or their synonyms) that appeared in the original crawling suggestions is included.
    - Synonyms are not necessarily required; if the keywords are sufficiently comprehensive, synonyms are not needed.
    """
    user_input = [
        {"type": "input_text", "text": ori_input}
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


async def _one_query(query, task_description, ori_task_constraints, sem):
    async with sem:
        content_constraints = await get_content_constraints_keyword(
            query, task_description, ori_task_constraints
        )
        task_constraints = ori_task_constraints + f"\ncontent constraints: {content_constraints}"
        final_search_query = await get_correct_search_query(
            query, task_description, task_constraints
        )
        return final_search_query


async def _wrapped_one(key, idx, query, task_description, ori_task_constraints, sem):
    try:
        res = await _one_query(query, task_description, ori_task_constraints, sem)
        return key, idx, res, None
    except Exception as e:
        return key, idx, None, str(e)


async def build_tasks_final_parallel_with_tqdm(data, max_concurrency: int = 5):
    sem = asyncio.Semaphore(max_concurrency)

    items = list(data["raw_tool_output"]["finest_granularity_tasks"].items())

    tasks_final = {
        key: [None] * len(data["tasks"][key])
        for key, _ in items
    }

    all_tasks = []
    total = 0

    for key, value in items:
        keyword_list = data["tasks"][key]
        task_description = value["task_description"]
        ori_task_constraints = value["core_constraints"]

        for idx, q in enumerate(keyword_list):
            total += 1
            all_tasks.append(
                asyncio.create_task(
                    _wrapped_one(key, idx, q, task_description, ori_task_constraints, sem)
                )
            )

    with tqdm(total=total, desc="Generating search queries") as pbar:
        for fut in asyncio.as_completed(all_tasks):
            key, idx, res, err = await fut
            tasks_final[key][idx] = res
            pbar.update(1)

    tasks_final = {k: [x for x in v if x is not None] for k, v in tasks_final.items()}

    return tasks_final


async def get_img_path(index, img_url, save_img_dir):
    if 'jpg' in img_url:
        img_path = os.path.join(save_img_dir, f'{index}.jpg')
        return img_path
    elif 'jpeg' in img_url:
        img_path = os.path.join(save_img_dir, f'{index}.jpg')
        return img_path
    elif 'png' in img_url:
        img_path = os.path.join(save_img_dir, f'{index}.jpg')
        return img_path
    else:
        return False


def _pick_proxy(proxies: dict | None) -> str | None:
    if not proxies:
        return None
    if isinstance(proxies, dict):
        return proxies.get("http") or proxies.get("https")
    if isinstance(proxies, str):
        return proxies
    return None


async def fetch_images_for_queries_aiohttp(
    url: str,
    headers: dict,
    query_list: list[str],
    total_images_per_task: int = 400,
    per_page: int = 10,
    proxy: str | None = None,
    max_concurrency: int = 20,
    timeout_sec: int = 10,
    tqdm_desc: str = "Crawling pages",
):
    sem = asyncio.Semaphore(max_concurrency)
    timeout = aiohttp.ClientTimeout(total=timeout_sec)

    n_q = max(len(query_list), 1)
    target_per_query = math.ceil(total_images_per_task / n_q)
    pages_per_query = max(1, math.ceil(target_per_query / per_page))

    async def _one(session: aiohttp.ClientSession, query: str, page: int):
        payload = {"q": query, "tbs": "qdr:m", "page": page}
        async with sem:
            try:
                async with session.post(url, headers=headers, json=payload, proxy=proxy) as resp:
                    resp.raise_for_status()
                    data = await resp.json(content_type=None)
                    images = data.get("images", [])
                    return [(img_info, query) for img_info in images]
            except Exception as e:
                print(e)
                return []

    total_pages = len(query_list) * pages_per_query
    img_results_list: list[tuple[dict, str]] = []

    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            asyncio.create_task(_one(session, query, page))
            for query in query_list
            for page in range(1, pages_per_query + 1)
        ]

        with tqdm(total=total_pages, desc=f"{tqdm_desc} (q={n_q}, per_q={target_per_query}, pages_q={pages_per_query})") as pbar:
            for fut in asyncio.as_completed(tasks):
                img_results_list.extend(await fut)
                pbar.update(1)

    return img_results_list


async def download_one_image_aiohttp(
    session: aiohttp.ClientSession,
    img_url: str,
    img_path: str,
    proxy: str | None = None,
    chunk_size: int = 1 << 16,
):
    try:
        async with session.get(img_url, proxy=proxy) as resp:
            resp.raise_for_status()
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            with open(img_path, "wb") as f:
                async for chunk in resp.content.iter_chunked(chunk_size):
                    f.write(chunk)
        return True
    except Exception:
        return False
    

async def download_images_batch_aiohttp(
    img_results_list: list[tuple[dict, str]],
    img_save_dir: str,
    proxy: str | None,
    timeout_sec: int,
    download_concurrency: int,
    start_index: int,
    tqdm_desc: str = "Downloading images",
):
    sem = asyncio.Semaphore(download_concurrency)
    timeout = aiohttp.ClientTimeout(total=timeout_sec)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def _one(item: tuple[dict, str], idx: int):
            img_info, search_query = item
            try:
                img_url = img_info["imageUrl"]
                img_title = img_info.get("title", "")
                source_link = img_info.get("link", "")

                img_path = await get_img_path(idx, img_url, img_save_dir)
                if not img_path:
                    return None

                async with sem:
                    ok = await download_one_image_aiohttp(session, img_url, img_path, proxy=proxy)

                if not ok:
                    return None

                img_filename = os.path.basename(img_path)
                return img_filename, {
                    "img_url": img_url,
                    "img_path": img_path,
                    "raw_content": img_title,
                    "source_url": source_link,
                    "search_query": search_query,
                    "img_title": img_title,
                }

            except KeyboardInterrupt:
                raise
            except Exception:
                return None

        tasks = [
            asyncio.create_task(_one(item, start_index + i))
            for i, item in enumerate(img_results_list)
        ]

        results = []
        with tqdm(total=len(tasks), desc=tqdm_desc) as pbar:
            for fut in asyncio.as_completed(tasks):
                r = await fut
                if r is not None:
                    results.append(r)
                pbar.update(1)

        return results
    

async def process_one_task_aiohttp_best(
    task_name: str,
    query_list: list[str],
    url: str,
    x_api_key: str,
    proxies: dict | None,
    save_image_dir: str,
    save_json_dir: str,
    exist_data: dict,
    crawl_concurrency: int = 20,
    download_concurrency: int = 30,
    timeout_sec: int = 10,
    total_images_per_task: int = 100
):
    task_name_safe = "_".join(task_name.split())

    img_save_dir = os.path.join(save_image_dir, task_name_safe)
    os.makedirs(img_save_dir, exist_ok=True)
    json_save_dir = os.path.join(save_json_dir, task_name_safe)
    os.makedirs(json_save_dir, exist_ok=True)
    save_json_path = os.path.join(json_save_dir, "img_info.json")

    headers = {"X-API-KEY": x_api_key, "Content-Type": "application/json"}
    proxy = _pick_proxy(proxies)


    img_results_list = await fetch_images_for_queries_aiohttp(
        url=url,
        headers=headers,
        query_list=query_list,
        total_images_per_task=total_images_per_task,
        per_page=10,
        proxy=proxy,
        max_concurrency=crawl_concurrency,
        timeout_sec=timeout_sec,
        tqdm_desc=f"[{task_name_safe}] Crawling pages",
    )
    index0 = len([
        f for f in os.listdir(img_save_dir)
        if os.path.isfile(os.path.join(img_save_dir, f))
    ])

    results = await download_images_batch_aiohttp(
        img_results_list=img_results_list,
        img_save_dir=img_save_dir,
        proxy=proxy,
        timeout_sec=timeout_sec,
        download_concurrency=download_concurrency,
        start_index=index0,
        tqdm_desc=f"[{task_name_safe}] Downloading images",
    )

    for img_filename, cmp_info in results:
        exist_data[img_filename] = cmp_info

    with open(save_json_path, "w") as f:
        json.dump(exist_data, f, indent=4)

    return len(results), save_json_path


async def fetch_images_for_queries_aiohttp_debug(
    url: str,
    headers: dict,
    query_list: list[str],
    pages_per_query: int = 2,
    proxy: str | None = None,
    max_concurrency: int = 5,
    timeout_sec: int = 10,
    print_first_n_errors: int = 5,
):
    sem = asyncio.Semaphore(max_concurrency)
    timeout = aiohttp.ClientTimeout(total=timeout_sec)

    err_count = 0
    ok_count = 0
    img_results_list = []

    async def _one(session: aiohttp.ClientSession, query: str, page: int):
        nonlocal err_count, ok_count

        payload = {"q": query, "tbs": "qdr:y", "page": page}

        async with sem:
            try:
                async with session.post(url, headers=headers, json=payload, proxy=proxy) as resp:
                    status = resp.status
                    text = await resp.text()

                    if status != 200:
                        err_count += 1
                        if err_count <= print_first_n_errors:
                            print(f"[HTTP {status}] query={query!r} page={page} body_head={text[:200]!r}")
                        return []

                    try:
                        data = json.loads(text)
                    except Exception:
                        err_count += 1
                        if err_count <= print_first_n_errors:
                            print(f"[JSON parse failed] query={query!r} page={page} body_head={text[:200]!r}")
                        return []

                    images = data.get("images")
                    if not isinstance(images, list):
                        err_count += 1
                        if err_count <= print_first_n_errors:
                            print(f"[No 'images' list] keys={list(data.keys())[:20]} query={query!r} page={page}")
                        return []

                    ok_count += 1
                    return [(img_info, query) for img_info in images]

            except Exception as e:
                err_count += 1
                if err_count <= print_first_n_errors:
                    print(f"[Exception] query={query!r} page={page} err={repr(e)}")
                return []

    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            asyncio.create_task(_one(session, query, page))
            for query in query_list
            for page in range(1, pages_per_query + 1)
        ]

        for fut in asyncio.as_completed(tasks):
            img_results_list.extend(await fut)

    print(f"done: ok_pages={ok_count}, err_pages={err_count}, images={len(img_results_list)}")
    return img_results_list


async def main():
    with open(BENCHMARK_SUMMARY_PATH, 'r') as f:
        data = json.load(f)

    final_data = {
        "raw_tool_output": data,
        "tasks": {}
    }

    for task_name in data["finest_granularity_tasks"]:
        task_info = data["finest_granularity_tasks"][task_name]

        input_task = str({
            k: v for k, v in task_info.items()
            if k not in ["constraints", "data_construction_method", "data_sources"]
        })
        
        crawler_plan = await crawler_planer(input_task)
        keyword_list = await keywords_generator(crawler_plan)
        
        final_data['tasks'][task_name] = json.loads(keyword_list)

        constraints = task_info["constraints"]
        description = task_info["task_description"]
        core_cons = await compress_constraints(description, constraints)
        data["finest_granularity_tasks"][task_name]["core_constraints"] = core_cons

    task_keyword = await build_tasks_final_parallel_with_tqdm(final_data)
    final_data['tasks_final'] = task_keyword

    exist_data = {}
    for task_name, query_list in task_keyword.items():
        n, json_path = await process_one_task_aiohttp_best(
            task_name=task_name,
            query_list=query_list,
            url=SERPER_URL,
            x_api_key=SERPER_API_KEY,
            save_image_dir=IMAGE_SVAE_DIR,
            save_json_dir=IMAGE_INFO_DIR,
            exist_data=exist_data,
            crawl_concurrency=5,
            download_concurrency=20,
            timeout_sec=10,
            total_images_per_task=TOTAL_IMAGE_PER_TASK
        )
        print(f"{task_name}: downloaded {n} images -> {json_path}")

    with open(BENCHMARK_SUMMARY_PATH, 'w') as f:
        json.dump(final_data, f, indent=4)


if __name__ == "__main__":
    asyncio.run(main())

