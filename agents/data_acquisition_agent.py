import os
import json
from openai import AsyncOpenAI
import requests
from typing import Any, Dict, List, Optional
import base64
import asyncio
from tqdm import tqdm
import math
import aiohttp
import re, json
from urllib.parse import urljoin, urldefrag, urlparse
from project_config import IMAGE_SVAE_DIR, IMAGE_INFO_DIR, BENCHMARK_SUMMARY_PATH, OPENAI_API_KEY, SERPER_API_KEY, SERPER_URL, TOTAL_IMAGE_PER_TASK, FLICKR_API_KEY, FLICKR_REST_ENDPOINT, TAVILY_SEARCH_URL, DATA_ACQUISITION_AGENT_MCP_PORT
from mcp.server.fastmcp import FastMCP

try:
    from playwright.async_api import async_playwright
except Exception:
    async_playwright = None

mcp = FastMCP("paper_analyse_tools", port=DATA_ACQUISITION_AGENT_MCP_PORT)

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

DEFAULT_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


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


@mcp.tool()
async def crawler_planer(task_name: str, json_path: str):
    """
    Task Data Crawling Plan Generator

    Description:
    - Generate a data crawling plan for a specific task based on the summary JSON.
    - Reads the summary JSON from the specified path and extracts the task information for planning.

    Parameters:
    - task_name (str): The name of the task to generate a crawling plan for.
    - json_path (str): Path to the summary JSON file.

    Returns:
    - str: A data crawling plan corresponding to the specified task.
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
    with open(json_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON file: {e}")
    
    task_info = data["finest_granularity_tasks"][task_name]
    task_input = str({
        k: v for k, v in task_info.items()
        if k not in ["constraints", "data_construction_method", "data_sources", "core_constraints"]
    })
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


async def build_single_task_final_with_tqdm(
    task_name: str,
    keyword_list: list,
    task_description: str,
    core_constraints: dict,
    max_concurrency: int = 5
):
    """
    Generate results for a single task in parallel with progress bar.

    Parameters:
    - task_name (str): Name of the task.
    - keyword_list (list): List of keywords or sub-tasks to process.
    - task_description (str): Description of the task.
    - ori_task_constraints (dict): Original constraints for the task.
    - max_concurrency (int): Maximum number of concurrent tasks.

    Returns:
    - list: List of results corresponding to each keyword/sub-task.
    """
    import asyncio
    from tqdm.asyncio import tqdm

    sem = asyncio.Semaphore(max_concurrency)

    tasks_final = [None] * len(keyword_list)
    all_tasks = []

    total = len(keyword_list)

    for idx, q in enumerate(keyword_list):
        all_tasks.append(
            asyncio.create_task(
                _wrapped_one(task_name, idx, q, task_description, core_constraints, sem)
            )
        )

    with tqdm(total=total, desc=f"Processing task '{task_name}'") as pbar:
        for fut in asyncio.as_completed(all_tasks):
            key, idx, res, err = await fut
            tasks_final[idx] = res
            pbar.update(1)

    tasks_final = [x for x in tasks_final if x is not None]

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


@mcp.tool()
async def keywords_guided_crawling(crawler_plan: str, task_name: str, json_path: str):
    """
    Keywords-Guided Crawling for a Task

    Description:
    - Perform data crawling for a specific task based on a given crawling plan.
    - The summary JSON is read from `json_path` to access task-related information.
    - The crawling uses keywords as guidance according to the provided plan.

    Parameters:
    - crawler_plan (str): The plan defining how to guide the crawling process.
    - task_name (str): The name of the task to crawl data for.
    - json_path (str): Path to the summary JSON file.

    Returns:
    - Any: Result of the crawling operation for the task.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON file: {e}")
        
    task_info = data["finest_granularity_tasks"][task_name]
    keyword_list = await keywords_generator(crawler_plan)
    ori_keyword_list = json.loads(keyword_list)
    final_keyword_list = await build_single_task_final_with_tqdm(task_name, ori_keyword_list, task_info["task_description"], task_info["core_constraints"])

    exist_data = {}
    for task_name, query_list in final_keyword_list.items():
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
    
    data["finest_granularity_tasks"][task_name]["ori_keyword_list"] = ori_keyword_list
    data["finest_granularity_tasks"][task_name]["final_keyword_list"] = final_keyword_list

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

    return f"{task_name}: downloaded {n} images."


async def fetch_flickr_latest_images(
    api_key: str,
    per_page: int = 100,
    page: int = 1,
    extras: Optional[List[str]] = None,
    timeout: int = 15,
    proxies: Optional[dict] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch latest public photos from Flickr via flickr.photos.getRecent.

    Returns:
      A list of photo dicts (each includes fields returned by Flickr, plus any requested extras like url_l/url_c/url_z).
    """
    if extras is None:
        extras = ["url_l", "url_c", "url_z", "date_upload", "owner_name", "tags", "media"]

    params = {
        "method": "flickr.photos.getRecent",
        "api_key": api_key,
        "format": "json",
        "nojsoncallback": 1,
        "per_page": per_page,
        "page": page,
        "extras": ",".join(extras),
    }

    resp = requests.get(
        FLICKR_REST_ENDPOINT,
        params=params,
        timeout=timeout,
        proxies=proxies,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    resp.raise_for_status()
    data = resp.json()


    if data.get("stat") != "ok":
        raise RuntimeError(f"Flickr API error: {data}")

    photos = data.get("photos", {}).get("photo", [])
    return photos


async def pick_best_image_url(photo: Dict[str, Any]):
    """
    Pick the best available direct image URL from a photo record.
    Order: url_l > url_c > url_z > url_m.
    """
    for k in ("url_l", "url_c", "url_z", "url_m"):
        if photo.get(k):
            return photo[k]
    return None


async def img_download(url, save_path):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
        else:
            return False
    except Exception as e:
        print(f"发生错误: {e}")
        return False
    

async def get_label(task_description: str, ori_json: dict):
    system_prompt = """
        You are an answer detector.

        Your goal:
        Given the task description, retrieval keywords, and an image title,
        determine whether the retrieval keywords and title contain enough information to answer the task.

        If the information is enough:
            → Output ONLY the task answer (a short and direct answer).
        If the information is NOT enough:
            → Output ONLY: nono

        Do not explain. Do not output anything else.
    """
    label_response = await client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": f"task description: {task_description}.\n the title of image: {ori_json['img_title']}.\nSearch query: {ori_json['search_query']}"}
            ]}
        ]
    )
    result = label_response.choices[0].message.content
    ori_json['label'] = result
    return ori_json



@mcp.tool()
async def open_domain_crawling(task_name: str, json_path: str, criteria: str):
    """
        Perform open_domain_crawling (without image content constraints crawling), downloading, filtering, and JSON record generation
        for a **single data collection task**, using the Flickr "recent photos" stream as the source.

        Unlike keyword-based crawling, this function does not rely on query terms to retrieve images.
        Instead, it pulls newly uploaded public images and uses a task-specific judge to determine whether
        each candidate satisfies the task requirements.

        This function executes the complete crawling workflow (crawling stage only):
        - Fetching the latest public photos from the Flickr API (getRecent)
        - Selecting a usable direct image URL from multiple size options (url_l/url_c/url_z/url_m)
        - Filtering candidates by calling `check_image()` under task constraints + extra `criteria`
        - Downloading accepted images to disk
        - Building a per-image metadata record and inferring labels/answers via `get_label()`
        - Incrementally writing validated samples into a JSON file (`img_info.json`)

        This function is **single-task only**:
        - The `task_description` corresponds to exactly one task definition.
        - Every downloaded sample will be validated and labeled according to this single task.
        - If multiple tasks need to be crawled, the caller must invoke this function repeatedly.

        Parameters
        criteria : str
            Additional selection criteria used to constrain the judge, appended to `task_constraints`
            as "Criteria: {criteria}". This is typically used to enforce recency-task-specific

        img_save_dir : str
            Directory where downloaded image files will be stored.
            The function automatically reuses the current file count in this directory to index new samples.

        json_save_dir : str
            Directory where the JSON record file (`img_info.json`) will be saved.
            Each successfully validated image will be written into this JSON file incrementally.

        num_images : int
            Intended number of images to collect. The function attempts to accumulate accepted samples
            up to this target (subject to available acceptable candidates).

        task_description : str
            The task definition used to guide semantic labeling or QA generation.
            Passed into `check_image()` and `get_label()`.

        task_constraints : str
            The constraints for the task derived from paper data analysis. These constraints are used to
            judge candidate suitability. The function additionally appends `criteria` to this string
            for stricter filtering.

        Returns
        bool
            Returns True when the crawling and processing pipeline completes.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON file: {e}")
        
    task_info = data["finest_granularity_tasks"][task_name]
    task_description = task_info["task_description"]
    task_constraints = task_info["core_constraints"]
    photos = await fetch_flickr_latest_images(FLICKR_API_KEY, per_page=40, page=1)
    total_num = TOTAL_IMAGE_PER_TASK
    save_image_dir = os.path.join(IMAGE_SVAE_DIR, task_name)
    os.makedirs(save_image_dir, exist_ok=True)
    save_json_dir = os.path.join(IMAGE_INFO_DIR, task_name)
    os.makedirs(save_json_dir, exist_ok=True)
    save_json_path = os.path.join(save_json_dir, 'img_info.json')
    index = len([f for f in os.listdir(save_image_dir) if os.path.isfile(os.path.join(save_image_dir, f))])
    exist_data = {}
    for p in photos:
        try:
            img_url = await pick_best_image_url(p)
            if not img_url:
                continue
            
            task_constraints += f"\n Criteria: {criteria}"
            check_result = await check_image(task_description, task_constraints, img_url)
            accept_result = check_result['decision'].lower()
            if 'accept' in accept_result:
                img_path = await get_img_path(index, img_url, save_image_dir)
                if not img_path:
                    continue
                img_filename = os.path.basename(img_path)
                download_result = await img_download(img_url, img_path)
                if download_result:
                    cmp_info = {
                        'img_url': img_url,
                        'img_path': img_path,
                        'raw_content': 'none',
                        'source_url': f"https://www.flickr.com/photos/{p.get('owner')}/{p.get('id')}",
                        'search_query': 'none',
                        'img_title': p.get("title"),
                    }
                    final_json = await get_label(task_description, cmp_info)
                    if final_json:
                        exist_data[img_filename] = final_json
                        with open(save_json_path, 'w') as f:
                            json.dump(exist_data, f, indent=4)
                            index += 1
                        if len(exist_data) >= total_num:
                            return True
        except:
            continue
    
    return f"{task_name}: downloaded {len(exist_data)} images."


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_url(url: Any, base_url: str = "") -> str:
    url = normalize_text(url)
    if not url:
        return ""
    if base_url:
        url = urljoin(base_url, url)
    url, _ = urldefrag(url)
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return ""
    return url


def extract_json_object(text: str) -> dict:
    text = normalize_text(text)
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else {}
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {}
    try:
        result = json.loads(match.group(0))
        return result if isinstance(result, dict) else {}
    except Exception:
        return {}


async def source_specific_query_generator(
    crawler_plan: str,
    task_description: str,
    task_constraints: str,
) -> list[str]:
    system_prompt = """
        You are a source-specific web search query planner.

        Goal:
        Generate a small set of Tavily web search queries that can retrieve source webpages
        likely to contain task-relevant images and nearby explanatory text.

        Rules:
        - Output JSON only.
        - Output format: {"queries": ["query 1", "query 2"]}
        - Generate 1 to 5 queries.
        - Preserve explicitly named sources, websites, datasets, museums, archives, or organizations.
        - Keep each query concise and searchable.
        - English only.
    """
    user_input = [
        {
            "type": "input_text",
            "text": (
                f"crawler_plan: {crawler_plan}\n"
                f"task_description: {task_description}\n"
                f"task_constraints: {task_constraints}"
            ),
        }
    ]
    resp = await client.responses.create(
        model="gpt-5-mini",
        instructions=system_prompt,
        input=[{"role": "user", "content": user_input}],
    )

    parsed = extract_json_object(resp.output_text)
    queries = parsed.get("queries", [])
    if not isinstance(queries, list):
        queries = []

    queries = [normalize_text(q) for q in queries if normalize_text(q)]
    if queries:
        return queries[:5]

    fallback_query = normalize_text(task_description)[:180]
    return [fallback_query or normalize_text(crawler_plan)[:180]]


async def search_webpages_with_tavily(
    query: str,
    max_results: int = 8,
    timeout_sec: int = 20,
) -> list[dict]:
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set.")

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced",
        "max_results": max(1, min(max_results, 20)),
        "include_images": False,
        "include_image_descriptions": False,
        "include_raw_content": False,
    }

    timeout = aiohttp.ClientTimeout(total=timeout_sec)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(TAVILY_SEARCH_URL, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json(content_type=None)

    results = data.get("results", [])
    if not isinstance(results, list):
        return []

    deduped_results = []
    seen_urls = set()
    for item in results:
        if not isinstance(item, dict):
            continue

        source_url = normalize_url(item.get("url"))
        if not source_url or source_url in seen_urls:
            continue

        seen_urls.add(source_url)
        deduped_results.append(
            {
                "source_url": source_url,
                "page_title": normalize_text(item.get("title")),
                "page_snippet": normalize_text(item.get("content")),
                "search_query": query,
            }
        )

    return deduped_results


async def search_webpages_for_queries_with_tavily(
    query_list: list[str],
    max_results_per_query: int = 8,
    max_concurrency: int = 3,
) -> list[dict]:
    sem = asyncio.Semaphore(max(1, max_concurrency))

    async def _one(query: str):
        async with sem:
            try:
                return await search_webpages_with_tavily(
                    query=query,
                    max_results=max_results_per_query,
                    timeout_sec=20,
                )
            except Exception as e:
                print(e)
                return []

    tasks = [asyncio.create_task(_one(query)) for query in query_list]
    webpage_results = []
    with tqdm(total=len(tasks), desc="Searching Tavily webpages") as pbar:
        for fut in asyncio.as_completed(tasks):
            webpage_results.extend(await fut)
            pbar.update(1)

    deduped_results = []
    seen_urls = set()
    for item in webpage_results:
        source_url = item.get("source_url", "")
        if not source_url or source_url in seen_urls:
            continue
        seen_urls.add(source_url)
        deduped_results.append(item)

    return deduped_results


async def prepare_page_for_html_capture(page):
    try:
        await page.evaluate(
            """
            async () => {
              const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
              for (let i = 0; i < 6; i += 1) {
                window.scrollBy(0, Math.max(window.innerHeight, 800));
                await sleep(250);
              }
              window.scrollTo(0, 0);
              await sleep(200);
            }
            """
        )
    except Exception:
        return


async def fetch_webpages_with_playwright(
    webpage_results: list[dict],
    max_concurrency: int = 3,
    timeout_ms: int = 20000,
    tqdm_desc: str = "Fetching webpages",
) -> list[dict]:
    if async_playwright is None:
        raise RuntimeError(
            "playwright is not installed. Please install playwright and browser binaries before using this tool."
        )

    sem = asyncio.Semaphore(max(1, max_concurrency))
    page_results = []

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=DEFAULT_BROWSER_USER_AGENT,
            ignore_https_errors=True,
        )

        async def _one(webpage_result: dict):
            async with sem:
                page = await context.new_page()
                source_url = webpage_result.get("source_url", "")

                try:
                    await page.goto(source_url, wait_until="domcontentloaded", timeout=timeout_ms)
                    try:
                        await page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 5000))
                    except Exception:
                        pass

                    await prepare_page_for_html_capture(page)
                    candidates = await page.evaluate(
                        """
                        () => {
                          const clean = (value) => String(value || '').replace(/\\s+/g, ' ').trim();
                          const absolute = (value) => {
                            try {
                              if (!value) return '';
                              return new URL(value, document.baseURI).href;
                            } catch (_) {
                              return '';
                            }
                          };
                          const nearbyText = (img) => {
                            const parts = [];
                            const attrs = ['alt', 'title', 'aria-label'];
                            for (const attr of attrs) {
                              const value = clean(img.getAttribute(attr));
                              if (value) parts.push(value);
                            }
                            const figure = img.closest('figure');
                            if (figure) {
                              const caption = clean(figure.querySelector('figcaption')?.innerText);
                              if (caption) parts.push(caption);
                            }
                            const container = img.closest('article, section, li, div, p');
                            if (container) {
                              const text = clean(container.innerText);
                              if (text) parts.push(text.slice(0, 800));
                            }
                            return clean([...new Set(parts)].join(' ')).slice(0, 1000);
                          };

                          return Array.from(document.images || []).map((img) => {
                            const rawSrc =
                              img.currentSrc ||
                              img.getAttribute('src') ||
                              img.getAttribute('data-src') ||
                              img.getAttribute('data-original') ||
                              img.getAttribute('data-lazy-src') ||
                              '';
                            const src = absolute(rawSrc);
                            const width = img.naturalWidth || img.width || 0;
                            const height = img.naturalHeight || img.height || 0;
                            const text = nearbyText(img);
                            return {
                              img_url: src,
                              raw_content: text,
                              img_title: clean(img.getAttribute('title') || img.getAttribute('alt') || document.title),
                              width,
                              height
                            };
                          }).filter((item) => {
                            const url = item.img_url.toLowerCase();
                            if (!url || url.startsWith('data:') || url.startsWith('blob:')) return false;
                            if (url.includes('.svg') || url.includes('favicon') || url.includes('logo')) return false;
                            if (item.width && item.height && (item.width < 80 || item.height < 80)) return false;
                            return true;
                          });
                        }
                        """
                    )

                    page_title = normalize_text(await page.title()) or webpage_result.get("page_title", "")
                    final_url = normalize_url(page.url) or source_url

                    return {
                        **webpage_result,
                        "source_url": final_url,
                        "page_title": page_title,
                        "image_candidates": candidates if isinstance(candidates, list) else [],
                        "fetch_status": "success",
                    }
                except Exception as e:
                    return {
                        **webpage_result,
                        "image_candidates": [],
                        "fetch_status": "failed",
                        "fetch_error": str(e),
                    }
                finally:
                    await page.close()

        tasks = [asyncio.create_task(_one(webpage_result)) for webpage_result in webpage_results]

        with tqdm(total=len(tasks), desc=tqdm_desc) as pbar:
            for fut in asyncio.as_completed(tasks):
                page_results.append(await fut)
                pbar.update(1)

        await context.close()
        await browser.close()

    return page_results


async def extract_image_text_pairs_from_webpages(webpage_results: list[dict]) -> list[dict]:
    pair_results = []
    seen_urls = set()

    for webpage_result in webpage_results:
        if webpage_result.get("fetch_status") != "success":
            continue

        page_url = webpage_result.get("source_url", "")
        page_title = webpage_result.get("page_title", "")
        search_query = webpage_result.get("search_query", "")
        for candidate in webpage_result.get("image_candidates", []):
            if not isinstance(candidate, dict):
                continue

            img_url = normalize_url(candidate.get("img_url"), base_url=page_url)
            if not img_url or img_url in seen_urls:
                continue

            seen_urls.add(img_url)
            raw_content = normalize_text(candidate.get("raw_content")) or normalize_text(webpage_result.get("page_snippet"))
            img_title = normalize_text(candidate.get("img_title")) or page_title
            pair_results.append(
                {
                    "img_url": img_url,
                    "raw_content": raw_content,
                    "img_title": img_title,
                    "source_url": page_url,
                    "search_query": search_query,
                    "page_title": page_title,
                    "page_snippet": webpage_result.get("page_snippet", ""),
                }
            )

    return pair_results


async def get_source_specific_img_path(index: int, img_url: str, save_img_dir: str):
    parsed_path = urlparse(img_url).path.lower()
    ext = os.path.splitext(parsed_path)[1]
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        ext = ".jpg"
    if ext == ".jpeg":
        ext = ".jpg"
    return os.path.join(save_img_dir, f"{index}{ext}")


async def download_source_specific_image(
    session: aiohttp.ClientSession,
    img_url: str,
    img_path: str,
    proxy: str | None = None,
):
    try:
        headers = {"User-Agent": DEFAULT_BROWSER_USER_AGENT}
        async with session.get(img_url, proxy=proxy, headers=headers) as resp:
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "").lower()
            if "image" not in content_type:
                return False
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            with open(img_path, "wb") as f:
                async for chunk in resp.content.iter_chunked(1 << 16):
                    f.write(chunk)
        return True
    except Exception:
        return False


async def save_checked_source_specific_pairs(
    task_name: str,
    pair_results: list[dict],
    task_description: str,
    task_constraints: str,
    criteria: str = "",
    total_images_per_task: int = TOTAL_IMAGE_PER_TASK,
    timeout_sec: int = 15,
    download_concurrency: int = 8,
):
    task_name_safe = "_".join(task_name.split())
    save_image_dir = os.path.join(IMAGE_SVAE_DIR, task_name_safe)
    os.makedirs(save_image_dir, exist_ok=True)
    save_json_dir = os.path.join(IMAGE_INFO_DIR, task_name_safe)
    os.makedirs(save_json_dir, exist_ok=True)
    save_json_path = os.path.join(save_json_dir, "img_info.json")

    if os.path.exists(save_json_path):
        with open(save_json_path, "r", encoding="utf-8") as f:
            try:
                exist_data = json.load(f)
            except json.JSONDecodeError:
                exist_data = {}
    else:
        exist_data = {}

    existing_urls = {
        item.get("img_url")
        for item in exist_data.values()
        if isinstance(item, dict) and item.get("img_url")
    }
    index = len([f for f in os.listdir(save_image_dir) if os.path.isfile(os.path.join(save_image_dir, f))])
    proxy = _pick_proxy(None)
    timeout = aiohttp.ClientTimeout(total=timeout_sec)
    sem = asyncio.Semaphore(max(1, download_concurrency))
    full_constraints = f"{task_constraints}\nCriteria: {criteria}" if criteria else str(task_constraints)

    downloaded_count = 0

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for pair in tqdm(pair_results, desc=f"[{task_name_safe}] Checking and downloading images"):
            if downloaded_count >= total_images_per_task:
                break

            img_url = pair.get("img_url", "")
            if not img_url or img_url in existing_urls:
                continue

            try:
                check_result = await check_image(task_description, full_constraints, img_url)
                if not isinstance(check_result, dict):
                    continue
                accept_result = (check_result.get("decision") or "").lower()
                if "accept" not in accept_result:
                    continue

                img_path = await get_source_specific_img_path(index, img_url, save_image_dir)
                async with sem:
                    download_result = await download_source_specific_image(
                        session=session,
                        img_url=img_url,
                        img_path=img_path,
                        proxy=proxy,
                    )
                if not download_result:
                    continue

                img_filename = os.path.basename(img_path)
                cmp_info = {
                    "img_url": img_url,
                    "img_path": img_path,
                    "raw_content": pair.get("raw_content", ""),
                    "source_url": pair.get("source_url", ""),
                    "search_query": pair.get("search_query", ""),
                    "img_title": pair.get("img_title", ""),
                    "page_title": pair.get("page_title", ""),
                    "page_snippet": pair.get("page_snippet", ""),
                    "check_reasons": check_result.get("reasons", []),
                }

                try:
                    final_json = await get_label(task_description, cmp_info)
                except Exception:
                    final_json = cmp_info
                    final_json["label"] = "nono"

                final_json["label_or_answer"] = final_json.get("label", "nono")
                exist_data[img_filename] = final_json
                existing_urls.add(img_url)

                with open(save_json_path, "w", encoding="utf-8") as f:
                    json.dump(exist_data, f, indent=4)

                downloaded_count += 1
                index += 1
            except KeyboardInterrupt:
                raise
            except Exception:
                continue

    return downloaded_count, save_json_path


@mcp.tool()
async def source_specific_crawling(
    crawler_plan: str,
    task_name: str,
    json_path: str,
    criteria: str = "",
    max_webpage_num: int = 8,
):
    """
    Source-Specific Crawling for a Task

    Description:
    - Generate source-aware Tavily search queries from the crawling plan.
    - Retrieve the most relevant webpages with Tavily.
    - Open those webpages with Playwright and extract image candidates plus nearby text.
    - Use `check_image()` to judge whether each image satisfies the task requirements.
    - Download accepted images and save metadata to `{IMAGE_INFO_DIR}/{task_name}/img_info.json`.

    Parameters:
    - crawler_plan (str): Source-specific crawling plan or site/data-source description.
    - task_name (str): The name of the task to crawl data for.
    - json_path (str): Path to the summary JSON file.
    - criteria (str): Optional extra visual selection criteria appended to task constraints.
    - max_webpage_num (int): Maximum number of Tavily webpage results per query.

    Returns:
    - str: Summary of downloaded images and saved JSON path.
    """
    benchmark_json_path = json_path
    with open(benchmark_json_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON file: {e}")

    task_info = data["finest_granularity_tasks"][task_name]
    task_description = task_info["task_description"]
    task_constraints = task_info["core_constraints"]

    query_list = await source_specific_query_generator(
        crawler_plan=crawler_plan,
        task_description=task_description,
        task_constraints=task_constraints,
    )

    webpage_results = await search_webpages_for_queries_with_tavily(
        query_list=query_list,
        max_results_per_query=max(1, max_webpage_num),
        max_concurrency=3,
    )
    webpage_results = webpage_results[: max(1, max_webpage_num * max(len(query_list), 1))]

    webpage_results = await fetch_webpages_with_playwright(
        webpage_results=webpage_results,
        max_concurrency=3,
        timeout_ms=20000,
        tqdm_desc="Fetching source webpages",
    )

    pair_results = await extract_image_text_pairs_from_webpages(webpage_results)

    n, save_json_path = await save_checked_source_specific_pairs(
        task_name=task_name,
        pair_results=pair_results,
        task_description=task_description,
        task_constraints=task_constraints,
        criteria=criteria,
        total_images_per_task=TOTAL_IMAGE_PER_TASK,
        timeout_sec=15,
        download_concurrency=8,
    )

    data["finest_granularity_tasks"][task_name]["source_specific_query_list"] = query_list
    data["finest_granularity_tasks"][task_name]["source_specific_webpages"] = [
        {
            k: v
            for k, v in item.items()
            if k not in ["image_candidates"]
        }
        for item in webpage_results
    ]

    with open(benchmark_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    return f"{task_name}: downloaded {n} images. Metadata saved to {save_json_path}."


if __name__ == "__main__":
    mcp.run(transport="sse")