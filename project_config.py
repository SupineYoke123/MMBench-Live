import os
from pathlib import Path


def _to_posix(path: Path) -> str:
    return path.as_posix()


PROJECT_ROOT = Path(__file__).resolve().parent
_RUN_CACHE_DIR = PROJECT_ROOT / "cache"
RUN_CACHE_DIR = _to_posix(_RUN_CACHE_DIR)
BENCHMARK_SUMMARY_PATH = _to_posix(_RUN_CACHE_DIR / "benchmark_summary.json")
IMAGE_SVAE_DIR = _to_posix(_RUN_CACHE_DIR / "img_folder")
IMAGE_INFO_DIR = _to_posix(_RUN_CACHE_DIR / "info_folder")
QA_DIR = _to_posix(_RUN_CACHE_DIR / "qa_folder")
WRONG_QA_PATH = _to_posix(_RUN_CACHE_DIR / "wrong_qa_pairs.json")
TOTAL_IMAGE_PER_TASK = 20

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_GPT_GENERATION_MODEL = "gpt-5o-mini"

GEMINI_API_KEY = os.getenv("GOOGLE_AI_STUDIO_KEY", "")
DEFAULT_GEMINI_GENERATION_MODEL = "gemini-3-flash-preview"

FLICKR_API_KEY = os.getenv("FLICKR_API_KEY", "")
FLICKR_REST_ENDPOINT = "https://api.flickr.com/services/rest/"

SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
SERPER_URL = "https://google.serper.dev/images"

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_SEARCH_URL = "https://api.tavily.com/search"

SEGMENTATION_MCP = "http://127.0.0.1:8010/infer"
VISION_MCP = "http://127.0.0.1:8011"
DEPTH_MCP = "http://127.0.0.1:8012/infer"

BENCHMARK_SUMMARY_AGENT_MCP_PORT = 9050
DATA_ACQUISITION_AGENT_MCP_PORT = 9051
QA_GENERATION_AGENT_MCP_PORT = 9052
QA_VALIDATE_AGENT_MCP_PORT = 9053

PAPER_PATH = ''
DATASET_PATH = ''
