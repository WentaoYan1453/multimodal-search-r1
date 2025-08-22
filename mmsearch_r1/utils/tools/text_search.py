from typing import Optional, Tuple
import os
from verl.tools.utils.search_r1_like_utils import perform_single_search_batch
from typing import Tuple, Dict
from ddgs import DDGS
import time

def call_text_search(
    text_query: str,
    retrieval_service_url: Optional[str] = None,
    topk: Optional[int] = None,
    timeout: Optional[int] = None
) -> Tuple[str, dict]:
    """
    Performs a text-based search using the configured retrieval service and returns the
    raw JSON results along with accompanying metadata.

    Args:
        text_query (str): The input query string for the text search.
        retrieval_service_url (str, optional): URL of the retrieval service API. If not provided,
            fetched from the 'RETRIEVAL_SERVICE_URL' environment variable.
        topk (int, optional): Number of top results to return. If not provided, fetched from the
            'TOPK' environment variable (defaults to 3).
        timeout (int, optional): Request timeout in seconds. If not provided, fetched from the
            'TIMEOUT' environment variable (defaults to 30).

    Returns:
        result_text (str): JSON-encoded string containing the search results under the 'result' key.
        metadata (dict): Metadata dictionary including keys such as 'query_count', 'status', 
            'total_results', 'api_request_error', and 'formatted_result'.

    Raises:
        ValueError: If 'retrieval_service_url' is not provided and not set in the environment.
    """
    # Determine retrieval service URL
    if retrieval_service_url is None:
        retrieval_service_url = "http://0.0.0.0:8000/retrieve"
    if not retrieval_service_url:
        raise ValueError(
            "Retrieval service URL must be provided via argument or set in 'RETRIEVAL_SERVICE_URL' environment variable."
        )

    # Determine topk
    if topk is None:
        topk_env = os.getenv("TOPK")
        topk = int(topk_env) if topk_env is not None and topk_env.isdigit() else 3

    # Determine timeout
    if timeout is None:
        timeout_env = os.getenv("TIMEOUT")
        timeout = int(timeout_env) if timeout_env is not None and timeout_env.isdigit() else 120

    # Execute search batch for the single query
    result_text, metadata = perform_single_search_batch(
        retrieval_service_url=retrieval_service_url,
        query_list=[text_query],
        topk=topk,
        concurrent_semaphore=None,
        timeout=timeout,
        max_chars=1200
    )
    header = (
        "[Text Search Results]"
    )
    result_text = header + result_text

    return result_text, metadata



def call_web_text_search(text_query: str) -> Tuple[str, Dict]:
    """
    Perform a real text-based web search using DuckDuckGo (DDGS).
    Returns only title + snippet (no href) for easier LLM input.
    """

    max_results = 5
    region = "us-en"          # worldwide (global results)
    safesearch = "moderate"
    timelimit = 120
    backend = "auto"

    t0 = time.time()
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(
                text_query,
                max_results=max_results,
                region=region,
                safesearch=safesearch,
                timelimit=timelimit,
                backend=backend,
            ))
        latency_ms = int((time.time() - t0) * 1000)

        if results:
            lines = []
            lines.append("[Text Search Results] Below are the text summaries of the most relevant webpages related to your query, ranked in descending order of relevance:")
            for i, r in enumerate(results, start=1):
                title = r.get("title") or "No title"
                body = (r.get("body") or "").strip()
                if len(body) > 400:  # 避免太长
                    body = body[:400].rstrip() + "..."
                lines.append(f"{i}. {title}\n   {body}")
            tool_returned_str = "\n".join(lines)
        else:
            tool_returned_str = "[Text Search Results] No results were found for your query."

        tool_stat = {
            "success": True,
            "engine": "duckduckgo",
            "num_results": len(results),
            "latency_ms": latency_ms,
        }
        return tool_returned_str, tool_stat

    except Exception as e:
        latency_ms = int((time.time() - t0) * 1000)
        tool_returned_str = (
            "[Text Search Results] There was an error performing the search. "
            "Please reason with your own capabilities or try again later."
        )
        tool_stat = {
            "success": False,
            "engine": "duckduckgo",
            "error": str(e),
            "latency_ms": latency_ms,
        }
        return tool_returned_str, tool_stat