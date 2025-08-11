from typing import Optional, Tuple
import os
from verl.tools.utils.search_r1_like_utils import perform_single_search_batch


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
        retrieval_service_url = os.getenv("RETRIEVAL_SERVICE_URL")
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
        timeout = int(timeout_env) if timeout_env is not None and timeout_env.isdigit() else 30

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
