import os
import pytest

# 假设 text_search.py 在当前目录下
from text_search import call_text_search

def test_missing_url_raises():
    # 确保环境变量未设置
    os.environ.pop("RETRIEVAL_SERVICE_URL", None)
    with pytest.raises(ValueError):
        call_text_search("any query")

def test_call_text_search_success(monkeypatch):
    # 1. 设置环境变量
    os.environ["RETRIEVAL_SERVICE_URL"] = "http://0.0.0.0:8000/retrieve"

    # 2. mock perform_single_search_batch
    def fake_perform_single_search_batch(retrieval_service_url, query_list, topk, concurrent_semaphore, timeout):
        # 验证参数被正确传入
        assert retrieval_service_url == "http://0.0.0.0:8000/retrieve"
        assert query_list == ["hello world"]
        assert topk == 5
        assert timeout == 10
        return ("{\"result\": []}", {"status": "ok", "total_results": 0})

    # 注意这里要 mock 到 text_search 模块内部引用的位置
    monkeypatch.setattr("text_search.perform_single_search_batch", fake_perform_single_search_batch)

    # 3. 调用并断言返回值
    result_text, metadata = call_text_search("hello world", topk=5, timeout=10)
    assert result_text == "{\"result\": []}"
    assert metadata["status"] == "ok"
    assert metadata["total_results"] == 0
