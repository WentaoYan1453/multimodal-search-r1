import os
import pytest

# 假设 text_search.py 在当前目录下
from text_search import call_text_search


os.environ["RETRIEVAL_SERVICE_URL"] = "http://0.0.0.0:8000/retrieve"

if __name__ == "__main__":
    # 假设你有一个 cache 文件夹 fvqa_train_cache/fvqa_train_0001
    test_id = "history of Schloss Uster and its commissioners ."
    image_url = "dummy_url"  # 当前版本中不会用到

    tool_returned_str, tool_stat = call_text_search(
                                text_query=test_id,
                            )
    print(tool_returned_str)