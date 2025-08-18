import os
import json
from PIL import Image
import numpy as np
from typing import List, Tuple, Dict


def call_image_search(image_url: str, cache_id: str):
    """
    基于cache数据的图像搜索工具。
    
    根据输入的id从fvqa_train_cache或fvqa_test_cache文件夹中读取相应的搜索结果数据，
    包括图片和标题信息。

    Args:
        image_url (str): 查询图像的URL或内部标识符（当前版本中未使用）
        cache_id (str): 查询ID，对应cache文件夹中的子文件夹名称

    Returns:
        tool_returned_str (str): 格式化的图像搜索结果字符串
        tool_returned_images (List[PIL.Image.Image]): 搜索结果图片列表
        tool_stat (dict): 工具执行状态和元数据
    """
    
    # 初始化返回值
    tool_returned_images = []
    tool_returned_str = ""
    tool_success = False
    
    try:
        # 从环境变量读取cache路径，如果没有设置则使用默认路径
        train_cache_base = os.getenv("FVQA_TRAIN_CACHE_PATH", "fvqa_train_cache")
        test_cache_base = os.getenv("FVQA_TEST_CACHE_PATH", "fvqa_test_cache")
        
        # 确定cache类型和路径
        cache_path = None
        if cache_id.startswith("fvqa_train_"):
            cache_path = os.path.join(train_cache_base, cache_id)
        elif cache_id.startswith("fvqa_test_"):
            cache_path = os.path.join(test_cache_base, cache_id)
        else:
            # 尝试两种cache类型
            train_path = os.path.join(train_cache_base, cache_id)
            test_path = os.path.join(test_cache_base, cache_id)
            
            if os.path.exists(train_path):
                cache_path = train_path
            elif os.path.exists(test_path):
                cache_path = test_path
            else:
                raise FileNotFoundError(f"找不到ID为 {cache_id} 的cache文件夹")
        
        # 检查cache文件夹是否存在
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Cache文件夹 {cache_path} 不存在")
        
        # 读取meta.json文件
        meta_file = os.path.join(cache_path, "meta.json")
        if not os.path.exists(meta_file):
            raise FileNotFoundError(f"Meta文件 {meta_file} 不存在")
        
        with open(meta_file, 'r', encoding='utf-8') as f:
            meta_data = json.load(f)
        
        # 获取标题列表和图片URL列表
        title_list = meta_data.get("title_list", [])
        image_urls = meta_data.get("image_urls", [])
        
        # 构建返回字符串
        tool_returned_str = "[Image Search Results] The result of the image search consists of web page information related to the image from the user's original question. Each result includes the main image from the web page and its title, ranked in descending order of search relevance, as demonstrated below:\n"
        
        # 读取图片文件并添加到返回列表
        for i, (title, img_url) in enumerate(zip(title_list, image_urls)):
            # 构建图片文件名
            img_filename = f"img_{i:03d}.jpg"
            img_path = os.path.join(cache_path, img_filename)
            
            # 如果jpg不存在，尝试png
            if not os.path.exists(img_path):
                img_filename = f"img_{i:03d}.png"
                img_path = os.path.join(cache_path, img_filename)
            
            # 读取图片
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path)
                    tool_returned_images.append(img)
                    
                    # 添加到返回字符串
                    tool_returned_str += f"{i+1}. image: <|vision_start|><|image_pad|><|vision_end|>\ntitle: {title}\n"
                except Exception as e:
                    print(f"读取图片 {img_path} 时出错: {e}")
                    # 如果图片读取失败，创建占位符图片
                    dummy_img = Image.fromarray(np.full((64, 64, 3), fill_value=100 + i * 30, dtype=np.uint8))
                    tool_returned_images.append(dummy_img)
                    tool_returned_str += f"{i+1}.title: {title}\n"
            else:
                # 如果图片文件不存在，创建占位符图片
                dummy_img = Image.fromarray(np.full((64, 64, 3), fill_value=100 + i * 30, dtype=np.uint8))
                tool_returned_images.append(dummy_img)
                tool_returned_str += f"{i+1}. image: <|vision_start|><|image_pad|><|vision_end|>\ntitle: {title}\n"
        
        tool_success = True
        
    except Exception as e:
        print(f"图像搜索工具执行出错: {e}")
        tool_returned_str = "[Image Search Results] There is an error encountered in performing search. Please reason with your own capaibilities."
        tool_returned_images = []
        tool_success = False
    
    # 构建工具状态信息
    tool_stat = {
        "success": tool_success,
        "num_images": len(tool_returned_images),
        "cache_id": cache_id,
        "cache_path": cache_path if 'cache_path' in locals() else None,
        "train_cache_base": train_cache_base if 'train_cache_base' in locals() else None,
        "test_cache_base": test_cache_base if 'test_cache_base' in locals() else None
    }
    
    return tool_returned_str, tool_returned_images, tool_stat