import os
import json
import requests
import numpy as np
from logger import logger

# 加载配置文件
with open('config.json', 'r') as f:
    config = json.load(f)

# 从配置文件获取API配置
API_KEY = config['api']['key']
API_BASE = config['api']['base_url']

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def get_embedding(text):
    """获取文本的嵌入向量"""
    logger.info(f"开始获取文本嵌入向量: {text[:50]}...")
    try:
        # 记录请求参数
        request_data = {
            "model": config['models']['embedding'],
            "input": text
        }
        logger.debug(f"发送嵌入向量请求: {json.dumps(request_data, ensure_ascii=False)}")
        
        response = requests.post(
            f"{API_BASE}/embeddings",
            headers=headers,
            json=request_data,
            timeout=30
        )
        response.raise_for_status()
        embedding = response.json()["data"][0]["embedding"]
        logger.info("成功获取嵌入向量")
        return embedding
    except requests.Timeout:
        print("获取嵌入向量超时")
        return None
    except requests.RequestException as e:
        print(f"获取嵌入向量请求失败: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"解析嵌入向量响应失败: {e}")
        return None
    except Exception as e:
        print(f"获取嵌入向量时发生未知错误: {e}")
        return None

def chat_completion(messages):
    """获取聊天完成结果"""
    logger.info("开始请求聊天完成")
    try:
        # 记录请求参数
        request_data = {
            "model": config['models']['chat'],
            "messages": messages
        }
        logger.debug(f"发送聊天完成请求: {json.dumps(request_data, ensure_ascii=False)}")
        
        response = requests.post(
            f"{API_BASE}/chat/completions",
            headers=headers,
            json=request_data,
            timeout=60
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        logger.info("成功获取聊天完成响应")
        return content
    except requests.Timeout:
        print("聊天完成请求超时")
        return None
    except requests.RequestException as e:
        print(f"聊天完成请求失败: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"解析聊天完成响应失败: {e}")
        return None
    except Exception as e:
        print(f"聊天完成请求时发生未知错误: {e}")
        return None