import os
import json
import requests
import numpy as np
from logger import logger

# 加载配置文件，获取API密钥和基础URL等配置信息
with open('config.json', 'r') as f:
    config = json.load(f)

# 从配置文件获取API配置
API_KEY = config['api']['key']  # API密钥
API_BASE = config['api']['base_url']  # API基础URL

# 设置请求头，包含认证信息和内容类型
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def get_embedding(text):
    """
    获取文本的嵌入向量
    
    原理：
    1. 调用Embedding API将文本转换为高维向量
    2. 向量维度通常是768维
    3. 向量可以用于计算文本间的语义相似度
    
    错误处理：
    1. 超时处理：设置30秒超时限制
    2. 网络异常：捕获请求相关异常
    3. 响应解析：处理JSON解析和数据提取异常
    
    Args:
        text: 需要向量化的文本
        
    Returns:
        list or None: 文本的嵌入向量，失败时返回None
    """
    logger.info(f"开始获取文本嵌入向量: {text[:50]}...")
    try:
        # 构建请求参数
        request_data = {
            "model": config['models']['embedding'],  # 使用配置文件指定的embedding模型
            "input": text
        }
        logger.debug(f"发送嵌入向量请求: {json.dumps(request_data, ensure_ascii=False)}")
        
        # 发送API请求
        response = requests.post(
            f"{API_BASE}/embeddings",
            headers=headers,
            json=request_data,
            timeout=30  # 设置30秒超时
        )
        response.raise_for_status()  # 检查响应状态
        
        # 提取嵌入向量
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
    """
    调用大模型API生成回答
    
    原理：
    1. 将用户问题和上下文组装成消息列表
    2. 调用Chat Completion API获取模型回答
    3. 从响应中提取生成的文本内容
    
    错误处理：
    1. 超时处理：设置60秒超时限制
    2. 网络异常：捕获请求相关异常
    3. 响应解析：处理JSON解析和数据提取异常
    
    Args:
        messages: 消息列表，包含对话历史
        
    Returns:
        str or None: 模型生成的回答，失败时返回None
    """
    logger.info("开始请求聊天完成")
    try:
        # 构建请求参数
        request_data = {
            "model": config['models']['chat'],  # 使用配置文件指定的chat模型
            "messages": messages
        }
        logger.debug(f"发送聊天完成请求: {json.dumps(request_data, ensure_ascii=False)}")
        
        # 发送API请求
        response = requests.post(
            f"{API_BASE}/chat/completions",
            headers=headers,
            json=request_data,
            timeout=60  # 设置60秒超时
        )
        response.raise_for_status()  # 检查响应状态
        
        # 提取生成的文本
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