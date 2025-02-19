import os
import json
import requests
import numpy as np

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
    print("\n=== 调用Embedding API ===")
    print(f"输入文本: {text[:100]}...")
    try:
        print("发送API请求...")
        response = requests.post(
            f"{API_BASE}/embeddings",
            headers=headers,
            json={
                "model": config['models']['embedding'],
                "input": text
            }
        )
        response.raise_for_status()
        embedding = response.json()["data"][0]["embedding"]
        print("成功获取嵌入向量")
        print(f"向量维度: {len(embedding)}")
        return embedding
    except Exception as e:
        print(f"获取嵌入向量时出错: {e}")
        return None

def chat_completion(messages):
    """获取聊天完成结果"""
    print("\n=== 调用Chat Completion API ===")
    print("系统消息:", messages[0]["content"])
    print("用户消息:", messages[1]["content"])
    try:
        print("发送API请求...")
        response = requests.post(
            f"{API_BASE}/chat/completions",
            headers=headers,
            json={
                "model": config['models']['chat'],
                "messages": messages
            }
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        print("\n获取到回复:")
        print(content)
        return content
    except Exception as e:
        print(f"聊天完成请求时出错: {e}")
        return None