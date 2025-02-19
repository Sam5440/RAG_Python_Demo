from logger import logger
import os
import json
import numpy as np
from api import get_embedding

# 加载配置文件
with open('config.json', 'r') as f:
    config = json.load(f)

class Kb:
    def __init__(self, filepath):
        logger.info(f"开始初始化知识库: {filepath}")
        self.filepath = filepath
        
        # 检查是否需要重新向量化
        if self.need_recompute():
            logger.info("需要重新计算向量")
            content = self.read_file(filepath)
            self.chunks = self.split_content(content)
            self.embeds = self.get_embeddings(self.chunks)
            self.save_embeddings()
        else:
            logger.info("加载已有向量数据")
            self.load_embeddings()
        logger.info("知识库初始化完成")

    def read_file(self, filepath):
        """读取文件内容"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return content

    @staticmethod
    def split_content(content):
        """拆分知识库内容为多个文本块"""
        chunks = content.split('# ')
        # 过滤掉空块
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        return chunks

    def get_embeddings(self, chunks):
        """批量获取文本块的向量表示"""
        embeds = []
        for chunk in chunks:
            embed = get_embedding(chunk)
            if embed is not None:
                embeds.append(embed)
        return np.array(embeds)

    def save_embeddings(self):
        """保存向量到本地文件"""
        save_dir = 'embeddings'
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存向量数据和文本块
        np.save(f"{save_dir}/vectors.npy", self.embeds)
        with open(f"{save_dir}/chunks.txt", 'w', encoding='utf-8') as f:
            f.write('\n===\n'.join(self.chunks))
        
        # 保存知识库文件的最后修改时间
        with open(f"{save_dir}/last_modified.txt", 'w') as f:
            f.write(str(os.path.getmtime(self.filepath)))

    def search(self, text):
        """搜索最相似的多个文本块
        Args:
            text: 查询文本
        Returns:
            list: 包含(文本块, 相似度)元组的列表，按相似度降序排列
        """
        logger.info(f"开始搜索相关内容，查询文本: {text}")
        ask_embed = get_embedding(text)
        if ask_embed is None:
            logger.error("获取查询文本的嵌入向量失败")
            return []
            
        logger.info("开始计算相似度并排序")
        # 计算所有文本块的相似度并排序
        similarities = []
        for chunk, kb_embed in zip(self.chunks, self.embeds):
            sim = self.similarity(kb_embed, ask_embed)
            similarities.append((chunk, sim))
            logger.debug(f"文本块相似度: {sim:.4f}\n文本块内容: {chunk[:100]}...")
            
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前top_k个结果
        top_k = config['retrieval']['top_k']
        results = similarities[:top_k]
        
        # 输出选中的结果详情
        for i, (chunk, sim) in enumerate(results):
            logger.info(f"Top {i+1} 匹配结果:\n相似度: {sim:.4f}\n文本块内容: {chunk[:200]}...")
            
        logger.info(f"搜索完成，找到 {len(results)} 个相关结果")
        return results

    @staticmethod
    def similarity(A, B):
        """计算两个向量的余弦相似度"""
        # 计算点积
        dot_product = np.dot(A, B)
        # 计算范数
        norm_A = np.linalg.norm(A)
        norm_B = np.linalg.norm(B)
        # 计算余弦相似度
        cosine_sim = dot_product / (norm_A * norm_B)
        return cosine_sim
    
    def need_recompute(self):
        """检查是否需要重新计算向量"""
        save_dir = 'embeddings'
        last_modified_file = f"{save_dir}/last_modified.txt"
        
        # 如果保存目录或最后修改时间文件不存在，需要重新计算
        if not os.path.exists(save_dir) or not os.path.exists(last_modified_file):
            return True
            
        # 读取保存的最后修改时间
        with open(last_modified_file, 'r') as f:
            saved_mtime = float(f.read().strip())
            
        # 获取当前文件的最后修改时间
        current_mtime = os.path.getmtime(self.filepath)
        
        # 如果当前修改时间晚于保存的修改时间，需要重新计算
        return current_mtime > saved_mtime

    def load_embeddings(self):
        """从本地加载向量数据"""
        save_dir = 'embeddings'
        
        # 加载向量数据
        self.embeds = np.load(f"{save_dir}/vectors.npy")
        print(f"已加载向量数据，向量维度: {self.embeds.shape}")
        
        # 加载文本块数据
        chunks = []
        current_chunk = []
        with open(f"{save_dir}/chunks.txt", 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() == '===':
                    if current_chunk:
                        chunks.append(''.join(current_chunk).strip())
                        current_chunk = []
                else:
                    current_chunk.append(line)
        if current_chunk:  # 处理最后一个块
            chunks.append(''.join(current_chunk).strip())
        self.chunks = chunks
        print(f"已加载文本块数据，共 {len(self.chunks)} 个文本块")
    
    