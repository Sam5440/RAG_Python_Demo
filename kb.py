from logger import logger
import os
import json
import numpy as np
from api import get_embedding

# 加载配置文件，用于获取检索时的top-k等参数
with open('config.json', 'r') as f:
    config = json.load(f)

class Kb:
    """
    知识库管理类，负责文本向量化和相似度检索
    
    主要功能：
    1. 文本分块：将长文本拆分成小段落
    2. 向量化：使用Embedding模型将文本转换为向量
    3. 向量存储：保存向量到本地文件，避免重复计算
    4. 相似度检索：计算查询文本与知识库的相似度，返回最相关的文本块
    """
    def __init__(self, filepath):
        """
        初始化知识库
        
        Args:
            filepath: 知识库文本文件路径
            
        工作流程：
        1. 检查是否需要重新计算向量(文件是否有更新)
        2. 如果需要重新计算：读取文件 -> 分块 -> 向量化 -> 保存
        3. 如果不需要重新计算：直接加载已有的向量数据
        """
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
        """读取文件内容
        
        Args:
            filepath: 文件路径
            
        Returns:
            str: 文件的全部内容
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return content

    @staticmethod
    def split_content(content):
        """拆分知识库内容为多个文本块
        
        原理：
        1. 使用'# '作为分隔符，将文本拆分成多个段落
        2. 过滤掉空白段落
        3. 每个段落作为一个独立的文本块用于后续向量化
        
        Args:
            content: 完整的文本内容
            
        Returns:
            list: 文本块列表
        """
        chunks = content.split('# ')
        # 过滤掉空块
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        return chunks

    def get_embeddings(self, chunks):
        """批量获取文本块的向量表示
        
        原理：
        1. 使用Embedding模型将文本转换为高维向量(通常是768维)
        2. 向量能够捕捉文本的语义信息
        3. 语义相近的文本，其向量距离也相近
        
        Args:
            chunks: 文本块列表
            
        Returns:
            np.array: 向量矩阵，每行是一个文本块的向量
        """
        embeds = []
        for chunk in chunks:
            embed = get_embedding(chunk)
            if embed is not None:
                embeds.append(embed)
        return np.array(embeds)

    def save_embeddings(self):
        """保存向量到本地文件
        
        保存内容：
        1. vectors.npy: 向量矩阵
        2. chunks.txt: 原始文本块
        3. last_modified.txt: 知识库文件的最后修改时间
        
        目的：
        1. 避免重复计算向量(很耗时)
        2. 便于快速加载已有向量
        """
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
        
        原理：
        1. 将查询文本转换为向量
        2. 计算与知识库中所有文本块的余弦相似度
        3. 按相似度降序排序
        4. 返回相似度最高的top-k个结果
        
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
        """计算两个向量的余弦相似度
        
        原理：
        1. 余弦相似度 = 向量点积 / (向量A的范数 * 向量B的范数)
        2. 结果范围[-1,1]，越接近1表示越相似
        3. 余弦相似度只关注向量的方向，不受向量长度影响
        
        Args:
            A: 向量A
            B: 向量B
            
        Returns:
            float: 余弦相似度
        """
        # 计算点积
        dot_product = np.dot(A, B)
        # 计算范数
        norm_A = np.linalg.norm(A)
        norm_B = np.linalg.norm(B)
        # 计算余弦相似度
        cosine_sim = dot_product / (norm_A * norm_B)
        return cosine_sim
    
    def need_recompute(self):
        """检查是否需要重新计算向量
        
        检查逻辑：
        1. 如果向量目录或最后修改时间文件不存在，需要重新计算
        2. 如果知识库文件的修改时间晚于记录的时间，需要重新计算
        
        Returns:
            bool: 是否需要重新计算向量
        """
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
        """从本地加载向量数据
        
        加载内容：
        1. vectors.npy: 向量矩阵
        2. chunks.txt: 原始文本块(使用===分隔)
        
        处理逻辑：
        1. 加载向量数据
        2. 读取chunks.txt，按===分隔符重建文本块列表
        """
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
    
    