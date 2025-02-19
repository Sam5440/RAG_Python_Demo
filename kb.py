import numpy as np
import os
import json
from api import get_embedding

# 加载配置文件
with open('config.json', 'r') as f:
    config = json.load(f)

class Kb:
    def __init__(self, filepath):
        print("=== 初始化知识库 ===")
        print(f"正在读取文件: {filepath}")
        self.filepath = filepath
        
        # 检查是否需要重新向量化
        if self.need_recompute():
            print("知识库文件已更新，需要重新向量化")
            # 读取文件内容
            content = self.read_file(filepath)
            print(f"文件内容长度: {len(content)} 字符")
            
            # 读取拆分好的数组
            print("\n=== 开始拆分文档 ===")
            self.chunks = self.split_content(content)
            print(f"拆分完成，共得到 {len(self.chunks)} 个文本块")
            
            # 转换成向量并保存
            print("\n=== 开始向量化 ===")
            self.embeds = self.get_embeddings(self.chunks)
            print(f"向量化完成，向量维度: {self.embeds.shape}")
            
            # 保存向量到本地
            self.save_embeddings()
        else:
            print("知识库文件未更新，直接加载本地向量数据")
            self.load_embeddings()

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

    def get_embedding(self, chunk):
        """获取单个文本块的向量表示"""
        print(f"正在处理文本块: {chunk[:50]}...")
        embed = get_embedding(chunk)
        if embed is None:
            print("警告：该文本块向量化失败")
        return embed

    def get_embeddings(self, chunks):
        """批量获取文本块的向量表示"""
        embeds = []
        for i, chunk in enumerate(chunks, 1):
            print(f"\n处理第 {i}/{len(chunks)} 个文本块")
            embed = self.get_embedding(chunk)
            if embed is not None:
                embeds.append(embed)
        return np.array(embeds)

    def save_embeddings(self):
        """保存向量到本地文件"""
        print("\n=== 保存向量到本地 ===")
        save_dir = 'embeddings'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 保存向量数据
        np.save(f"{save_dir}/vectors.npy", self.embeds)
        print(f"向量数据已保存到: {save_dir}/vectors.npy")
        
        # 保存对应的文本块
        with open(f"{save_dir}/chunks.txt", 'w', encoding='utf-8') as f:
            for chunk in self.chunks:
                f.write(chunk + '\n===\n')
        print(f"文本块数据已保存到: {save_dir}/chunks.txt")
        
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
        print("\n=== 开始相似度搜索 ===")
        print(f"查询文本: {text}")
        
        # 获取查询文本的向量表示
        print("正在对查询文本进行向量化...")
        ask_embed = self.get_embedding(text)
        if ask_embed is None:
            print("错误：查询文本向量化失败")
            return []
            
        # 计算所有文本块的相似度
        print("计算相似度...")
        similarities = []
        for kb_embed_index, kb_embed in enumerate(self.embeds):
            similarity = self.similarity(kb_embed, ask_embed)
            similarities.append((self.chunks[kb_embed_index], similarity))
            print(f"与第 {kb_embed_index + 1} 个文本块的相似度: {similarity:.4f}")
        
        # 按相似度降序排序并获取前top_k个结果
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = config['retrieval']['top_k']
        top_results = similarities[:top_k]
        
        print(f"\n找到 {len(top_results)} 个最相似的文本块：")
        for i, (chunk, similarity) in enumerate(top_results, 1):
            print(f"\n第 {i} 相似的文本块 (相似度: {similarity:.4f})：")
            print(chunk)
        
        return top_results

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
    
    