import numpy as np
from ollama import embeddings

class Kb:
    def __init__(self, filepath):
        # 读取文件内容
        content = self.read_file(filepath)
        # print(content)
        # 读取拆分好的数组
        self.chunks = self.split_content(content)
        # print(chunks)
        # for chunk in chunks:
        #     print(chunk)
        #     print('=' * 10)
        # 转换成向量
        self.embeds = self.get_embeddings(self.chunks)

    # 读取文件
    def read_file(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return content

    # 拆分知识库
    @staticmethod
    def split_content(content):
        chunks = content.split('# ')
        # 过滤掉空块
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        return chunks

    # 字符串转向量(embeddings)
    def get_embedding(self, chunk):
        # milkey/m3e    0.642084887746903
        # bge-m3    0.6073383067378445
        # nomic-embed-text  完全找不到
        # res = embeddings(model='milkey/m3e', prompt=chunk)
        res  = res['embeddings'][0]
        # print(chunk)
        # print(res)
        # print(res['embedding'])
        return res['embedding']

    def get_embeddings(self, chunks):
        embeds = []
        for chunk in chunks:
            embed = self.get_embedding(chunk)
            embeds.append(embed)
        return np.array(embeds)

    # 查询相似性向量
    def search(self, text):
        print(text)
        max_similarity = 0
        max_similarity_index = 0
        ask_embed = self.get_embedding(text)
        for kb_embed_index, kb_embed in enumerate(self.embeds):
            similarity = self.similarity(kb_embed, ask_embed)
            # print(similarity)
            # print(self.chunks[kb_embed_index])
            if similarity > max_similarity:
                max_similarity = similarity
                max_similarity_index = kb_embed_index
        print(max_similarity)
        print(self.chunks[max_similarity_index])
        # print(self.embeds[max_similarity_index])
        # 返回查到的相关文本
        return self.chunks[max_similarity_index]

    # 相似度
    @staticmethod
    def similarity(A, B):
        # 计算点积
        dot_product = np.dot(A, B)
        # 计算范数
        norm_A = np.linalg.norm(A)
        norm_B = np.linalg.norm(B)
        # 计算余弦相似度
        cosine_sim = dot_product / (norm_A * norm_B)
        return cosine_sim
    
    