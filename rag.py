from kb import Kb
from api import chat_completion
from logger import logger

class Rag:
    """
    RAG(Retrieval-Augmented Generation)系统的核心类
    
    主要功能：
    1. 知识检索：从知识库中检索与用户问题最相关的内容
    2. 上下文增强：将检索到的内容作为上下文提供给大模型
    3. 回答生成：调用大模型API生成最终的回答
    
    工作原理：
    1. 初始化时加载并向量化知识库
    2. 收到用户问题后，从知识库检索相关文本
    3. 将检索结果作为上下文，结合用户问题生成提示词
    4. 调用大模型API生成回答
    """
    def __init__(self, kb_filepath):
        """
        初始化RAG系统
        
        Args:
            kb_filepath: 知识库文件路径
            
        初始化流程：
        1. 记录知识库文件路径
        2. 初始化知识库对象(包括文本分块、向量化等)
        """
        logger.info(f"初始化RAG系统，知识库文件路径: {kb_filepath}")
        self.kb_filepath = kb_filepath
        self.kb = Kb(kb_filepath)  # 初始化知识库
        logger.info("RAG系统初始化完成")


    def chat(self, message):
        """
        处理用户消息并返回回答
        
        工作流程：
        1. 从知识库检索与问题相关的文本块
        2. 将检索到的文本块组合成上下文
        3. 构建提示信息(包含上下文和用户问题)
        4. 调用大模型API生成回答
        
        Args:
            message: 用户输入的消息
            
        Returns:
            str: 模型生成的回答
        """
        logger.info(f"收到用户消息: {message}")
        
        # 从知识库检索相关上下文
        logger.info("开始从知识库检索相关上下文")
        search_results = self.kb.search(message)
        logger.info(f"检索到 {len(search_results)} 条相关内容")
        
        # 提取相关文本块并组合成字符串
        context = "\n".join(chunk for chunk, _ in search_results)
        logger.debug(f"组合后的上下文内容: {context}")
        
        # 构建提示信息并调用模型生成回答
        logger.info("开始生成回答")
        prompt = '请基于以下内容回答问题：\n' + context
        response = chat_completion([{'role': 'system', 'content': prompt}, {'role': 'user', 'content': message}])
        
        if response is not None:
            logger.info("成功生成回答")
            return response
        else:
            logger.warning("生成回答失败")
            return "抱歉，我暂时无法回答这个问题。"
