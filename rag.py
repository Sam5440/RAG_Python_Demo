from kb import Kb
from api import chat_completion
from logger import logger

class Rag:
    def __init__(self, kb_filepath):
        """初始化RAG系统
        Args:
            kb_filepath: 知识库文件路径
        """
        logger.info(f"初始化RAG系统，知识库文件路径: {kb_filepath}")
        self.kb_filepath = kb_filepath
        self.kb = Kb(kb_filepath)  # 初始化知识库
        logger.info("RAG系统初始化完成")


    def chat(self, message):
        """处理用户消息并返回回答
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
