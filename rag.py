from kb import Kb
from api import chat_completion

class Rag:
    def __init__(self, kb_filepath):
        """初始化RAG系统
        Args:
            kb_filepath: 知识库文件路径
        """
        print("\n=== 初始化RAG系统 ===")
        # print(f"使用模型: {model}")
        print(f"知识库路径: {kb_filepath}")
        
        self.kb_filepath = kb_filepath
        self.kb = Kb(kb_filepath)  # 初始化知识库
        
        # 设置提示模板
        self.prompt_template = """
        基于：%s
        回答：%s
        """
        print("RAG系统初始化完成")

    def chat(self, message):
        """处理用户消息并返回回答
        Args:
            message: 用户输入的消息
        Returns:
            str: 模型生成的回答
        """
        print("\n=== 开始处理用户消息 ===")
        print(f"用户消息: {message}")
        
        # 从知识库检索相关上下文
        print("\n正在从知识库检索相关内容...")
        search_results = self.kb.search(message)
        
        # 提取相关文本块并组合成字符串
        context_texts = [chunk for chunk, similarity in search_results]
        context = "\n".join(context_texts)
        
        # 构建提示信息
        print("\n构建提示信息...")
        prompt = '请基于以下内容回答问题：\n' + context
        print(f"系统提示: {prompt}")
        
        # 调用模型生成回答
        print("\n正在生成回答...")
        response = chat_completion([{'role': 'system', 'content': prompt}, {'role': 'user', 'content': message}])
        
        if response is None:
            print("警告：模型未能生成有效回答")
            return "抱歉，我暂时无法回答这个问题。"
            
        print("\n=== 处理完成 ===")
        return response
