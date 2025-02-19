from kb import Kb
from ollama import chat, Message

class Rag:
    def __init__(self, model, kb_filepath):
        self.kb_filepath = kb_filepath
        self.kb = Kb(kb_filepath)
        self.model = model
        self.prompt_template = """
        基于：%s
        回答：%s
        """

    def chat(self, message):
        # 用户消息检索相关上下文
        context = self.kb.search(message)
        # print(context)
        # prompt = self.prompt_template % (context, message)
        prompt = '请基于以下内容回答问题：\n' + context
        response = chat(self.model, [Message(role='system', content=prompt), Message(role='user', content=message)])
        return response['message']
