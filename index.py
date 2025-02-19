from rag import Rag

rag = Rag( '私人知识库.txt')
msg = rag.chat('请介绍下刘芳')
print(msg)