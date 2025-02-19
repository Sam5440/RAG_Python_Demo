from rag import Rag

rag = Rag( '私人知识库.txt')
msg = rag.chat('请介绍下刘芳')
msg = rag.chat('喜欢唱歌的有哪些')
msg = rag.chat('喜欢画画的有哪些')
msg = rag.chat('喜欢唱歌画画的有哪些')


print(msg)