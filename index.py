from rag import Rag

rag = Rag('私人知识库.txt')

# 查询并打印刘芳的介绍
msg = rag.chat('请介绍下刘芳')
print('问题1：请介绍下刘芳')
print('回答：', msg)
print('\n')

# 查询并打印喜欢唱歌的人
msg = rag.chat('喜欢唱歌的有哪些')
print('问题2：喜欢唱歌的有哪些')
print('回答：', msg)
print('\n')

# 查询并打印喜欢画画的人
msg = rag.chat('喜欢画画的有哪些')
print('问题3：喜欢画画的有哪些')
print('回答：', msg)
print('\n')

# 查询并打印喜欢唱歌画画的人
msg = rag.chat('喜欢唱歌画画的有哪些')
print('问题4：喜欢唱歌画画的有哪些')
print('回答：', msg)


print(msg)