# RAG系统运行流程详解（以"请介绍下刘芳"为例）

> 现在通过本链接点击注册[硅基流动](https://cloud.siliconflow.cn/i/g8snVG3G)即可获得 2000万 Tokens，折合人民币14元！
>
> 受邀好友作为新用户完成 SiliconCloud 账号注册，立刻获得 2000万 Tokens。
>
> 链接：[https://cloud.siliconflow.cn/i/g8snVG3G](https://cloud.siliconflow.cn/i/g8snVG3G)

> **更新于2025.2.19,所有内容基于 `Commit:完善注释`版本书写。**


**从零开始模拟 RAG 系统运行流程：以 **msg = rag.chat('请介绍下刘芳')** 为例**

**欢迎体验这个详细的教程！我们将以 **index.py** 中的代码 **msg = rag.chat('请介绍下刘芳')** 为例，模拟整个 RAG（Retrieval-Augmented Generation，检索增强生成）系统的运行流程。我们会从第一次向量化开始，一步步分析每个环节，包括代码、原理和日志输出，帮助你（即使是零基础的小白）理解整个过程。我们将参考日志文件 **logs/rag_20250219_215251.log**，并结合代码文件逐步讲解。**

---

**一、RAG 系统是什么？**

**在我们开始之前，先简单了解一下 RAG 系统。它是一个智能问答系统，能够根据你的问题从知识库中检索相关信息，然后生成自然语言回答。想象一下，你有一个私人档案（比如 **私人知识库.txt**），里面记录了很多人的信息，RAG 系统就像一个聪明的小助手，可以快速找到相关内容并回答你的问题。**

**在这个例子中：**

* **输入**：**请介绍下刘芳**
* **输出**：系统会返回刘芳的介绍，比如她的性别、爱好、电话和籍贯等。

**现在，让我们从头开始，模拟整个流程！**

---

**二、系统启动：初始化阶段**

**当你运行 **index.py** 时，程序首先会初始化整个 RAG 系统。我们从日志的第一行开始看起。**

**2.1 创建日志记录器（**logger.py**）**

**代码位置**：**logger.py**

**python**

```python
import os
import logging
from datetime import datetime

# 创建logs目录（如果不存在）
ifnot os.path.exists('logs'):
    os.makedirs('logs')

# 生成日志文件名（使用当前时间戳）
log_filename =f'logs/rag_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

# 配置日志记录器
logger = logging.getLogger('rag_system')
logger.setLevel(logging.INFO)

# 创建文件处理器
file_handler = logging.FileHandler(log_filename, encoding='utf-8')
file_handler.setLevel(logging.INFO)

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 设置日志格式，添加代码位置信息
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d:%(funcName)s] - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 添加处理器到日志记录器
logger.addHandler(file_handler)
logger.addHandler(console_handler)
```

**原理讲解**：

* **为什么要用日志？** 日志就像日记本，记录程序运行的每一步。如果出了问题，我们可以回看日记找到原因。
* **代码做什么？**
  * **os.makedirs('logs')**：如果 **logs** 文件夹不存在，就创建一个，用来存放日志文件。
  * **log_filename**：生成一个独特的日志文件名，比如 **rag_20250219_215251.log**，用当前时间（如 2025 年 2 月 19 日 21:52:51）命名。
  * **logging.getLogger('rag_system')**：创建一个名叫 **rag_system** 的日志记录器。
  * **setLevel(logging.INFO)**：设置日志级别为 **INFO**，只会记录重要信息（比调试信息更高级别）。
  * **FileHandler** 和 **StreamHandler**：一个把日志写到文件，一个显示在屏幕上。
  * **Formatter**：定义日志的格式，比如时间、文件名、行号和消息内容。

**日志输出**：这一步没有直接输出日志，但它为后续步骤准备好了记录工具。

---

**2.2 初始化 RAG 系统（**index.py** 和 **rag.py**）**

**代码位置**：**index.py**

**python**

```python
from rag import Rag

rag = Rag('私人知识库.txt')
```

**代码位置**：**rag.py**

**python**

```python
from kb import Kb
from api import chat_completion
from logger import logger

classRag:
def__init__(self, kb_filepath):
        logger.info(f"初始化RAG系统，知识库文件路径: {kb_filepath}")
        self.kb_filepath = kb_filepath
        self.kb = Kb(kb_filepath)# 初始化知识库
        logger.info("RAG系统初始化完成")
```

**日志输出**：

```text
2025-02-19 21:52:52,030 - rag_system - INFO - [rag.py:11:__init__] - 初始化RAG系统，知识库文件路径: 私人知识库.txt
```

**原理讲解**：

* **rag = Rag('私人知识库.txt')**：在 **index.py** 中，我们创建了一个 **Rag** 对象，传入知识库文件路径 **私人知识库.txt**。
* **__init__** 方法**：**rag.py** 中的 **Rag** 类初始化时，会记录一条日志，然后调用 **Kb** 类来加载知识库。**
* **日志的作用**：告诉你系统正在启动，并确认知识库文件是 **私人知识库.txt**。

---

**2.3 初始化知识库（**kb.py**）**

**代码位置**：**kb.py**

**python**

```python
classKb:
def__init__(self, filepath):
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
```

**日志输出**：

```text
2025-02-19 21:52:52,031 - rag_system - INFO - [kb.py:13:__init__] - 开始初始化知识库: 私人知识库.txt
2025-02-19 21:52:52,031 - rag_system - INFO - [kb.py:18:__init__] - 需要重新计算向量
```

**原理讲解**：

* **知识库是什么？** 知识库是一个文件（这里是 **私人知识库.txt**），里面存了很多人的信息，比如姓名、性别、爱好等。
* **self.filepath = filepath**：保存文件路径。
* **need_recompute()**：检查是否需要重新处理知识库（稍后解释）。
* **日志输出**：
  * **第一条告诉你知识库开始初始化。**
  * **第二条说明需要重新计算向量（可能是因为这是第一次运行，或者文件被修改过）。**

---

**2.4 检查是否需要重新向量化（**kb.py**）**

**代码位置**：**kb.py**

**python**

```python
defneed_recompute(self):
    save_dir ='embeddings'
    last_modified_file =f"{save_dir}/last_modified.txt"
  
ifnot os.path.exists(save_dir)ornot os.path.exists(last_modified_file):
returnTrue
    
withopen(last_modified_file,'r')as f:
        saved_mtime =float(f.read().strip())
    
    current_mtime = os.path.getmtime(self.filepath)
  
return current_mtime > saved_mtime
```

**原理讲解**：

* **为什么要检查？** 知识库的内容会变成“向量”（一种数学表示，后面会讲），但计算向量很费时间。如果文件没变，我们可以直接用之前的向量。
* **代码做什么？**
  * **save_dir = 'embeddings'**：向量会保存在 **embeddings** 文件夹。
  * **last_modified_file**：一个记录文件上次修改时间的小文件。
  * **如果 **embeddings** 文件夹或 **last_modified.txt** 不存在（比如第一次运行），返回 **True**，表示需要重新计算。**
  * **否则，比较文件的当前修改时间（**current_mtime**）和保存的时间（**saved_mtime**），如果文件更新了，就需要重新计算。**
* **结果**：日志显示 **需要重新计算向量**，说明这是第一次运行，或者文件被修改过。

---

**2.5 读取知识库文件（**kb.py**）**

**代码位置**：**kb.py**

**python**

```python
defread_file(self, filepath):
withopen(filepath,'r', encoding='utf-8')as f:
        content = f.read()
return content
```

**原理讲解**：

* **做什么？** 把 **私人知识库.txt** 的全部内容读出来，存到变量 **content** 中。
* **with open**：打开文件，**encoding='utf-8'** 确保能读中文，**f.read()** 把文件内容读成一个大字符串。
* **结果**：**content** 包含了整个文件，比如：

  ```text
  MIS部门人员名单
  # 1. 张小刚
  姓名：张小刚
  性别：男
  爱好：打篮球、踢足球
  ...
  # 5. 刘芳
  姓名：刘芳
  性别：女
  爱好：瑜伽、绘画
  ...
  ```

**日志输出**：这一步没有直接日志，但它是后续步骤的基础。

---

**2.6 拆分内容为文本块（**kb.py**）**

**代码位置**：**kb.py**

**python**

```python
@staticmethod
defsplit_content(content):
    chunks = content.split('# ')
    chunks =[chunk.strip()for chunk in chunks if chunk.strip()]
return chunks
```

**原理讲解**：

* **为什么要拆分？** 文件太大，不能直接处理。我们按 **# ** 分成小块，每块是一个人的信息。
* **代码做什么？**
  * **content.split('# ')**：用 **# ** 作为分隔符，把内容分成列表。
  * **[chunk.strip() for chunk in chunks if chunk.strip()]**：去掉每块前后的空格，过滤掉空块。
* **结果**：**self.chunks** 是一个列表，比如：
  * **chunks[0] = "MIS部门人员名单"**
  * **chunks[1] = "1. 张小刚\n姓名：张小刚\n性别：男\n爱好：打篮球、踢足球\n电话：13223344422\n籍贯：山东菏泽"**
  * **chunks[5] = "5. 刘芳\n姓名：刘芳\n性别：女\n爱好：瑜伽、绘画\n电话：13711223344\n籍贯：广东深圳"**

**日志输出**：这一步也没有直接日志。

---

**2.7 获取文本块的向量表示（**kb.py** 和 **api.py**）**

**代码位置**：**kb.py**

**python**

```python
defget_embeddings(self, chunks):
    embeds =[]
for chunk in chunks:
        embed = get_embedding(chunk)
if embed isnotNone:
            embeds.append(embed)
return np.array(embeds)
```

**代码位置**：**api.py**

**python**

```python
defget_embedding(text):
    logger.info(f"开始获取文本嵌入向量: {text[:50]}...")
try:
        request_data ={
"model": config['models']['embedding'],
"input": text
}
        logger.debug(f"发送嵌入向量请求: {json.dumps(request_data, ensure_ascii=False)}")
  
        response = requests.post(
f"{API_BASE}/embeddings",
            headers=headers,
            json=request_data,
            timeout=30
)
        response.raise_for_status()
        embedding = response.json()["data"][0]["embedding"]
        logger.info("成功获取嵌入向量")
return embedding
except:
# 异常处理代码略（超时、请求失败等）
returnNone
```

**日志输出**（部分示例）：

```text
2025-02-19 21:52:52,032 - rag_system - INFO - [api.py:22:get_embedding] - 开始获取文本嵌入向量: MIS部门人员名单...
2025-02-19 21:52:52,238 - rag_system - INFO - [api.py:39:get_embedding] - 成功获取嵌入向量
...
2025-02-19 21:52:53,066 - rag_system - INFO - [api.py:22:get_embedding] - 开始获取文本嵌入向量: 5. 刘芳
姓名：刘芳
性别：女
爱好：瑜伽、绘画
电话：13711223344
籍贯：广东深圳...
2025-02-19 21:52:53,236 - rag_system - INFO - [api.py:39:get_embedding] - 成功获取嵌入向量
```

**原理讲解**：

* **向量是什么？** 向量是一串数字，表示文本的意思。计算机不能直接理解文字，但能处理数字，向量就像文字的“数字版”。
* **get_embeddings**：循环处理每个文本块，调用 **get_embedding** 获取向量。
* **get_embedding**：
  * **发送请求给 API（**https://api.siliconflow.cn/v1/embeddings**），用 **BAAI/bge-m3** 模型把文本变成向量。**
  * **requests.post**：发送网络请求，带着文本内容。
  * **返回值是一个向量，比如 **[0.1, -0.2, 0.5, ...]**（通常几百个数）。**
* **结果**：**self.embeds** 是一个数组，每个元素是一个文本块的向量。比如：
  * **embeds[0]** 是 “MIS部门人员名单” 的向量。
  * **embeds[5]** 是刘芳信息的向量。

---

**2.8 保存向量到本地（**kb.py**）**

**代码位置**：**kb.py**

**python**

```python
defsave_embeddings(self):
    save_dir ='embeddings'
    os.makedirs(save_dir, exist_ok=True)
  
    np.save(f"{save_dir}/vectors.npy", self.embeds)
withopen(f"{save_dir}/chunks.txt",'w', encoding='utf-8')as f:
        f.write('\n===\n'.join(self.chunks))
  
withopen(f"{save_dir}/last_modified.txt",'w')as f:
        f.write(str(os.path.getmtime(self.filepath)))
```

**日志输出**：

```text
2025-02-19 21:53:00,110 - rag_system - INFO - [kb.py:26:__init__] - 知识库初始化完成
2025-02-19 21:53:00,110 - rag_system - INFO - [rag.py:14:__init__] - RAG系统初始化完成
```

**原理讲解**：

* **为什么要保存？** 计算向量很慢，下次启动时直接加载更快。
* **代码做什么？**
  * **os.makedirs**：创建 **embeddings** 文件夹。
  * **np.save**：把向量数组保存为 **vectors.npy** 文件。
  * **chunks.txt**：把文本块保存下来，用 **===** 分隔。
  * **last_modified.txt**：记录知识库文件的修改时间。
* **结果**：
  * **vectors.npy**：所有文本块的向量。
  * **chunks.txt**：所有文本块的原文。
  * **last_modified.txt**：一个时间戳，比如 **1708350772.0**。

---

**三、处理用户查询：**msg = rag.chat('请介绍下刘芳')

**初始化完成后，系统开始处理你的问题。**

**3.1 接收用户消息（**rag.py**）**

**代码位置**：**index.py**

**python**

```python
msg = rag.chat('请介绍下刘芳')
```

**代码位置**：**rag.py**

**python**

```python
defchat(self, message):
    logger.info(f"收到用户消息: {message}")
```

**日志输出**：

```text
2025-02-19 21:53:00,110 - rag_system - INFO - [rag.py:24:chat] - 收到用户消息: 请介绍下刘芳
```

**原理讲解**：

* **做什么？** 用户输入的问题 **请介绍下刘芳** 被传给 **chat** 方法，系统记录下来。
* **日志**：确认收到消息，方便调试。

---

**3.2 从知识库检索相关上下文（**rag.py** 和 **kb.py**）**

**代码位置**：**rag.py**

**python**

```python
logger.info("开始从知识库检索相关上下文")
search_results = self.kb.search(message)
logger.info(f"检索到 {len(search_results)} 条相关内容")
```

**日志输出**：

```text
2025-02-19 21:53:00,110 - rag_system - INFO - [rag.py:27:chat] - 开始从知识库检索相关上下文
```

**原理讲解**：

* **做什么？** 系统要从知识库中找到和 **请介绍下刘芳** 最相关的信息。
* **下一步**：调用 **kb.py** 的 **search** 方法。

**3.2.1 搜索相关内容（**kb.py**）**

**代码位置**：**kb.py**

**python**

```python
defsearch(self, text):
    logger.info(f"开始搜索相关内容，查询文本: {text}")
    ask_embed = get_embedding(text)
if ask_embed isNone:
        logger.error("获取查询文本的嵌入向量失败")
return[]
    
    logger.info("开始计算相似度并排序")
    similarities =[]
for chunk, kb_embed inzip(self.chunks, self.embeds):
        sim = self.similarity(kb_embed, ask_embed)
        similarities.append((chunk, sim))
        logger.debug(f"文本块相似度: {sim:.4f}\n文本块内容: {chunk[:100]}...")
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_k = config['retrieval']['top_k']
    results = similarities[:top_k]
  
for i,(chunk, sim)inenumerate(results):
        logger.info(f"Top {i+1} 匹配结果:\n相似度: {sim:.4f}\n文本块内容: {chunk[:200]}...")
    
    logger.info(f"搜索完成，找到 {len(results)} 个相关结果")
return results
```

**日志输出**：

```text
2025-02-19 21:53:00,110 - rag_system - INFO - [kb.py:72:search] - 开始搜索相关内容，查询文本: 请介绍下刘芳
2025-02-19 21:53:00,110 - rag_system - INFO - [api.py:22:get_embedding] - 开始获取文本嵌入向量: 请介绍下刘芳...
2025-02-19 21:53:00,276 - rag_system - INFO - [api.py:39:get_embedding] - 成功获取嵌入向量
2025-02-19 21:53:00,277 - rag_system - INFO - [kb.py:78:search] - 开始计算相似度并排序
2025-02-19 21:53:00,282 - rag_system - INFO - [kb.py:94:search] - Top 1 匹配结果:
相似度: 0.6185
文本块内容: 5. 刘芳
姓名：刘芳
性别：女
爱好：瑜伽、绘画
电话：13711223344
籍贯：广东深圳...
2025-02-19 21:53:00,282 - rag_system - INFO - [kb.py:94:search] - Top 2 匹配结果:
相似度: 0.4328
文本块内容: 21. 韩雪
姓名：韩雪
性别：女
爱好：跳舞、摄影
电话：13122334455
籍贯：甘肃兰州...
2025-02-19 21:53:00,282 - rag_system - INFO - [kb.py:94:search] - Top 3 匹配结果:
相似度: 0.4305
文本块内容: 7. 孙婷婷
姓名：孙婷婷
性别：女
爱好：跳舞、唱歌
电话：13677889900
籍贯：福建厦门...
2025-02-19 21:53:00,282 - rag_system - INFO - [kb.py:96:search] - 搜索完成，找到 3 个相关结果
2025-02-19 21:53:00,282 - rag_system - INFO - [rag.py:29:chat] - 检索到 3 条相关内容
```

**原理讲解**：

* **获取查询向量**：
  * **ask_embed = get_embedding(text)**：把 **请介绍下刘芳** 变成向量。
  * **日志显示从 21:53:00,110 到 21:53:00,276，花了约 0.166 秒完成。**
* **计算相似度**：
  * **self.similarity(kb_embed, ask_embed)**：用余弦相似度（后面解释）比较查询向量和每个文本块向量。
  * **余弦相似度：衡量两个向量的“方向”是否接近，值从 -1 到 1，越大越相似。**
* **排序和选择**：
  * **similarities.sort**：按相似度从高到低排序。
  * **top_k = config['retrieval']['top_k']**：从 **config.json** 取 **top_k=3**，选前 3 个结果。
* **结果**：
  * **Top 1：刘芳（相似度 0.6185），最相关。**
  * **Top 2：韩雪（0.4328）。**
  * **Top 3：孙婷婷（0.4305）。**
* **返回值**：**search_results** 是一个列表，包含 3 个 (文本块, 相似度) 对。

**3.2.2 计算余弦相似度（**kb.py**）**

**代码位置**：**kb.py**

**python**

```python
@staticmethod
defsimilarity(A, B):
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    cosine_sim = dot_product /(norm_A * norm_B)
return cosine_sim
```

**原理讲解**：

* **余弦相似度公式**：**cos(θ) = A·B / (|A| * |B|)**。
  * **np.dot(A, B)**：向量点积。
  * **np.linalg.norm**：计算向量长度。
  * **结果是 0 到 1 的值（这里都是正数）。**
* **例子**：刘芳的向量和查询向量的相似度是 0.6185，说明它们很接近。

---

**3.3 组合上下文（**rag.py**）**

**代码位置**：**rag.py**

**python**

```python
context ="\n".join(chunk for chunk, _ in search_results)
logger.debug(f"组合后的上下文内容: {context}")
```

**原理讲解**：

* **做什么？** 把检索到的 3 个文本块拼接成一个大字符串，用换行符 **\n** 分隔。
* **结果**：**context** 可能是：

  ```text
  5. 刘芳
  姓名：刘芳
  性别：女
  爱好：瑜伽、绘画
  电话：13711223344
  籍贯：广东深圳
  21. 韩雪
  姓名：韩雪
  性别：女
  爱好：跳舞、摄影
  电话：13122334455
  籍贯：甘肃兰州
  7. 孙婷婷
  姓名：孙婷婷
  性别：女
  爱好：跳舞、唱歌
  电话：13677889900
  籍贯：福建厦门
  ```
* **日志**：用 **debug** 级别记录，但日志文件中没显示（可能级别不够）。

---

**3.4 生成回答（**rag.py** 和 **api.py**）**

**代码位置**：**rag.py**

**python**

```python
logger.info("开始生成回答")
prompt ='请基于以下内容回答问题：\n'+ context
response = chat_completion([{'role':'system','content': prompt},{'role':'user','content': message}])
```

**代码位置**：**api.py**

**python**

```python
defchat_completion(messages):
    logger.info("开始请求聊天完成")
try:
        request_data ={
"model": config['models']['chat'],
"messages": messages
}
        response = requests.post(
f"{API_BASE}/chat/completions",
            headers=headers,
            json=request_data,
            timeout=60
)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        logger.info("成功获取聊天完成响应")
return content
except:
# 异常处理略
returnNone
```

**日志输出**：

```text
2025-02-19 21:53:00,282 - rag_system - INFO - [rag.py:36:chat] - 开始生成回答
2025-02-19 21:53:00,282 - rag_system - INFO - [api.py:56:chat_completion] - 开始请求聊天完成
2025-02-19 21:53:02,150 - rag_system - INFO - [api.py:73:chat_completion] - 成功获取聊天完成响应
2025-02-19 21:53:02,151 - rag_system - INFO - [rag.py:41:chat] - 成功生成回答
```

**原理讲解**：

* **构建提示**：

  * **prompt** 是系统指令，告诉模型：“用这些信息回答问题”。
  * **示例：**

    ```text
    请基于以下内容回答问题：
    5. 刘芳
    姓名：刘芳
    性别：女
    爱好：瑜伽、绘画
    电话：13711223344
    籍贯：广东深圳
    21. 韩雪
    ...
    ```
* **消息格式**：

  * **messages** 是一个列表：
    * **{'role': 'system', 'content': prompt}**：系统指令。
    * **{'role': 'user', 'content': '请介绍下刘芳'}**：用户问题。
* **调用 API**：

  * **用 **Qwen/Qwen2.5-72B-Instruct-128K** 模型生成回答。**
  * **发送请求，从 21:53:00,282 到 21:53:02,150，约 1.868 秒。**
* **结果**：**response** 是模型生成的回答，比如：

  ```text
  刘芳是MIS部门的一员，性别女，爱好包括瑜伽和绘画，电话号码是13711223344，籍贯是广东深圳。
  ```

---

**3.5 返回回答（**rag.py**）**

**代码位置**：**rag.py**

**python**

```python
if response isnotNone:
    logger.info("成功生成回答")
return response
else:
    logger.warning("生成回答失败")
return"抱歉，我暂时无法回答这个问题。"
```

**日志输出**：

```text
2025-02-19 21:53:02,151 - rag_system - INFO - [rag.py:41:chat] - 成功生成回答
```

**原理讲解**：

* **做什么？** 检查 **response** 是否有效，如果是，就返回给用户。
* **结果**：**msg** 变量接收到回答。

---

**3.6 输出结果（**index.py**）**

**代码位置**：**index.py**

**python**

```python
print('问题1：请介绍下刘芳')
print('回答：', msg)
print('\n')
```

**输出示例**：

```text
问题1：请介绍下刘芳
回答： 刘芳是MIS部门的一员，性别女，爱好包括瑜伽和绘画，电话号码是13711223344，籍贯是广东深圳。
```

**原理讲解**：

* **做什么？** 把问题和回答打印到屏幕上。
* **结果**：用户看到刘芳的介绍。

---

**四、总结**

**从启动到回答 **请介绍下刘芳**，整个流程是：**

1. **初始化**：
   * **创建日志记录器。**
   * **初始化 RAG 和知识库。**
   * **第一次运行时，向量化知识库并保存。**
2. **处理查询**：
   * **接收消息。**
   * **检索相关上下文（用向量相似度）。**
   * **生成回答并返回。**

**关键原理**：

* **向量**：把文字变成数字，让计算机理解。
* **相似度**：找到最匹配的内容。
* **API**：借助外部模型生成回答。

**希望这篇详细的教程让你明白 RAG 系统是怎么工作的！如果有疑问，随时问我哦！**
