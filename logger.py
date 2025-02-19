import os
import logging
from datetime import datetime

# 创建logs目录（如果不存在）
if not os.path.exists('logs'):
    os.makedirs('logs')

# 生成日志文件名（使用当前时间戳）
log_filename = f'logs/rag_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

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