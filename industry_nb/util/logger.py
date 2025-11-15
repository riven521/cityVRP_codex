# utils/logger.py
import os
from loguru import logger
from datetime import datetime

# 根据操作系统确定日志路径
if os.name == 'nt':  # Windows
    log_dir = "Log"
    log_file = f"Log/log_{datetime.now().strftime('%Y%m%d')}.log"
else:  # Linux, MacOS, or in a Docker container
    log_dir = "/app/logs"
    log_file = "/app/logs/log_{time:YYYY-MM-DD}.log"

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
    log_dir = os.path.join(BASE_DIR, "logs")
    log_file = os.path.join(log_dir, "log_{time:YYYY-MM-DD}.log")


# 确保日志目录存在
os.makedirs(log_dir, exist_ok=True)

# 移除Loguru的默认配置（如果不需要输出到控制台，可以移除这步）
logger.remove(0) # 移除默认的sink（控制台输出）

# 添加你的自定义配置，这只需要执行一次
logger.add(
    sink=log_file,
    rotation="00:00",       # 每天午夜创建新文件
    retention="7 days",     # 保留最近7天的日志
    encoding="utf-8",
    enqueue=True,          # 多进程/多线程安全
    backtrace=True,        # 允许记录异常回溯
    diagnose=True,         # 显示详细的变量值用于调试
    mode="a"
)


# 导出配置好的logger
__all__ = ['logger']