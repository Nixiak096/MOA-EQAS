"""测试包。

运行方式（在项目根目录）：
    python -m unittest discover -s tests -v     # 全部用例
    python -m unittest tests.test_indicators -v # 单个文件
装了 pytest 也可直接 pytest tests/ —— 用例是 unittest 风格，两者都认。

设计原则：
  * 默认不调用大模型（AGENT_USE_LLM=False），保证离线、快速、结果可复现；
  * 依赖 MySQL / Milvus 的用例在环境不可用时**跳过**而不是失败，
    这样「没装数据库」时仍能跑纯逻辑用例；
  * 只读断言为主，唯一的写操作（知识库上传）会在用例内自行清理。
"""

import os
import sys

# 允许从任意目录运行测试
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
