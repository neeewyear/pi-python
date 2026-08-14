"""上下文压缩子包。

包含：
- ``utils.py`` — 文件操作跟踪 + 序列化
- ``compaction_token.py`` — token 估算 + 切点查找
- ``compaction_summary.py`` — 摘要生成 + LLM 提示词
- ``compaction.py`` — 压缩主流程
- ``branch_summarization.py`` — 分支摘要
"""

from . import (
    branch_summarization,
    compaction,
    compaction_summary,
    compaction_token,
    utils,
)

__all__ = [
    "branch_summarization",
    "compaction",
    "compaction_summary",
    "compaction_token",
    "utils",
]