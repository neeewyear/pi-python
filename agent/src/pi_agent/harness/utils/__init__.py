"""工具函数。

- ``truncate``：文本截断算法（truncateHead / truncateTail / truncateLine / formatSize）
- ``shell_output``：Shell 输出捕获（executeShellWithCapture / sanitizeBinaryOutput）
"""

from . import truncate, shell_output

__all__ = ["truncate", "shell_output"]