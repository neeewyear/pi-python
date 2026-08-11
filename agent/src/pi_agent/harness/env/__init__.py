"""执行环境子包（对应 ``harness/env/``）。

包含：
- ``node_fs.py`` — 文件系统操作（``aiofiles`` 实现 ``FileSystem`` 协议）
- ``node.py`` — Shell 执行 + ``NodeExecutionEnv``（组合 ``FileSystem`` + ``Shell``）
"""

from . import node, node_fs

__all__ = ["node", "node_fs"]
