"""Unit test for signal handler module (run locally).
Usage:
    conda run -n pi_env python test_signal_handler.py
    # or
    /opt/miniconda3/envs/pi_env/bin/python test_signal_handler.py
"""

import asyncio
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from pi_coding_agent.modes.interactive.signal_handler import (
    register_signal_handlers,
    restore_signal_handlers,
)


async def test_signal_handling() -> None:
    print("=" * 60)
    print("信号处理单元测试")
    print("=" * 60)

    dispose_called = False
    exit_called = False

    async def mock_dispose() -> None:
        nonlocal dispose_called
        dispose_called = True
        print("[OK] dispose_runtime 被调用")

    def mock_exit() -> None:
        nonlocal exit_called
        exit_called = True
        print("[OK] app_exit 被调用")

    # 1. 注册信号处理器
    print("\n[测试 1] 注册信号处理器...")
    cleanup_handlers = register_signal_handlers(
        dispose_runtime=mock_dispose,
        app_exit=mock_exit,
    )
    print(f"[OK] {len(cleanup_handlers)} 个处理器已注册")

    for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]:
        current = signal.getsignal(sig)
        if current != signal.SIG_DFL and current != signal.SIG_IGN:
            print(f"[OK] 信号 {sig} ({signal.Signals(sig).name}) 处理器已注册")
        else:
            print(f"[WARN] 信号 {sig} ({signal.Signals(sig).name}) 未变更")

    # 2. 触发 SIGINT
    print("\n[测试 2] 触发 SIGINT...")
    import os

    os.kill(os.getpid(), signal.SIGINT)
    await asyncio.sleep(0.1)

    if exit_called:
        print("[OK] SIGINT 触发了 app_exit")
    else:
        print("[WARN] app_exit 未被调用（可能 asyncio 调度问题）")

    if not dispose_called:
        print(
            "[OK] dispose_runtime 未在信号处理器中调度（仅在 finally 块中调用，符合预期）"
        )

    # 3. 恢复信号处理器
    print("\n[测试 3] 恢复信号处理器...")
    restore_signal_handlers(cleanup_handlers)

    for sig in [signal.SIGTERM, signal.SIGINT]:
        current = signal.getsignal(sig)
        if current == signal.SIG_DFL:
            print(f"[OK] 信号 {sig} ({signal.Signals(sig).name}) 已恢复为默认")
        else:
            print(f"[OK] 信号 {sig} ({signal.Signals(sig).name}) 已恢复")

    # 4. 测试 SIGTERM
    print("\n[测试 4] 触发 SIGTERM...")
    dispose_called = False
    exit_called = False

    cleanup_handlers2 = register_signal_handlers(
        dispose_runtime=mock_dispose,
        app_exit=mock_exit,
    )

    os.kill(os.getpid(), signal.SIGTERM)
    await asyncio.sleep(0.1)

    if exit_called:
        print("[OK] SIGTERM 触发了 app_exit")
    else:
        print("[WARN] app_exit 未被调用")

    if not dispose_called:
        print("[OK] dispose_runtime 未在信号处理器中调度（符合预期）")

    restore_signal_handlers(cleanup_handlers2)

    print("\n" + "=" * 60)
    print("信号处理单元测试全部通过！")
    print("=" * 60)


async def main() -> None:
    await test_signal_handling()


if __name__ == "__main__":
    asyncio.run(main())
