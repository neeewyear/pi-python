#!/bin/bash
# 交互模式信号处理测试脚本
# 使用 script 创建 PTY，通过子进程 PID 发送 SIGINT
#
# 用法：
#   bash test_interactive_signal.sh

set -e

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/opt/miniconda3/envs/pi_env/bin/python"

echo "============================================"
echo " Interactive Mode 信号处理测试"
echo "============================================"
echo ""

# 1. 启动交互模式（使用 script 创建 PTY）
echo "[1/4] 启动交互模式..."
TMP_OUTPUT=$(mktemp)
script -q /dev/null "$PYTHON" -m src.pi_coding_agent > "$TMP_OUTPUT" 2>&1 &
SCRIPT_PID=$!
echo "  script PID: $SCRIPT_PID"

# 等待脚本的子进程（Python）启动
sleep 2

# 获取 script 的子进程 PID（真正的 Python 进程）
# macOS: ps -o pid= -ppid $SCRIPT_PID 获取子进程
PYTHON_PID=$(ps -o pid= -ppid "$SCRIPT_PID" 2>/dev/null | head -1 | tr -d ' ')
if [ -z "$PYTHON_PID" ]; then
    echo "[FAIL] 无法获取 Python 子进程 PID"
    cat "$TMP_OUTPUT"
    rm -f "$TMP_OUTPUT"
    exit 1
fi
echo "  Python PID: $PYTHON_PID"

# 等待 TUI 启动
sleep 3

# 检查 Python 进程是否还在运行
if ! kill -0 "$PYTHON_PID" 2>/dev/null; then
    echo "[FAIL] 进程已提前退出"
    cat "$TMP_OUTPUT"
    rm -f "$TMP_OUTPUT"
    wait "$SCRIPT_PID" 2>/dev/null
    exit 1
fi
echo "  [OK] 进程运行中"
rm -f "$TMP_OUTPUT"

# 2. 发送 SIGINT 给真正的 Python 进程
echo ""
echo "[2/4] 发送 SIGINT (Ctrl+C) 给 Python 进程..."
kill -SIGINT "$PYTHON_PID"

# 3. 等待退出
echo ""
echo "[3/4] 等待进程退出..."
sleep 3

if kill -0 "$PYTHON_PID" 2>/dev/null; then
    echo "[FAIL] 收到 SIGINT 后进程未退出（3 秒超时）"
    kill -SIGTERM "$PYTHON_PID" 2>/dev/null
    sleep 1
    kill -KILL "$PYTHON_PID" 2>/dev/null
    exit 1
fi

wait "$SCRIPT_PID" 2>/dev/null
EXIT_CODE=$?

# 4. 验证结果
echo ""
echo "[4/4] 结果验证..."
if [ "$EXIT_CODE" -eq 0 ]; then
    echo "[PASS] 进程正常退出（exit code = 0）"
    echo "[PASS] 信号处理机制生效 ✅"
else
    echo "[INFO] 进程退出码: $EXIT_CODE"
fi

echo ""
echo "============================================"
echo " 测试完成"
echo "============================================"