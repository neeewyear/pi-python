# Pi TUI

Terminal User Interface library with differential rendering — Python port of `@earendil-works/pi-tui` (TypeScript)。

## 项目概况

一个轻量级的终端 UI 库，基于差分渲染（differential rendering）算法，仅将变化的部分写入终端，实现高性能 TUI 渲染：

- **差分渲染引擎** — `TUI` 主类，通过对比前后帧的状态差异，仅输出变化部分到终端
- **组件系统** — `Component` 协议定义的渲染/交互组件：Box、Text、Input、Editor、Markdown、Stack、ScrollView、SelectList、SettingsList、Loader、Image 等
- **布局系统** — 基于 Flexbox 的布局引擎（`Layout` / `LayoutNode`），支持水平/垂直/居中对齐
- **编辑器** — 全功能文本编辑器组件（`Editor` / `EditorComponent`），支持语法高亮、自动补全、斜杠命令、快捷键绑定、多行编辑
- **终端支持** — Kitty 图像协议、终端颜色探测（OSC 11/4）、256 色/真彩色检测
- **Markdown 渲染** — 基于 mistune 的 Markdown 渲染（支持代码块、标题、列表、表格等）
- **LaTeX 渲染** — 基础 LaTeX 数学公式渲染
- **模糊搜索** — 内置模糊匹配/过滤（`fuzzy_filter` / `fuzzy_match`）
- **类型安全** — 全量 Pydantic 风格 + mypy 严格模式

## 架构

```
pi_tui/
├── tui.py                        # TUI 主类（差分渲染引擎）
├── tui_alt_screen.py             # 备用屏幕（Alt Screen）管理
├── layout.py / layout_node.py    # Flexbox 布局引擎
├── terminal.py                   # 终端抽象（ProcessTerminal）
├── terminal_colors.py            # 终端颜色探测
├── terminal_image.py             # Kitty 图像协议支持
│
├── components/                   # UI 组件
│   ├── __init__.py
│   ├── box.py                    # 边框盒子
│   ├── text.py                   # 文本
│   ├── truncated_text.py         # 截断文本
│   ├── h_stack.py / v_stack.py   # 水平/垂直堆栈
│   ├── stack.py                  # 通用堆栈
│   ├── spacer.py                 # 空白填充
│   ├── input.py                  # 输入框
│   ├── editor.py                 # 编辑器（多行输入）
│   ├── markdown.py               # Markdown 渲染
│   ├── scroll_view.py            # 滚动视图
│   ├── select_list.py            # 选择列表
│   ├── settings_list.py          # 设置列表
│   ├── loader.py                 # 加载动画
│   ├── cancellable_loader.py     # 可取消加载动画
│   ├── image.py                  # 图片渲染
│   └── alt_screen_flash.py       # 备用屏幕闪烁
│
├── editor_component.py           # 编辑器组件协议
├── autocomplete.py               # 自动补全（斜杠命令 + 路径）
├── keybindings.py                # 快捷键绑定管理
├── keys.py                       # 按键解析
├── kill_ring.py                  # 剪切环
├── stdin_buffer.py               # 标准输入缓冲区
├── undo_stack.py                 # 撤消栈
├── alt_screen_search.py          # 备用屏幕搜索
├── fuzzy.py                      # 模糊搜索
├── latex.py                      # LaTeX 数学公式渲染
└── utils.py                      # 工具函数（ANSI / 宽度 / 换行）
```

## 安装

```bash
conda activate pi_env

# 安装 pi-tui（无其他 pi 包依赖）
cd tui
pip install -e ".[dev]"
```

## 快速开始

```python
import asyncio
from pi_tui import TUI, ProcessTerminal, Text, Box, VStack, HStack, Spacer

async def main():
    # 创建终端和 TUI 实例
    terminal = ProcessTerminal()
    tui = TUI(terminal, width=80, height=24)

    # 构建 UI
    header = Box(Text(" Pi TUI Demo "), width=80)
    content = VStack([
        Text("Hello, World!"),
        Spacer(),
        Text("Press Ctrl+C to exit", style="dim"),
    ])
    layout = VStack([header, Spacer(), content])

    # 渲染
    tui.render(layout)
    tui.commit()

    # 事件循环
    async for event in terminal.events():
        if event.type == "key" and event.key == "ctrl+c":
            break

    terminal.cleanup()

asyncio.run(main())
```

## 开发

```bash
# 类型检查
conda run -n pi_env python -m mypy src/pi_tui/ --strict

# 运行测试
conda run -n pi_env python -m pytest tests/ -v
```

## 技术栈

- Python 3.11+
- wcwidth（终端字符宽度）
- mistune（Markdown 解析）
- pytest + pytest-asyncio（测试）

## 许可证

Apache 2.0