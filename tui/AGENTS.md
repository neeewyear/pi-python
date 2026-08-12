# Pi TUI — 开发指南

## 项目介绍

Terminal User Interface library with differential rendering，Python port of `@earendil-works/pi-tui` (TypeScript)。提供差分渲染引擎、组件系统、布局引擎、编辑器和自动补全等终端 UI 能力。作为底层 TUI 库，被 `pi-coding-agent` 的交互模式消费。

## 开发环境

```bash
conda activate pi_env

# 安装 pi-tui（无其他 pi 包依赖）
cd tui
pip install -e ".[dev]"
```

## 目录结构

```
tui/
├── src/pi_tui/                     # 源码
│   ├── __init__.py                 # 导出面
│   ├── tui.py                      # TUI 主类（差分渲染引擎）
│   ├── tui_alt_screen.py           # 备用屏幕管理
│   ├── layout.py / layout_node.py  # Flexbox 布局引擎
│   ├── terminal.py                 # 终端抽象
│   ├── terminal_colors.py          # 终端颜色探测
│   ├── terminal_image.py           # Kitty 图像协议
│   │
│   ├── components/                 # UI 组件（17 个）
│   │   ├── box.py / text.py / truncated_text.py
│   │   ├── h_stack.py / v_stack.py / stack.py / spacer.py
│   │   ├── input.py / editor.py / markdown.py / scroll_view.py
│   │   ├── select_list.py / settings_list.py
│   │   ├── loader.py / cancellable_loader.py / image.py
│   │   └── alt_screen_flash.py
│   │
│   ├── editor_component.py / autocomplete.py / keybindings.py
│   ├── keys.py / kill_ring.py / stdin_buffer.py / undo_stack.py
│   ├── alt_screen_search.py / fuzzy.py / latex.py
│   └── utils.py
│
├── tests/                          # 测试
├── pyproject.toml                  # 项目配置
└── README.md                       # 使用文档
```

## 核心约定

### 差分渲染
- `TUI` 维护一个 `Component` 树，每次 `render()` 计算当前帧与上一帧的差异。
- 仅将变化的屏幕区域（脏矩形）写入终端，避免全屏重绘。
- `commit()` 方法将累积的脏区域刷新到终端。

### 组件协议
- 所有组件实现 `Component` 协议（`render(width, height) -> list[list[StyledChar]]`）。
- `Focusable` 协议用于可获取焦点的组件（Input、Editor、SelectList 等）。
- `Container` 协议用于包含子组件的容器（Box、Stack、ScrollView 等）。

### 布局系统
- 基于 Flexbox 的布局引擎，支持 `width`、`height`、`grow`、`shrink`、`direction`、`align` 等属性。
- 布局计算在 `LayoutNode` 中完成，`Layout` 负责整体布局编排。

### 编辑器
- `Editor` 组件支持多行文本编辑、语法高亮、自动补全、快捷键绑定。
- 自动补全通过 `AutocompleteProvider` 实现，支持斜杠命令和路径补全。
- 快捷键通过 `EditorKeybindingsManager` 管理，支持自定义绑定。

### 类型安全
- 全项目遵守 `mypy --strict`，零 `Any`。
- `py.typed` 已随包发布。

### 代码风格
- PEP 8：函数/变量 `snake_case`，类 `PascalCase`。
- 公共方法必须有 docstring（中文）。
- 对应 TS 实现的方法注释标注 `（对应 TS ``xxx``）` 便于溯源。

## 常用命令

```bash
# 类型检查（strict）
conda run -n pi_env python -m mypy src/pi_tui/ --strict

# 运行测试
conda run -n pi_env python -m pytest tests/ -v
```

## 任务完成约定

- 任务完成后更新 `todo/` 下的规划文档状态。
- 结构性改动需同步更新 `README.md` 与本文档。