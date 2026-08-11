#!/usr/bin/env python3
"""Generate templates.py by reading source template files."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(
    "/Users/alex/code/学习/pi-python/pi/packages/coding-agent/src/core/export-html"
)
OUTPUT = Path(
    "/Users/alex/code/学习/pi-python/coding-agent/src/pi_coding_agent/core/export_html/templates.py"
)


def read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


TRIPLE_DQ = '"""'
TRIPLE_SQ = "'''"


def escape_for_triple_quotes(value: str, quote_char: str) -> str:
    if quote_char == '"':
        return value.replace(TRIPLE_DQ, '\\"\\"\\"')
    else:
        return value.replace(TRIPLE_SQ, "\\'\\'\\'")


def main() -> None:
    template_html = read_file(REPO_ROOT / "template.html")
    template_css_raw = read_file(REPO_ROOT / "template.css")
    template_js = read_file(REPO_ROOT / "template.js")
    marked_js = read_file(REPO_ROOT / "vendor" / "marked.min.js")
    highlight_js = read_file(REPO_ROOT / "vendor" / "highlight.min.js")

    template_css = (
        "<!-- template_css_start -->\n"
        + template_css_raw
        + "\n<!-- template_css_end -->"
    )

    lines = []
    lines.append('"""HTML 导出模板字符串常量。')
    lines.append("")
    lines.append("此文件包含嵌入的 HTML、CSS、JS 模板字符串，")
    lines.append("以及 vendored 的 marked.min.js 和 highlight.min.js 库。")
    lines.append('"""')
    lines.append("")
    lines.append("from __future__ import annotations")
    lines.append("")

    # Helper to write a variable with triple-quoted string
    def write_var(name: str, value: str) -> None:
        has_double = TRIPLE_DQ in value
        has_single = TRIPLE_SQ in value

        if has_double and has_single:
            # Both present - use """ and escape """ sequences
            escaped = escape_for_triple_quotes(value, '"')
            lines.append(f'{name} = """')
            lines.append(escaped)
            lines.append('"""')
        elif has_double:
            # Only """ present - use '''
            lines.append(f"{name} = '''")
            lines.append(value)
            lines.append("'''")
        else:
            # Neither present - use """ (preferred for readability)
            lines.append(f'{name} = """')
            lines.append(value)
            lines.append('"""')
        lines.append("")

    write_var("TEMPLATE_HTML", template_html)
    write_var("TEMPLATE_CSS", template_css)
    write_var("TEMPLATE_JS", template_js)
    write_var("MARKED_JS", marked_js)
    write_var("HIGHLIGHT_JS", highlight_js)

    output = "\n".join(lines)
    OUTPUT.write_text(output, encoding="utf-8")
    print(
        f"Generated {OUTPUT} ({len(output)} bytes, ~{output.count(chr(10)) + 1} lines)"
    )


if __name__ == "__main__":
    main()
