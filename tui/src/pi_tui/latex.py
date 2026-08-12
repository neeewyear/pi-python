"""LaTeX 数学渲染器 — 对标 packages/tui/src/latex.ts

将 LaTeX 数学表达式渲染为终端友好的 Unicode 文本。
- 纯 Python 文本近似方案：与 latex.ts 一致，不依赖任何外部 LaTeX 渲染器
  （KaTeX / MathJax / 系统命令），仅使用 utils.visible_width 计算终端列宽。
- 支持的语法：符号表（希腊字母/运算符/关系符）、上下标（Unicode 上/下标字符）、
  \\frac（display 模式纵向堆叠）、\\sqrt、矩阵/行列式环境、cases 等。
- 不支持或语法错误的输入返回 None（与 TS 返回 undefined 对齐）。

对外 API：
- render_latex(source: str, display: bool = False) -> str | None
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, NamedTuple

from .utils import visible_width

# ─────────────────────────────────────────────────────────────────────────────
# 符号与命令表（逐条对照 latex.ts 的 SYMBOLS / NAMED_OPERATORS 等常量）
# ─────────────────────────────────────────────────────────────────────────────

SYMBOLS: dict[str, str] = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ϵ",
    "varepsilon": "ε",
    "zeta": "ζ",
    "eta": "η",
    "theta": "θ",
    "vartheta": "ϑ",
    "iota": "ι",
    "kappa": "κ",
    "varkappa": "ϰ",
    "lambda": "λ",
    "mu": "μ",
    "nu": "ν",
    "xi": "ξ",
    "pi": "π",
    "varpi": "ϖ",
    "rho": "ρ",
    "varrho": "ϱ",
    "sigma": "σ",
    "varsigma": "ς",
    "tau": "τ",
    "upsilon": "υ",
    "phi": "ϕ",
    "varphi": "φ",
    "chi": "χ",
    "psi": "ψ",
    "omega": "ω",
    "Gamma": "Γ",
    "Delta": "Δ",
    "Theta": "Θ",
    "Lambda": "Λ",
    "Xi": "Ξ",
    "Pi": "Π",
    "Sigma": "Σ",
    "Upsilon": "Υ",
    "Phi": "Φ",
    "Psi": "Ψ",
    "Omega": "Ω",
    "pm": "±",
    "mp": "∓",
    "times": "×",
    "div": "÷",
    "cdot": "·",
    "ast": "∗",
    "star": "⋆",
    "circ": "∘",
    "bullet": "•",
    "oplus": "⊕",
    "ominus": "⊖",
    "otimes": "⊗",
    "oslash": "⊘",
    "odot": "⊙",
    "bigcirc": "○",
    "dagger": "†",
    "ddagger": "‡",
    "amalg": "⨿",
    "uplus": "⊎",
    "sqcap": "⊓",
    "sqcup": "⊔",
    "triangleleft": "◁",
    "triangleright": "▷",
    "wr": "≀",
    "cap": "∩",
    "cup": "∪",
    "bigcap": "⋂",
    "bigcup": "⋃",
    "bigwedge": "⋀",
    "bigvee": "⋁",
    "bigsqcup": "⨆",
    "biguplus": "⨄",
    "bigoplus": "⨁",
    "bigotimes": "⨂",
    "bigodot": "⨀",
    "setminus": "∖",
    "in": "∈",
    "notin": "∉",
    "ni": "∋",
    "subset": "⊂",
    "supset": "⊃",
    "subseteq": "⊆",
    "supseteq": "⊇",
    "sqsubset": "⊏",
    "sqsupset": "⊐",
    "sqsubseteq": "⊑",
    "sqsupseteq": "⊒",
    "prec": "≺",
    "preceq": "≼",
    "succ": "≻",
    "succeq": "≽",
    "ll": "≪",
    "gg": "≫",
    "le": "≤",
    "leq": "≤",
    "leqslant": "≤",
    "ge": "≥",
    "geq": "≥",
    "geqslant": "≥",
    "ne": "≠",
    "neq": "≠",
    "equiv": "≡",
    "approx": "≈",
    "sim": "∼",
    "simeq": "≃",
    "cong": "≅",
    "asymp": "≍",
    "doteq": "≐",
    "propto": "∝",
    "parallel": "∥",
    "perp": "⊥",
    "mid": "∣",
    "vdash": "⊢",
    "dashv": "⊣",
    "models": "⊨",
    "Vdash": "⊩",
    "Vvdash": "⊪",
    "nvdash": "⊬",
    "nvDash": "⊭",
    "forall": "∀",
    "exists": "∃",
    "nexists": "∄",
    "neg": "¬",
    "land": "∧",
    "wedge": "∧",
    "lor": "∨",
    "vee": "∨",
    "to": "→",
    "rightarrow": "→",
    "longrightarrow": "→",
    "leftarrow": "←",
    "longleftarrow": "←",
    "gets": "←",
    "leftrightarrow": "↔",
    "longleftrightarrow": "↔",
    "hookleftarrow": "↩",
    "hookrightarrow": "↪",
    "twoheadleftarrow": "↞",
    "twoheadrightarrow": "↠",
    "leftharpoonup": "↼",
    "leftharpoondown": "↽",
    "rightharpoonup": "⇀",
    "rightharpoondown": "⇁",
    "rightleftharpoons": "⇌",
    "leftrightharpoons": "⇋",
    "nearrow": "↗",
    "searrow": "↘",
    "swarrow": "↙",
    "nwarrow": "↖",
    "rightsquigarrow": "⇝",
    "leadsto": "⇝",
    "Rightarrow": "⇒",
    "Longrightarrow": "⇒",
    "Leftarrow": "⇐",
    "Longleftarrow": "⇐",
    "Leftrightarrow": "⇔",
    "Longleftrightarrow": "⇔",
    "implies": "⇒",
    "iff": "⇔",
    "mapsto": "↦",
    "longmapsto": "↦",
    "uparrow": "↑",
    "downarrow": "↓",
    "partial": "∂",
    "nabla": "∇",
    "int": "∫",
    "iint": "∬",
    "iiint": "∭",
    "oint": "∮",
    "sum": "∑",
    "prod": "∏",
    "coprod": "∐",
    "infty": "∞",
    "emptyset": "∅",
    "varnothing": "∅",
    "angle": "∠",
    "therefore": "∴",
    "because": "∵",
    "aleph": "ℵ",
    "beth": "ℶ",
    "gimel": "ℷ",
    "daleth": "ℸ",
    "top": "⊤",
    "bot": "⊥",
    "triangle": "△",
    "square": "□",
    "lozenge": "◊",
    "checkmark": "✓",
    "complement": "∁",
    "wp": "℘",
    "prime": "′",
    "ldots": "…",
    "dots": "…",
    "cdots": "⋯",
    "vdots": "⋮",
    "ddots": "⋱",
    "ell": "ℓ",
    "hbar": "ℏ",
    "Im": "ℑ",
    "Re": "ℜ",
    "langle": "⟨",
    "rangle": "⟩",
    "vert": "|",
    "lvert": "|",
    "rvert": "|",
    "Vert": "‖",
    "lVert": "‖",
    "rVert": "‖",
    "lbrace": "{",
    "rbrace": "}",
    "backslash": "\\",
    "lfloor": "⌊",
    "rfloor": "⌋",
    "lceil": "⌈",
    "rceil": "⌉",
    "colon": ":",
}

NAMED_OPERATORS: frozenset[str] = frozenset(
    {
        "arccos",
        "arcsin",
        "arctan",
        "arg",
        "cos",
        "cosh",
        "cot",
        "coth",
        "csc",
        "deg",
        "det",
        "dim",
        "exp",
        "gcd",
        "hom",
        "inf",
        "ker",
        "lg",
        "lim",
        "liminf",
        "limsup",
        "ln",
        "log",
        "max",
        "min",
        "Pr",
        "sec",
        "sin",
        "sinh",
        "sup",
        "tan",
        "tanh",
    }
)

LIMIT_OPERATORS: frozenset[str] = frozenset(
    {
        "argmax",
        "argmin",
        "inf",
        "injlim",
        "lim",
        "liminf",
        "limsup",
        "max",
        "min",
        "projlim",
        "sup",
    }
)

DISPLAY_LIMIT_SYMBOLS: frozenset[str] = frozenset(
    {
        "bigcap",
        "bigcup",
        "bigodot",
        "bigoplus",
        "bigotimes",
        "bigsqcup",
        "biguplus",
        "bigvee",
        "bigwedge",
        "coprod",
        "int",
        "iint",
        "iiint",
        "oint",
        "prod",
        "sum",
    }
)

RELATION_COMMANDS: frozenset[str] = frozenset(
    {
        "Leftarrow",
        "Leftrightarrow",
        "Longleftarrow",
        "Longleftrightarrow",
        "Longrightarrow",
        "Rightarrow",
        "Vdash",
        "Vvdash",
        "approx",
        "asymp",
        "cong",
        "dashv",
        "doteq",
        "downarrow",
        "equiv",
        "ge",
        "geq",
        "geqslant",
        "gets",
        "gg",
        "hookleftarrow",
        "hookrightarrow",
        "iff",
        "implies",
        "in",
        "leadsto",
        "le",
        "leftarrow",
        "leftharpoondown",
        "leftharpoonup",
        "leftrightarrow",
        "leftrightharpoons",
        "leq",
        "leqslant",
        "ll",
        "longleftarrow",
        "longleftrightarrow",
        "longmapsto",
        "longrightarrow",
        "mapsto",
        "mid",
        "models",
        "ne",
        "nearrow",
        "neq",
        "ni",
        "notin",
        "nvdash",
        "nvDash",
        "nwarrow",
        "parallel",
        "perp",
        "prec",
        "preceq",
        "propto",
        "rightharpoondown",
        "rightharpoonup",
        "rightleftharpoons",
        "rightarrow",
        "rightsquigarrow",
        "searrow",
        "sim",
        "simeq",
        "sqsubset",
        "sqsubseteq",
        "sqsupset",
        "sqsupseteq",
        "subset",
        "subseteq",
        "succ",
        "succeq",
        "supset",
        "supseteq",
        "swarrow",
        "to",
        "triangleleft",
        "triangleright",
        "twoheadleftarrow",
        "twoheadrightarrow",
        "uparrow",
        "vdash",
    }
)

NEGATED_SYMBOLS: dict[str, str] = {
    "<": "≮",
    ">": "≯",
    "=": "≠",
    "∈": "∉",
    "∋": "∌",
    "∣": "∤",
    "∥": "∦",
    "∼": "≁",
    "≃": "≄",
    "≅": "≇",
    "≈": "≉",
    "≡": "≢",
    "≤": "≰",
    "≥": "≱",
    "≺": "⊀",
    "≻": "⊁",
    "⊂": "⊄",
    "⊃": "⊅",
    "⊆": "⊈",
    "⊇": "⊉",
    "⊢": "⊬",
    "⊨": "⊭",
    "↔": "↮",
    "←": "↚",
    "→": "↛",
    "⇒": "⇏",
    "⇐": "⇍",
    "⇔": "⇎",
    "≼": "⋠",
    "≽": "⋡",
}

BLACKBOARD: dict[str, str] = {
    "C": "ℂ",
    "H": "ℍ",
    "N": "ℕ",
    "P": "ℙ",
    "Q": "ℚ",
    "R": "ℝ",
    "Z": "ℤ",
}

SUPERSCRIPTS: dict[str, str] = {
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
    "+": "⁺",
    "-": "⁻",
    "=": "⁼",
    "(": "⁽",
    ")": "⁾",
    "a": "ᵃ",
    "b": "ᵇ",
    "c": "ᶜ",
    "d": "ᵈ",
    "e": "ᵉ",
    "f": "ᶠ",
    "g": "ᵍ",
    "h": "ʰ",
    "i": "ⁱ",
    "j": "ʲ",
    "k": "ᵏ",
    "l": "ˡ",
    "m": "ᵐ",
    "n": "ⁿ",
    "o": "ᵒ",
    "p": "ᵖ",
    "r": "ʳ",
    "s": "ˢ",
    "t": "ᵗ",
    "u": "ᵘ",
    "v": "ᵛ",
    "w": "ʷ",
    "x": "ˣ",
    "y": "ʸ",
    "z": "ᶻ",
}

SUBSCRIPTS: dict[str, str] = {
    "0": "₀",
    "1": "₁",
    "2": "₂",
    "3": "₃",
    "4": "₄",
    "5": "₅",
    "6": "₆",
    "7": "₇",
    "8": "₈",
    "9": "₉",
    "+": "₊",
    "-": "₋",
    "=": "₌",
    "(": "₍",
    ")": "₎",
    "a": "ₐ",
    "e": "ₑ",
    "h": "ₕ",
    "i": "ᵢ",
    "j": "ⱼ",
    "k": "ₖ",
    "l": "ₗ",
    "m": "ₘ",
    "n": "ₙ",
    "o": "ₒ",
    "p": "ₚ",
    "r": "ᵣ",
    "s": "ₛ",
    "t": "ₜ",
    "u": "ᵤ",
    "v": "ᵥ",
    "x": "ₓ",
}

SPACING_COMMANDS: frozenset[str] = frozenset(
    {
        ",",
        ":",
        ";",
        " ",
        ">",
        "enspace",
        "enskip",
        "medspace",
        "quad",
        "qquad",
        "thickspace",
        "thinspace",
    }
)
NEGATIVE_SPACING_COMMANDS: frozenset[str] = frozenset(
    {"!", "negmedspace", "negthickspace", "negthinspace"}
)
NEGATIVE_SPACE = "\u0000"
IGNORED_COMMANDS: frozenset[str] = frozenset(
    {
        "displaystyle",
        "limits",
        "nolimits",
        "scriptstyle",
        "scriptscriptstyle",
        "textstyle",
    }
)
SIZE_COMMANDS: frozenset[str] = frozenset(
    {
        "big",
        "Big",
        "bigg",
        "Bigg",
        "bigl",
        "Bigl",
        "biggl",
        "Biggl",
        "bigr",
        "Bigr",
        "biggr",
        "Biggr",
    }
)
PLAIN_WRAPPERS: frozenset[str] = frozenset(
    {
        "emph",
        "mathcal",
        "mathbf",
        "mathfrak",
        "mathit",
        "mathrm",
        "mathnormal",
        "mathscr",
        "mathsf",
        "mathtt",
        "mathup",
        "mbox",
        "overbrace",
        "pmb",
        "smash",
        "substack",
        "text",
        "textbf",
        "textit",
        "textmd",
        "textnormal",
        "textrm",
        "textsc",
        "textsf",
        "textsl",
        "texttt",
        "textup",
        "underbrace",
        "bm",
        "boldsymbol",
    }
)
ACCENTS: dict[str, str] = {
    "acute": "\u0301",
    "bar": "\u0305",
    "breve": "\u0306",
    "check": "\u030c",
    "ddot": "\u0308",
    "dot": "\u0307",
    "grave": "\u0300",
    "hat": "\u0302",
    "mathring": "\u030a",
    "overleftarrow": "\u20d6",
    "overleftrightarrow": "\u20e1",
    "overline": "\u0305",
    "overrightarrow": "\u20d7",
    "tilde": "\u0303",
    "underline": "\u0332",
    "vec": "\u20d7",
    "widehat": "\u0302",
    "widetilde": "\u0303",
}

# ─────────────────────────────────────────────────────────────────────────────
# 私有区标记字符（对应 TS 的 \u{f0000} ~ \u{f0005} 等布局/命名操作符标记）
# ─────────────────────────────────────────────────────────────────────────────

NAMED_OPERATOR_START = "\U000f0004"
NAMED_OPERATOR_END = "\U000f0005"
LAYOUT_MARKER_START = "\U000f0000"
LAYOUT_MARKER_END = "\U000f0001"
PROTECTED_SPACE = "\U000f0002"

_LAYOUT_MARKER_RE = re.compile(rf"{LAYOUT_MARKER_START}(\d+){LAYOUT_MARKER_END}")
_TRAILING_LAYOUT_MARKER_RE = re.compile(
    rf"{LAYOUT_MARKER_START}(\d+){LAYOUT_MARKER_END}$"
)
_LIMITS_MODIFIER_RE = re.compile(r"\\(limits|nolimits)(?![A-Za-z])")
_ENV_ROW_SPLIT_RE = re.compile(r"\\\\(?:\[[^\]\n]*\])?")


def _is_ascii_letter(ch: str) -> bool:
    """是否 ASCII 字母（对应 TS 的 /[A-Za-z]/）。"""
    return ("A" <= ch <= "Z") or ("a" <= ch <= "z")


def _is_ascii_letters(value: str) -> bool:
    """是否全部为 ASCII 字母（对应 TS 的 /^[A-Za-z]+$/）。"""
    return bool(value) and all(_is_ascii_letter(ch) for ch in value)


def _is_math_char(ch: str) -> bool:
    """是否属于 \\p{L} 或 \\p{N}（Unicode 字母或数字，对应 TS 字符类 [\\p{L}\\p{N}]）。"""
    return ch.isalnum()


def _is_simple_math(value: str) -> bool:
    """对应 TS /^[\\p{L}\\p{N}.]+$/：仅由字母、数字或小数点组成。"""
    return bool(value) and all(_is_math_char(ch) or ch == "." for ch in value)


def _is_simple_number(value: str) -> bool:
    """对应 TS /^[\\p{N}.]+$/：仅由数字或小数点组成。"""
    return bool(value) and all(ch.isnumeric() or ch == "." for ch in value)


def _replace_characters(value: str, replacements: dict[str, str]) -> str | None:
    """逐个字符查替换表；任一字符未命中返回 None（对应 TS replaceCharacters）。"""
    result = ""
    for ch in value:
        replacement = replacements.get(ch)
        if replacement is None:
            return None
        result += replacement
    return result


def _format_script(value: str, kind: Literal["sub", "sup"]) -> str:
    """渲染上下标：优先使用 Unicode 上下标字符，否则回退为 _x / ^x / _(x) / ^(x)。"""
    value = value.strip()
    replacements = SUBSCRIPTS if kind == "sub" else SUPERSCRIPTS
    unicode_value = _replace_characters(
        re.sub(r"\s*([=+-])\s*", r"\1", value), replacements
    )
    if unicode_value is not None:
        return unicode_value

    prefix = "_" if kind == "sub" else "^"
    if len(value) == 1 or (kind == "sub" and _is_ascii_letters(value)):
        return f"{prefix}{value}"
    return f"{prefix}({value})"


def _format_fraction(numerator: str, denominator: str) -> str:
    """行内分数的文本近似：简单内容直接写 a/b，复杂内容加括号。"""
    numerator = numerator.strip()
    denominator = denominator.strip()
    simple_numerator = _is_simple_math(numerator)
    simple_denominator = _is_simple_number(denominator) or len(denominator) == 1
    num = numerator if simple_numerator else f"({numerator})"
    den = denominator if simple_denominator else f"({denominator})"
    return f"{num}/{den}"


def _format_root(value: str, symbol: str = "√") -> str:
    """根号的文本近似：√x 或 √(x)。"""
    value = value.strip()
    return f"{symbol}{value}" if _is_simple_math(value) else f"{symbol}({value})"


def _replace_named_operator_left_spacing(value: str) -> str:
    """对应 TS NAMED_OPERATOR_LEFT_SPACING_PATTERN（(?<=[\\p{L}\\p{N])\\}]\\u{f0001}])\\u{f0004}）。

    命名操作符（如 sin/cos）左侧若紧跟字母、数字、')'、'}' 或布局标记，则补一个空格。
    """
    result = ""
    for index, ch in enumerate(value):
        if ch == NAMED_OPERATOR_START and index > 0:
            prev = value[index - 1]
            if _is_math_char(prev) or prev in ")}\U000f0001":
                result += " "
                continue
        result += ch
    return result


def _replace_named_operator_right_spacing(value: str) -> str:
    """对应 TS NAMED_OPERATOR_RIGHT_SPACING_PATTERN（\\u{f0005}(?=[\\p{L}\\p{N}√\\u{f0000}])）。

    命名操作符右侧若紧跟字母、数字、√ 或布局标记，则补一个空格。
    """
    result = ""
    for index, ch in enumerate(value):
        if ch == NAMED_OPERATOR_END and index + 1 < len(value):
            nxt = value[index + 1]
            if _is_math_char(nxt) or nxt in "√\U000f0000":
                result += " "
                continue
        result += ch
    return result


def _normalize_output(value: str) -> str:
    """输出规范化：清理命名操作符标记、合并空白、删除首尾空行（对应 TS normalizeOutput）。"""
    value = _replace_named_operator_left_spacing(value)
    value = value.replace(NAMED_OPERATOR_START, "")
    value = _replace_named_operator_right_spacing(value)
    value = value.replace(NAMED_OPERATOR_END, "")

    lines = value.split("\n")
    normalized: list[str] = []
    for index, line in enumerate(lines):
        collapsed = re.sub(r"[ \t]+", " ", line).strip()
        # 保留非空行，以及中间的（非首尾）空行
        if len(collapsed) > 0 or (0 < index < len(lines) - 1):
            normalized.append(collapsed)
    return "\n".join(normalized).strip()


# ─────────────────────────────────────────────────────────────────────────────
# 布局节点：对应 TS 的 FractionNode / OperatorNode / MatrixNode / Layout
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class FractionNode:
    type: Literal["fraction"]
    numerator: str
    denominator: str


@dataclass
class OperatorNode:
    type: Literal["operator"]
    operator: str
    lower: str | None = None
    upper: str | None = None


@dataclass
class MatrixNode:
    type: Literal["matrix"]
    lines: list[str]
    baseline: int


LayoutNode = FractionNode | OperatorNode | MatrixNode


class Layout(NamedTuple):
    lines: list[str]
    width: int
    baseline: int


def _pad_layout_line(line: str, width: int, centered: bool = False) -> str:
    """按宽度补齐布局行（可选居中），对应 TS padLayoutLine。"""
    padding = max(0, width - visible_width(line))
    left = padding // 2 if centered else 0
    return f"{' ' * left}{line}{' ' * (padding - left)}"


def _join_layouts(layouts: list[Layout]) -> Layout:
    """把多个并排布局按基线对齐拼接为一行布局，对应 TS joinLayouts。"""
    if not layouts:
        return Layout([""], 0, 0)
    baseline = max(layout.baseline for layout in layouts)
    below = max(len(layout.lines) - layout.baseline - 1 for layout in layouts)
    lines: list[str] = []
    for row in range(baseline + below + 1):
        line = ""
        for layout in layouts:
            source_row = row - baseline + layout.baseline
            if 0 <= source_row < len(layout.lines):
                line += _pad_layout_line(layout.lines[source_row], layout.width)
            else:
                line += " " * layout.width
        lines.append(line.rstrip())
    return Layout(lines, sum(layout.width for layout in layouts), baseline)


def _render_layout(source: str, nodes: list[LayoutNode]) -> Layout:
    """把渲染文本中的布局标记展开为多行布局，对应 TS renderLayout。"""
    rendered_lines: list[str] = []
    first_baseline = 0
    for source_line in source.split("\n"):
        layouts: list[Layout] = []
        position = 0
        previous_node: LayoutNode | None = None
        for match in _LAYOUT_MARKER_RE.finditer(source_line):
            index = match.start()
            node_index = int(match.group(1))
            if node_index >= len(nodes):
                continue
            node = nodes[node_index]
            if index > position:
                sliced = source_line[position:index]
                trimmed = (
                    sliced.lstrip() if previous_node is not None else sliced
                ).rstrip()
                preserve_leading = (
                    previous_node is not None
                    and previous_node.type == "matrix"
                    and bool(re.match(r"^\s", sliced))
                )
                preserve_trailing = node.type == "matrix" and bool(
                    re.search(r"\s$", sliced)
                )
                if trimmed:
                    text = (
                        f"{' ' if preserve_leading else ''}{trimmed}"
                        f"{' ' if preserve_trailing else ''}"
                    )
                else:
                    text = " " if (preserve_leading or preserve_trailing) else ""
                layouts.append(Layout([text], visible_width(text), 0))

            if node.type == "fraction":
                numerator = _render_layout(node.numerator, nodes)
                denominator = _render_layout(node.denominator, nodes)
                content_width = max(numerator.width, denominator.width, 1)
                width = content_width + 2
                layouts.append(
                    Layout(
                        [
                            *[
                                _pad_layout_line(line, width, True)
                                for line in numerator.lines
                            ],
                            f" {'─' * content_width} ",
                            *[
                                _pad_layout_line(line, width, True)
                                for line in denominator.lines
                            ],
                        ],
                        width,
                        len(numerator.lines),
                    )
                )
            elif node.type == "operator":
                content_width = max(
                    visible_width(node.operator),
                    0 if node.lower is None else visible_width(node.lower),
                    0 if node.upper is None else visible_width(node.upper),
                )
                op_lines: list[str] = []
                if node.upper is not None:
                    op_lines.append(
                        f"{_pad_layout_line(node.upper, content_width, True)} "
                    )
                op_lines.append(
                    f"{_pad_layout_line(node.operator, content_width, True)} "
                )
                if node.lower is not None:
                    op_lines.append(
                        f"{_pad_layout_line(node.lower, content_width, True)} "
                    )
                layouts.append(
                    Layout(
                        op_lines,
                        content_width + 1,
                        0 if node.upper is None else 1,
                    )
                )
            else:
                width = max(0, *(visible_width(line) for line in node.lines))
                layouts.append(
                    Layout(
                        [_pad_layout_line(line, width) for line in node.lines],
                        width,
                        node.baseline,
                    )
                )
            position = match.end()
            previous_node = node

        if position < len(source_line):
            sliced = source_line[position:]
            trimmed = sliced.lstrip() if previous_node is not None else sliced
            text = (
                f" {trimmed}"
                if previous_node is not None
                and previous_node.type == "matrix"
                and bool(re.match(r"^\s", sliced))
                else trimmed
            )
            layouts.append(Layout([text], visible_width(text), 0))

        line_layout = _join_layouts(layouts)
        if not rendered_lines:
            first_baseline = line_layout.baseline
        rendered_lines.extend(line_layout.lines)

    return Layout(
        rendered_lines,
        max(0, *(visible_width(line) for line in rendered_lines)),
        first_baseline,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 解析器：对应 TS 的 LatexParser 类
# ─────────────────────────────────────────────────────────────────────────────


class LatexParser:
    """LaTeX 数学表达式解析器（逐字符扫描 + 命令分派），对应 latex.ts 的 LatexParser。"""

    def __init__(
        self, source: str, layout_nodes: list[LayoutNode], display: bool
    ) -> None:
        self._source = source
        self._layout_nodes = layout_nodes
        self._display = display
        self._position = 0
        self._supported = True
        self._stack_fractions = True

    def render(self) -> str | None:
        """解析整个源码；遇到不支持或残缺的语法返回 None（对应 TS render()）。"""
        rendered = self._parse_sequence()
        if not self._supported or self._position != len(self._source):
            return None
        return _normalize_output(rendered)

    def _parse_sequence(self, end_character: str | None = None) -> str:
        """解析一段字符序列，直到遇到结束字符或源码末尾（对应 TS parseSequence）。"""
        result = ""
        while self._position < len(self._source):
            character = self._source[self._position]
            if end_character is not None and character == end_character:
                self._position += 1
                return result

            if character == "}":
                # 遇到未配对的 } 视为不支持
                self._supported = False
                return result

            if character == "{":
                self._position += 1
                result += self._parse_sequence("}")
                continue

            if character == "\\":
                command = self._parse_command()
                if command == NEGATIVE_SPACE:
                    # 负间距：去掉前导空白（及命名操作符标记），对应 TS 的处理
                    result = result.rstrip()
                    result = result.removesuffix(NAMED_OPERATOR_END)
                else:
                    result += command
                continue

            if character == "^" or character == "_":
                self._position += 1
                result = result.rstrip()
                script = _format_script(
                    self._parse_required_argument(False),
                    "sub" if character == "_" else "sup",
                )
                if result.endswith(NAMED_OPERATOR_END):
                    result = (
                        f"{result[: -len(NAMED_OPERATOR_END)]}{script}"
                        f"{NAMED_OPERATOR_END}"
                    )
                else:
                    result += script
                continue

            if character.isspace():
                result += self._parse_whitespace()
                continue

            if character in "=<>":
                # 关系运算符两侧加空格
                result = f"{result.rstrip()} {character} "
                self._position += 1
                continue

            if character == "&":
                self._position += 1
                continue

            if character == "~":
                self._position += 1
                result += " "
                continue

            if character == ".":
                # 矩阵后紧跟的句号追加到矩阵最后一行（对应 TS 对 trailing marker 的处理）
                marker = _TRAILING_LAYOUT_MARKER_RE.search(result)
                matrix_node: MatrixNode | None = None
                if marker is not None:
                    node_index = int(marker.group(1))
                    if node_index < len(self._layout_nodes):
                        candidate = self._layout_nodes[node_index]
                        if candidate.type == "matrix":
                            matrix_node = candidate
                if matrix_node is not None:
                    matrix_node.lines[-1] = f"{matrix_node.lines[-1]}{character}"
                    self._position += 1
                    continue

            result += character
            self._position += 1

        if end_character is not None:
            # 没等到结束字符（未闭合花括号）→ 不支持
            self._supported = False
        return result

    def _parse_whitespace(self) -> str:
        """折叠连续空白为单个空格（对应 TS parseWhitespace）。"""
        while (
            self._position < len(self._source)
            and self._source[self._position].isspace()
        ):
            self._position += 1
        return " "

    def _parse_command(self) -> str:
        """解析 \\ 开头的命令并返回其渲染结果（对应 TS parseCommand）。"""
        self._position += 1
        if self._position >= len(self._source):
            self._supported = False
            return ""

        command = ""
        first = self._source[self._position]
        if first in "\n\r":
            # \\ 换行 → 空格
            self._position += 1
            if (
                first == "\r"
                and self._position < len(self._source)
                and self._source[self._position] == "\n"
            ):
                self._position += 1
            return " "
        if _is_ascii_letter(first):
            start = self._position
            while self._position < len(self._source) and _is_ascii_letter(
                self._source[self._position]
            ):
                self._position += 1
            command = self._source[start : self._position]
        else:
            command = first
            self._position += 1

        if command == "\\":
            # \\\\ 表示换行
            return "\n"
        if command in SPACING_COMMANDS:
            return " "
        if command in NEGATIVE_SPACING_COMMANDS:
            return NEGATIVE_SPACE
        if command in IGNORED_COMMANDS:
            return ""
        if command in "{}$_#&":
            return command
        if command == "|":
            return "‖"
        if command == "not":
            value = self._parse_required_argument(False).strip()
            negated = NEGATED_SYMBOLS.get(value)
            if negated is not None:
                return f" {negated} "
            characters = list(value)
            if not characters:
                self._supported = False
                return ""
            return f" {characters[0]}\u0338{''.join(characters[1:])} "
        if command in LIMIT_OPERATORS:
            return self._parse_operator(command, "bracket", True, True)

        symbol = SYMBOLS.get(command)
        if symbol is not None:
            if command in DISPLAY_LIMIT_SYMBOLS:
                return self._parse_operator(symbol, "script", True)
            if command in ("cdot", "times") or command in RELATION_COMMANDS:
                return f" {symbol} "
            return symbol
        if command in NAMED_OPERATORS:
            return f"{NAMED_OPERATOR_START}{command}{NAMED_OPERATOR_END}"
        if command in SIZE_COMMANDS:
            return ""
        if command in ("left", "middle", "right"):
            # \left. 等空定界符：吞掉点号
            if (
                self._position < len(self._source)
                and self._source[self._position] == "."
            ):
                self._position += 1
            return ""
        if command in ("frac", "dfrac", "tfrac"):
            # display 模式下 \frac/\dfrac 纵向堆叠（保留为布局标记）
            should_stack = (
                self._display and self._stack_fractions and command != "tfrac"
            )
            numerator = self._parse_required_argument(not should_stack)
            denominator = self._parse_required_argument(not should_stack)
            if should_stack:
                self._layout_nodes.append(
                    FractionNode(
                        "fraction",
                        _normalize_output(numerator),
                        _normalize_output(denominator),
                    )
                )
                return (
                    f"{LAYOUT_MARKER_START}{len(self._layout_nodes) - 1}"
                    f"{LAYOUT_MARKER_END}"
                )
            return _format_fraction(numerator, denominator)
        if command == "sqrt":
            degree = self._parse_optional_argument()
            value = self._parse_required_argument()
            if degree is None or degree == "2":
                return _format_root(value)
            if degree == "3":
                return _format_root(value, "∛")
            if degree == "4":
                return _format_root(value, "∜")
            return f"{_format_script(degree, 'sup')}{_format_root(value)}"
        if command in ("boxed", "fbox"):
            return f"[{self._parse_required_argument().strip()}]"
        if command in ("binom", "dbinom", "tbinom"):
            return (
                f"({self._parse_required_argument()} choose "
                f"{self._parse_required_argument()})"
            )
        accent = ACCENTS.get(command)
        if accent is not None:
            value = self._parse_required_argument()
            if len(value) == 1:
                return f"{value}{accent}"
            return f"{command}({value})"
        if command == "mathbb":
            value = self._parse_required_argument()
            return "".join(BLACKBOARD.get(ch, ch) for ch in value)
        if command == "operatorname":
            starred = (
                self._position < len(self._source)
                and self._source[self._position] == "*"
            )
            if starred:
                self._position += 1
            operator = _normalize_output(self._parse_required_argument()).strip()
            return self._parse_operator(operator, "bracket", starred, True)
        if command in ("mod", "bmod"):
            return " mod "
        if command in ("pmod", "pod"):
            value = self._parse_required_argument().strip()
            if command == "pmod":
                return f" (mod {value})"
            return f" ({value})"
        if command in ("overset", "stackrel"):
            upper = self._parse_required_argument()
            value = self._parse_required_argument().strip()
            return f"{value}{_format_script(upper, 'sup')}"
        if command == "underset":
            lower = self._parse_required_argument()
            value = self._parse_required_argument().strip()
            return f"{value}{_format_script(lower, 'sub')}"
        if command in PLAIN_WRAPPERS:
            value = self._parse_required_argument()
            if command.startswith("text") or command == "mbox":
                return value
            return value.strip()
        if command == "begin":
            return self._parse_environment()
        if command == "end":
            self._supported = False
            return ""

        # 未知命令 → 不支持
        self._supported = False
        return f"\\{command}"

    def _parse_operator(
        self,
        operator: str,
        inline_lower_style: Literal["bracket", "script"],
        display_limits: bool,
        spaced: bool = False,
    ) -> str:
        """解析带上下标/limits 修饰的运算符（对应 TS parseOperator）。"""
        use_display_limits = display_limits
        modifier_position = self._position
        while (
            modifier_position < len(self._source)
            and self._source[modifier_position] in " \t"
        ):
            modifier_position += 1
        modifier = _LIMITS_MODIFIER_RE.match(self._source[modifier_position:])
        if modifier is not None:
            use_display_limits = modifier.group(1) == "limits"
            self._position = modifier_position + modifier.end()

        lower: str | None = None
        upper: str | None = None
        while True:
            script_position = self._position
            while (
                script_position < len(self._source)
                and self._source[script_position] in " \t"
            ):
                script_position += 1
            if script_position >= len(self._source):
                break
            kind = self._source[script_position]
            if kind != "_" and kind != "^":
                break
            self._position = script_position + 1
            value = _normalize_output(self._parse_required_argument(False)).replace(
                " ", ""
            )
            if kind == "_":
                if lower is not None:
                    self._supported = False
                lower = value
            else:
                if upper is not None:
                    self._supported = False
                upper = value

        if (
            self._display
            and use_display_limits
            and (lower is not None or upper is not None)
        ):
            self._layout_nodes.append(OperatorNode("operator", operator, lower, upper))
            return (
                f"{LAYOUT_MARKER_START}{len(self._layout_nodes) - 1}{LAYOUT_MARKER_END}"
            )

        rendered = operator
        if lower is not None:
            if inline_lower_style == "bracket":
                rendered += f"[{lower}]"
            else:
                rendered += _format_script(lower, "sub")
        if upper is not None:
            rendered += _format_script(upper, "sup")
        return f" {rendered} " if spaced else rendered

    def _parse_required_argument(self, stack_fractions: bool = True) -> str:
        """解析必选参数（大括号分组/单字符/命令），对应 TS parseRequiredArgument。"""
        previous_stack_fractions = self._stack_fractions
        self._stack_fractions = previous_stack_fractions and stack_fractions
        value = self._parse_required_argument_value()
        self._stack_fractions = previous_stack_fractions
        return value

    def _parse_required_argument_value(self) -> str:
        """必选参数的实际取值逻辑（对应 TS parseRequiredArgumentValue）。"""
        while (
            self._position < len(self._source)
            and self._source[self._position].isspace()
        ):
            self._position += 1
        if self._position >= len(self._source):
            self._supported = False
            return ""
        if self._source[self._position] == "{":
            self._position += 1
            return self._parse_sequence("}")
        if self._source[self._position] == "\\":
            return self._parse_command()
        value = self._source[self._position]
        self._position += 1
        return value

    def _parse_optional_argument(self) -> str | None:
        """解析可选参数 [..]（对应 TS parseOptionalArgument）。"""
        while (
            self._position < len(self._source) and self._source[self._position] in " \t"
        ):
            self._position += 1
        if self._position >= len(self._source) or self._source[self._position] != "[":
            return None
        end = self._source.find("]", self._position + 1)
        if end < 0:
            self._supported = False
            return None
        value = self._source[self._position + 1 : end]
        self._position = end + 1
        return self._render_nested(value)

    def _read_raw_group(self) -> str | None:
        """读取未渲染的原始 {..} 分组（用于环境名，对应 TS readRawGroup）。"""
        while (
            self._position < len(self._source) and self._source[self._position] in " \t"
        ):
            self._position += 1
        if self._position >= len(self._source) or self._source[self._position] != "{":
            self._supported = False
            return None

        self._position += 1
        start = self._position
        depth = 1
        while self._position < len(self._source):
            character = self._source[self._position]
            if character == "\\":
                self._position += 2
                continue
            if character == "{":
                depth += 1
            if character == "}":
                depth -= 1
            if depth == 0:
                value = self._source[start : self._position]
                self._position += 1
                return value
            self._position += 1
        self._supported = False
        return None

    def _split_environment_rows(self, body: str) -> list[str]:
        """按 \\\\ 或 \\\\[...] 切分环境行（对应 TS splitEnvironmentRows）。"""
        return _ENV_ROW_SPLIT_RE.split(body)

    def _parse_environment(self) -> str:
        """解析 \\begin{env} ... \\end{env} 环境（对应 TS parseEnvironment）。"""
        environment = self._read_raw_group()
        if environment is None:
            return ""
        end_marker = f"\\end{{{environment}}}"
        end = self._source.find(end_marker, self._position)
        if end < 0:
            self._supported = False
            return ""
        body = self._source[self._position : end]
        self._position = end + len(end_marker)

        if environment in ("equation", "equation*", "displaymath"):
            return self._render_nested(body).strip()

        if environment in (
            "aligned",
            "align",
            "align*",
            "alignedat",
            "alignat",
            "alignat*",
            "gather",
            "gathered",
            "multline",
            "multline*",
            "split",
        ):
            aligned_at = environment in ("alignedat", "alignat", "alignat*")
            aligned_body = re.sub(r"^\s*\{[^}]*\}", "", body) if aligned_at else body
            rendered_rows: list[str] = []
            for aligned_row in self._split_environment_rows(aligned_body):
                cells = aligned_row.split("&")
                if aligned_at:
                    source = " ".join(
                        "".join(cells[index * 2 : index * 2 + 2])
                        for index in range((len(cells) + 1) // 2)
                    )
                else:
                    source = "".join(cells)
                rendered = self._render_nested(source).strip()
                if rendered:
                    rendered_rows.append(rendered)
            return "\n".join(rendered_rows)

        if environment in ("cases", "cases*"):
            rows: list[list[str]] = []
            for cases_row in self._split_environment_rows(body):
                cells = [
                    self._render_nested(cell, False).strip()
                    for cell in cases_row.split("&")
                ]
                if any(cells):
                    rows.append(cells)
            case_rows: list[str] = []
            for index, row_cells in enumerate(rows):
                value = re.sub(r",\s*$", "", row_cells[0] if len(row_cells) > 0 else "")
                condition = row_cells[1] if len(row_cells) > 1 else ""
                if index == 0:
                    delimiter = "⎧"
                elif index == len(rows) - 1:
                    delimiter = "⎩"
                else:
                    delimiter = "⎨"
                if re.match(r"^(?:if|when|for|otherwise)\b", condition, re.IGNORECASE):
                    condition_prefix = " "
                else:
                    condition_prefix = " if "
                if condition:
                    case_rows.append(
                        f"{delimiter} {value}{condition_prefix}{condition}"
                    )
                else:
                    case_rows.append(f"{delimiter} {value}")
            return "\n".join(case_rows)

        if environment in (
            "array",
            "matrix",
            "smallmatrix",
            "pmatrix",
            "bmatrix",
            "Bmatrix",
            "vmatrix",
            "Vmatrix",
        ):
            matrix_body = (
                re.sub(r"^\s*\{[^}]*\}", "", body) if environment == "array" else body
            )
            return self._render_matrix(environment, matrix_body)

        self._supported = False
        return body

    def _render_matrix(self, environment: str, body: str) -> str:
        """渲染矩阵环境（对应 TS renderMatrix）。"""
        matrix: list[list[str]] = [
            [self._render_nested(cell, False).strip() for cell in row.split("&")]
            for row in self._split_environment_rows(body)
        ]
        matrix = [row for row in matrix if any(row)]
        column_count = max(0, *(len(row) for row in matrix))
        column_widths: list[int] = []
        for column in range(column_count):
            widths = [visible_width(row[column]) for row in matrix if column < len(row)]
            column_widths.append(max(0, *widths))

        rows: list[str] = []
        for matrix_row in matrix:
            cells: list[str] = []
            for column in range(column_count):
                cell = matrix_row[column] if column < len(matrix_row) else ""
                padding = max(0, column_widths[column] - visible_width(cell))
                cells.append(f"{cell}{PROTECTED_SPACE * padding}")
            rows.append(" │ ".join(cells))

        if environment in ("array", "matrix", "smallmatrix"):
            lines = rows
        else:
            delimiters: dict[str, tuple[str, str, str, str, str, str]] = {
                "pmatrix": ("⎛", "⎞", "⎜", "⎟", "⎝", "⎠"),
                "bmatrix": ("⎡", "⎤", "⎢", "⎥", "⎣", "⎦"),
                "Bmatrix": ("⎧", "⎫", "⎨", "⎬", "⎩", "⎭"),
                "vmatrix": ("│", "│", "│", "│", "│", "│"),
                "Vmatrix": ("║", "║", "║", "║", "║", "║"),
            }
            delimiter = delimiters.get(environment)
            if delimiter is None:
                self._supported = False
                return "\n".join(rows)
            lines = []
            for index, row in enumerate(rows):
                if index == 0:
                    left, right = delimiter[0], delimiter[1]
                elif index == len(rows) - 1:
                    left, right = delimiter[4], delimiter[5]
                else:
                    left, right = delimiter[2], delimiter[3]
                lines.append(f"{left} {row} {right}")

        if len(lines) <= 1:
            return lines[0] if lines else ""
        self._layout_nodes.append(MatrixNode("matrix", lines, 0))
        return f"{LAYOUT_MARKER_START}{len(self._layout_nodes) - 1}{LAYOUT_MARKER_END}"

    def _render_nested(self, source: str, stack_fractions: bool = True) -> str:
        """渲染嵌套源码片段（对应 TS renderNested）。"""
        rendered = LatexParser(
            source, self._layout_nodes, self._display and stack_fractions
        ).render()
        if rendered is None:
            self._supported = False
            return source
        return rendered


def render_latex(source: str, display: bool = False) -> str | None:
    """将 LaTeX 数学表达式渲染为终端友好的 Unicode 文本（对标 latex.ts 的 renderLatex）。

    参数：
        source: LaTeX 数学源码（不含 $ 定界符，由调用方剥除）。
        display: 是否为块级公式。True 时分数纵向堆叠、运算符上下限上下排列。

    返回：
        渲染后的 Unicode 文本；语法不支持或残缺时返回 None。
    """
    layout_nodes: list[LayoutNode] = []
    rendered = LatexParser(source, layout_nodes, display).render()
    if rendered is None:
        return None
    if not layout_nodes:
        return rendered.replace(PROTECTED_SPACE, " ")
    lines = _render_layout(rendered, layout_nodes).lines
    non_empty_lines = [line for line in lines if line.strip()]
    indentation = min(
        (len(line) - len(line.lstrip()) for line in non_empty_lines), default=0
    )
    return (
        "\n".join(line[indentation:].rstrip() for line in lines)
        .rstrip()
        .replace(PROTECTED_SPACE, " ")
    )
