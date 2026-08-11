"""Result 类型与异常基类。

对应 TS ``harness/types.ts`` 的 ``Result<T, E>``、``ok``/``err``/``getOrThrow``
以及 ``harness/result.ts`` 的 ``TaggedError`` 体系。

核心约定：**可预期失败通过 ``Result`` 返回，不抛异常**；只有编程错误与
不可恢复错误才抛出继承自 ``AgentError`` 的异常。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Generic, TypeVar

import orjson

TValue = TypeVar("TValue")
TError = TypeVar("TError", bound=BaseException)


class _Missing:
    """哨兵：标记「未存储值」，与「存储的 None」区分开。"""

    __slots__ = ()


_MISSING = _Missing()


@dataclass(frozen=True)
class Result(Generic[TValue, TError]):
    """可失败操作的结果：``{ok: true, value} | {ok: false, error}``。

    - ``ok(value)`` 构造成功结果
    - ``err(error)`` 构造失败结果
    - ``is_ok()`` / ``is_err()`` 判断
    - ``value`` / ``error`` 只读属性（仅配合 ``is_ok()`` / ``is_err()`` 访问）
    - ``get_or_throw()`` 解包成功值，失败则抛出错误（用于测试与显式适配边界）

    ``ok(None)`` 合法：成功值可以是 ``None``，内部用哨兵 ``_MISSING``
    区分「未存储值」（err 结果）与「存储的 None」。
    """

    _ok: bool
    _value: TValue | _Missing
    _error: TError | None

    @property
    def value(self) -> TValue:
        """成功值（对应 TS ``result.value``，仅在 ``is_ok()`` 为真时访问）。"""
        if not self._ok:
            raise RuntimeError("Result 是错误结果；仅在 is_ok() 为真时访问 .value")
        assert self._value is not _MISSING
        return self._value  # type: ignore[return-value]

    @property
    def error(self) -> TError:
        """错误值（对应 TS ``result.error``，仅在 ``is_err()`` 为真时访问）。"""
        if self._ok:
            raise RuntimeError("Result 是成功结果；仅在 is_err() 为真时访问 .error")
        assert self._error is not None
        return self._error

    def is_ok(self) -> bool:
        """是否为成功结果。"""
        return self._ok

    def is_err(self) -> bool:
        """是否为失败结果。"""
        return not self._ok

    def get_or_throw(self) -> TValue:
        """返回成功值；若为失败结果则抛出错误。"""
        if not self._ok:
            assert self._error is not None
            raise self._error
        assert self._value is not _MISSING
        return self._value  # type: ignore[return-value]

    def unwrap_or(self, default: TValue) -> TValue:
        """返回成功值；失败时返回 ``default``。"""
        if self._ok:
            assert self._value is not _MISSING
            return self._value  # type: ignore[return-value]
        return default


def ok(value: TValue) -> Result[TValue, TError]:
    """构造成功结果。"""
    return Result(_ok=True, _value=value, _error=None)


def err(error: TError) -> Result[TValue, TError]:
    """构造失败结果。"""
    return Result(_ok=False, _value=_MISSING, _error=error)


def get_or_undefined(result: Result[TValue, TError]) -> TValue | None:
    """返回成功值或 ``None``。"""
    return result.value if result._ok else None


def get_or_throw(result: Result[TValue, TError]) -> TValue:
    """返回成功值；失败结果抛出错误（对应 TS ``Result.getOrThrow``）。"""
    return result.get_or_throw()


def to_error(error: object) -> Exception:
    """把未知抛出的值规范化为 ``Exception``，作为类型化错误的 cause。"""
    if isinstance(error, BaseException):
        return error if isinstance(error, Exception) else RuntimeError(str(error))
    if isinstance(error, str):
        return RuntimeError(error)
    try:
        return RuntimeError(orjson.dumps(error).decode("utf-8"))
    except Exception:
        return RuntimeError(str(error))


class AgentError(Exception):
    """所有可预期失败异常的基类（对应 TS 的 ``TaggedError`` 体系）。

    子类通过类属性 ``code`` 声明稳定错误码，供 ``match`` 分发。
    """

    code: ClassVar[str] = "agent_error"

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.message = message
        if cause is not None:
            self.__cause__ = cause

    def to_json(self) -> dict[str, object]:
        """序列化为 ``{"_tag": code, "message": ...}``，对应 TS 的 ``toJSON()``。"""
        return {"_tag": self.code, "message": self.message}
