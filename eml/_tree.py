from __future__ import annotations

from collections import Counter, deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ._core import ArrayLike, eml


@dataclass(frozen=True, slots=True)
class EMLTree:
    value: str
    left: EMLTree | None = None
    right: EMLTree | None = None

    def __post_init__(self) -> None:
        has_left = self.left is not None
        has_right = self.right is not None
        if has_left != has_right:
            raise ValueError("EMLTree nodes must have either zero or two children.")
        if not self.is_leaf and self.value != "EML":
            raise ValueError("Non-leaf EMLTree nodes must use the 'EML' operator label.")

    @classmethod
    def leaf(cls, value: str) -> EMLTree:
        return cls(value=value)

    @classmethod
    def node(cls, left: EMLTree, right: EMLTree) -> EMLTree:
        return cls(value="EML", left=left, right=right)

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def depth(self) -> int:
        if self.is_leaf:
            return 0
        return 1 + max(self.left.depth(), self.right.depth())

    def leaf_count(self) -> int:
        if self.is_leaf:
            return 1
        return self.left.leaf_count() + self.right.leaf_count()

    def node_count(self) -> int:
        if self.is_leaf:
            return 1
        return 1 + self.left.node_count() + self.right.node_count()

    def to_source(self) -> str:
        if self.is_leaf:
            return self.value
        return f"EML({self.left.to_source()}, {self.right.to_source()})"

    def to_nested(self) -> dict[str, Any] | str:
        if self.is_leaf:
            return self.value
        return {
            "op": "EML",
            "left": self.left.to_nested(),
            "right": self.right.to_nested(),
        }

    def pretty(self, *, indent: str = "  ", level: int = 0) -> str:
        prefix = indent * level
        if self.is_leaf:
            return f"{prefix}{self.value}"
        left_text = self.left.pretty(indent=indent, level=level + 1)
        right_text = self.right.pretty(indent=indent, level=level + 1)
        return f"{prefix}EML\n{left_text}\n{right_text}"

    def leaf_values(self) -> list[str]:
        if self.is_leaf:
            return [self.value]
        return self.left.leaf_values() + self.right.leaf_values()

    def variables(self) -> list[str]:
        return sorted({value for value in self.leaf_values() if value != "1"})

    def leaf_frequencies(self) -> dict[str, int]:
        return dict(sorted(Counter(self.leaf_values()).items()))

    def level_widths(self) -> list[int]:
        widths: list[int] = []
        queue = deque([(self, 0)])
        while queue:
            node, level = queue.popleft()
            if level == len(widths):
                widths.append(0)
            widths[level] += 1
            if not node.is_leaf:
                queue.append((node.left, level + 1))
                queue.append((node.right, level + 1))
        return widths

    def internal_node_count(self) -> int:
        return self.node_count() - self.leaf_count()

    def to_mermaid(self) -> str:
        lines = ["graph TD"]
        counter = [0]

        def visit(node: EMLTree) -> str:
            node_id = f"n{counter[0]}"
            counter[0] += 1
            label = node.value if node.is_leaf else "EML"
            safe_label = label.replace('"', "\\\"")
            lines.append(f'  {node_id}["{safe_label}"]')
            if not node.is_leaf:
                left_id = visit(node.left)
                right_id = visit(node.right)
                lines.append(f"  {node_id} --> {left_id}")
                lines.append(f"  {node_id} --> {right_id}")
            return node_id

        visit(self)
        return "\n".join(lines)

    def to_dot(self) -> str:
        lines = ["digraph EMLTree {", "  rankdir=TB;"]
        counter = [0]

        def visit(node: EMLTree) -> str:
            node_id = f"n{counter[0]}"
            counter[0] += 1
            label = node.value if node.is_leaf else "EML"
            safe_label = label.replace('"', "\\\"")
            lines.append(f'  {node_id} [label="{safe_label}"];')
            if not node.is_leaf:
                left_id = visit(node.left)
                right_id = visit(node.right)
                lines.append(f"  {node_id} -> {left_id};")
                lines.append(f"  {node_id} -> {right_id};")
            return node_id

        visit(self)
        lines.append("}")
        return "\n".join(lines)

    def evaluate(self, variables: Mapping[str, Any] | None = None) -> ArrayLike:
        bindings = variables or {}
        if self.is_leaf:
            if self.value == "1":
                return 1.0
            if self.value not in bindings:
                raise KeyError(f"Missing value for symbol '{self.value}'.")
            return bindings[self.value]
        return eml(self.left.evaluate(bindings), self.right.evaluate(bindings))


__all__ = ["EMLTree"]
