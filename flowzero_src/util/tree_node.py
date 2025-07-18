"""Generalizable Tree Node Class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable


class TreeNode:
    """A general tree node class that can be extended for various applications."""

    def __init__(
        self,
        data: Any,  # noqa: ANN401
        children: list[TreeNode] | None = None,
        parent: TreeNode | None = None,
    ):
        """Initialize the tree node with data and optional children."""
        self.data = data
        self.children = children if children is not None else []
        self.parent = parent

    def get_num_children(self) -> int:
        """Return the number of children."""
        return len(self.children)

    def get_children(self) -> Iterable[TreeNode]:
        """Return an iterable of children."""
        return iter(self.children)

    def add_child(self, child: TreeNode) -> None:
        """Add a child to this node."""
        child.parent = self
        self.children.append(child)

    def remove_child(self, child: TreeNode) -> None:
        """Remove a child from this node."""
        if child in self.children:
            self.children.remove(child)
            child.parent = None

    def is_leaf(self) -> bool:
        """Check if this node is a leaf (has no children)."""
        return len(self.children) == 0

    def is_root(self) -> bool:
        """Check if this node is a root (has no parent)."""
        return self.parent is None

    def __str__(self) -> str:
        """Return a string representation of the node."""
        return f"TreeNode(data={self.data}, num_children={self.get_num_children()})"

    def __repr__(self) -> str:
        """Return a string representation of the node."""
        return f"TreeNode(data={self.data}, num_children={self.get_num_children()})"

    def __eq__(self, other: object) -> bool:
        """Check equality based on only data and children."""
        if not isinstance(other, TreeNode):
            return False
        return (
            self.data == other.data
            and self.get_num_children() == other.get_num_children()
            and all(
                c == oc for c, oc in zip(self.get_children(), other.get_children(), strict=True)
            )
        )

    def __iter__(self) -> Iterable[TreeNode]:
        """Return an iterator over the children."""
        return self.get_children()
