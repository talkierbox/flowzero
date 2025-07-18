"""Tree Node for MCTS."""

from __future__ import annotations

from flowzero_src.util.tree_node import TreeNode


class MCTSNodeData:
    """Data structure for MCTS node data."""

    # TODO: Implement this
    pass


class MCTSNode(TreeNode):
    """A node in the Monte-Carlo Tree Search (MCTS) tree."""

    def __init__(self, data: MCTSNodeData, parent: MCTSNode | None = None):
        """Initialize the MCTS node with data and optional parent."""
        super().__init__(data, parent=parent)
        self.visits = 0  # Number of times this node has been visited
        self.data: MCTSNodeData = data

    def __str__(self) -> str:
        """Return a string representation of the MCTS node."""
        return f"MCTSNode(data={self.data}, visits={self.visits})"
