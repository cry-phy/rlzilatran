from __future__ import annotations
from collections import deque
from typing import cast

import torch

from agents import BaseAgent
from game import Game
from game.utils import Constants


class Node:
    """Abstract Base Class for Tree node."""

    def __init__(self, child_index: torch.Tensor) -> None:
        """Generate empty node."""
        self._mean_value: float = 0
        self._game: Game = Game.empty_game()
        self._move_tensor: torch.Tensor = torch.empty(0)
        self._score: float = 0
        self._visit_count: int = 0
        self.parent: Node = self
        self._parent_child_index: torch.Tensor = child_index
        self.children: list[Node | None] = []

    @property
    def visit_count(self) -> int:
        """The visit_count property."""
        return self._visit_count

    @property
    def mean_value(self) -> float:
        """The visit_count property."""
        return self._mean_value

    @property
    def parent_child_index(self):
        """The _parent_child_index property."""
        return self._parent_child_index

    def is_outer(self) -> bool:
        """Check if this is the end of rollout."""
        raise NotImplementedError(
            "Node is an abstract base class not designed for direct usage."
        )

    def set(self, move_tensor: torch.Tensor, value: float) -> None:
        """Set the move tensors."""
        raise NotImplementedError(
            "Node is an abstract base class not designed for direct usage."
        )

    def rollout(self) -> Node:
        """Propagate rollout."""
        raise NotImplementedError(
            "Node is an abstract base class not designed for direct usage."
        )

    def backpropagate(
        self,
        child_index: torch.Tensor,
    ) -> Node:
        """Backpropagate values of this node to the root."""
        raise NotImplementedError(
            "Node is an abstract base class not designed for direct usage."
        )


class Leaf(Node):
    """Leaf node."""

    def __init__(
        self,
        game: Game,
        child_index: torch.Tensor,
        parent: Node,
        puct_factor: float,
    ) -> None:
        """Initialize Leaf node."""
        # this might be slow but lets take a look first
        super().__init__(child_index)
        self._game = game
        self._is_terminal: bool = self._game.game_over()
        self.puct_factor: float = puct_factor
        self.parent = parent

        self._child_values: torch.Tensor = torch.empty(0)
        self._child_visit_counts: torch.Tensor = torch.empty(0)
        self._child_puct_factors: torch.Tensor = torch.empty(0)
        self._valid_mask: torch.Tensor = torch.empty(0)

    def set(self, move_tensor: torch.Tensor, value: float) -> None:
        """Set the move tensors."""
        self._valid_mask = self._game.move_mask()
        move_tensor[~self._valid_mask] = 0
        self._move_tensor = move_tensor
        self._mean_value = value
        self.children = [None] * self._move_tensor.numel()
        self._child_values = self._game.full_move_tensor(0)
        self._child_visit_counts = self._game.full_move_tensor(1e-5)
        self._child_puct_factors = self._game.full_move_tensor(self.puct_factor)

    def _choose_child(self) -> tuple[Node | None, torch.Tensor]:
        """Select child node."""
        flat_scores = (
            self._child_values / self._child_visit_counts
            + self._child_puct_factors
            * self._move_tensor
            * torch.sqrt(self.visit_count / self._child_visit_counts)
        ).flatten()
        flat_mask = self._valid_mask.flatten()
        flat_scores[~flat_mask] = float("-inf")
        child_index = torch.argmax(flat_scores)
        node = self.children[child_index]
        return node, child_index

    def _new_node(self, child_index: torch.Tensor) -> Node:
        """Generate new node."""
        new_game = self._game.copy()
        x1, y1, index = torch.unravel_index(child_index, self._move_tensor.shape)
        x2, y2, flag = new_game.move_from_index(index)
        new_game.make_move(x1, y1, x2, y2, flag)
        new_node = Leaf(new_game, child_index, self, self.puct_factor)
        return new_node

    def is_outer(self) -> bool:
        """Check if this is the end of rollout."""
        # on first visit we are outer
        return self._is_terminal or self.visit_count <= 1

    def rollout(self) -> Node:
        """Propagate rollout."""
        if not self._is_terminal:
            new_node, new_child_index = self._choose_child()
            if new_node is not None:
                node = new_node.rollout()
            else:
                node = self._new_node(new_child_index)
                self.children[new_child_index] = node
            return node
        else:
            return self

    def _adjust_child(self, child_index: torch.Tensor) -> None:
        """Adjust child specific values."""
        child_node = cast(Leaf, self.children[child_index])
        leaf_value = -child_node.mean_value
        move_indices = torch.unravel_index(child_index, self._move_tensor.shape)
        self._child_values[move_indices] += leaf_value
        self._child_puct_factors[move_indices] = child_node.puct_factor
        self._child_visit_counts[move_indices] += 1

        self._mean_value += (leaf_value - self._mean_value) / self._visit_count
        self._score += leaf_value

    def backpropagate(
        self,
        child_index: torch.Tensor,
    ) -> Node:
        """Propagate Backpropagation until root node is reached."""
        # Return negative value here to account for opponent
        self._visit_count += 1
        if not self.is_outer():
            self._adjust_child(child_index)
        node = self.parent.backpropagate(self.parent_child_index)
        return node

    def to_root(self) -> Root:
        """Convert this leaf node to a root node.

        Disconnects the other parts of the tree to free memory while allowing for reuse of the generated leaf nodes.
        """
        return Root.from_leaf(self)


class Root(Leaf):
    """Root node."""

    def __init__(self, game: Game, puct_factor: float) -> None:
        """Generate Root Node."""
        super().__init__(game, torch.empty(), self, puct_factor)

    def backpropagate(
        self,
        child_index: torch.Tensor,
    ) -> Node:
        """Terminate Backpropagation."""
        self._visit_count += 1
        self._adjust_child(child_index)
        return self

    @classmethod
    def from_leaf(cls, leaf: Leaf) -> Root:
        """Dynamic Cast from Leaf to root."""
        instance = cls(leaf._game, leaf.puct_factor)
        instance._visit_count = leaf.visit_count
        instance.parent = instance
        instance.children = leaf.children
        instance._is_terminal = leaf._is_terminal
        instance._mean_value = leaf.mean_value
        instance._game = leaf._game.copy()
        instance._score = leaf._score
        return instance

    def extract_policy(self) -> torch.Tensor:
        """Extract policy from child node visit counts."""
        node_value_list = [
            node.visit_count for node in self.children if node is not None
        ]
        node_value_tensor = torch.tensor(node_value_list, dtype=torch.float32)
        move_tensor = (
            node_value_tensor.reshape(self._move_tensor.shape) / node_value_tensor.sum()
        )
        return move_tensor

    def extract_value(self) -> float:
        """Extract value from node."""
        return self._mean_value


class SubtreeLeaf(Node):
    """Subtree future from a different Rank."""


class SubtreeRoot(Root):
    """Subtree root for a different Rank."""


class Tree:
    """Encapsulate Tree building, caching and iterating."""

    def __init__(self, game: Game, maxlen: int, puct_factor: float) -> None:
        """Initialize Tree, new Node list."""
        self.to_evaluate: deque[Node] = deque(maxlen=maxlen)
        self.root: Root = Root(game.copy(), puct_factor)
        self.to_evaluate.append(self.root)

    def recurse_to_str(self, node: Node, lines: list[str]) -> list[str]:
        """Recurse through tree and add dot file sytax to lines."""
        nid = f"n{id(node)}"
        lines.append(
            f'  {nid} [label="{node.mean_value:.2f}\\nvisits={node.visit_count}"];'
        )
        for child in node.children:
            if child is not None:
                cid = f"n{id(child)}"
                self.recurse_to_str(child, lines)
                lines.append(f"  {nid} -> {cid};")
        return lines

    def to_dot(self, path: str) -> None:
        """Generate and write a dotfile containing the Tree."""
        lines = ["digraph Tree {", "  node [shape=circle];"]
        lines = self.recurse_to_str(self.root, lines)
        lines.append("}")
        string = "\n".join(lines)
        with open(path, "w+") as file:
            file.write(string)

    def make_move(self, x1: int, y1: int, x2: int, y2: int, flag: int) -> None:
        """Make specific move in tree and move Root there."""
        ...

    def get_value(self) -> float:
        """Retrieve Value from root node."""
        return self.root.extract_value()

    def get_move_tensor(self) -> torch.Tensor:
        """Retrieve move tensor from Root node."""
        return self.root.extract_policy()


class MCTSAgent(BaseAgent):
    """Monte Carlo Tree Search Agent."""

    def __init__(self, game: Game) -> None:
        """Initialize MCTS Agent."""
        super().__init__(game)
        self.tree = Tree(self._game, Constants.BATCH_SIZE, 1.5)

    def make_move(self, x1: int, y1: int, x2: int, y2: int, flag: int) -> None:
        """Make specific move and propagate to tree."""
        self.tree.make_move(x1, y1, x2, y2, flag)
        return super().make_move(x1, y1, x2, y2, flag)

    def generate_move(
        self, move_tensor: torch.Tensor
    ) -> tuple[int, int, int, int, int]:
        """Generate a move for the current game."""
        raise NotImplementedError

    def generate_move_tensor(self) -> None:
        """Generate move tensor and value."""
        self._move_tensor = self.tree.get_move_tensor()
        self._value = self.tree.get_value()


# MCTS control flow
# while searching:
#     rollout
#     leaf batch append leaf
#     if leaf batch enough:
#         isend leaf batch
#     while value batch exist:
#         recv value batch
#     Adjust Tree weights
# Return value, move tensor
