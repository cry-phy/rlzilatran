from __future__ import annotations
from collections import deque
from time import monotonic
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
        self.selected: bool = (
            False  # Indicates if the node is part of the selected path
        )

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

    @property
    def state(self) -> torch.Tensor:
        """The state tensor of the game."""
        return self._game.state_tensor

    def is_outer(self) -> bool:
        """Check if this is the end of rollout."""
        raise NotImplementedError(
            "Node is an abstract base class not designed for direct usage."
        )

    def set(self, move_tensor: torch.Tensor, value: float | torch.Tensor) -> None:
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

        self._valid_mask = self._game.move_mask()
        move_tensor = self._game.full_move_tensor(1e-5)
        move_tensor[self._valid_mask] = 0
        self._move_tensor = move_tensor
        self._child_values: torch.Tensor = self._game.full_move_tensor(0)
        self._child_visit_counts: torch.Tensor = self._game.full_move_tensor(0)
        self._child_puct_factors: torch.Tensor = self._game.full_move_tensor(
            self.puct_factor
        )
        self.children = [None] * self._move_tensor.numel()

    def set(self, move_tensor: torch.Tensor, value: float) -> None:
        """Set the move tensors."""
        move_tensor[~self._valid_mask] = 0
        self._move_tensor = move_tensor
        self._mean_value = value
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

    def child_from_move(self, x1: int, y1: int, x2: int, y2: int, flag: int) -> Node:
        """Get Child from move identifier."""
        # TODO to be tested
        _, d1, d2 = self._move_tensor.shape
        if flag != 0:
            index = 33
        else:
            index = x2 + y2 * d1
        child_index = x1 * (d1 * d2) + y1 * d2 + index
        potential_child = self.children[child_index]
        if potential_child:
            return potential_child
        else:
            return self._new_node(cast(torch.Tensor, child_index))


class Root(Leaf):
    """Root node."""

    def __init__(self, game: Game, puct_factor: float) -> None:
        """Generate Root Node."""
        super().__init__(game, torch.empty(0), self, puct_factor)

    def backpropagate(
        self,
        child_index: torch.Tensor,
    ) -> Node:
        """Terminate Backpropagation."""
        self._visit_count += 1
        self._adjust_child(child_index)
        return self

    @classmethod
    def from_node(cls, node: Node) -> Root:
        """Promote an existing Node (e.g., Leaf) to Root."""
        instance = cls(node._game.copy(), getattr(node, "puct_factor", 1.0))
        instance._visit_count = node.visit_count
        instance.parent = instance
        instance.children = node.children
        instance._mean_value = node.mean_value
        instance._game = node._game.copy()
        instance._score = node._score
        instance._move_tensor = node._move_tensor
        instance._parent_child_index = node.parent_child_index
        instance.selected = node.selected

        # Handle optional Leaf fields if they exist
        if isinstance(node, Leaf):
            instance._is_terminal = node._is_terminal
            instance._valid_mask = node._valid_mask
            instance._child_values = node._child_values
            instance._child_visit_counts = node._child_visit_counts
            instance._child_puct_factors = node._child_puct_factors
            instance.puct_factor = node.puct_factor
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

    # Dot file generation methods
    def recurse_to_str(self, node: Node, lines: list[str]) -> list[str]:
        """Recurse through tree and add dot file syntax to lines."""
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

    def full_dot(self, path: str) -> None:
        """Generate and write a dotfile containing the Tree."""
        lines = ["digraph Tree {", "  node [shape=circle];"]
        lines = self.recurse_to_str(self.root, lines)
        lines.append("}")
        string = "\n".join(lines)
        with open(path, "w+") as file:
            file.write(string)

    def optimal_path_dot(self, path: str) -> None:
        """Export all direct children at depth 1 along the optimal path."""
        lines = ["digraph OptimalPathChildren {", "  node [shape=circle];"]

        def add_children_to_lines(node: Node, lines: list[str]) -> None:
            """Add all direct children of a node to the dot file syntax."""
            nid = f"n{id(node)}"  # Use depth as the identifier for the node
            for child in node.children:
                if child is not None:
                    cid = f"n{id(child)}"  # Increment depth for child nodes
                    lines.append(
                        f'  {cid} [label="{child.mean_value:.2f}\\nvisits={child.visit_count}"];'
                    )
                    lines.append(f"  {nid} -> {cid};")

        current_node = self.root
        lines.append(
            f'  n{id(current_node)} [label="{current_node.mean_value:.2f}\\nvisits={current_node.visit_count}"];'
        )
        while current_node.children:
            add_children_to_lines(current_node, lines)
            optimal_child = max(
                (child for child in current_node.children if child is not None),
                key=lambda child: child.mean_value,
                default=None,
            )
            if optimal_child is None:
                break
            current_node = optimal_child

        lines.append("}")
        string = "\n".join(lines)
        with open(path, "w+") as file:
            file.write(string)

    def make_move(self, x1: int, y1: int, x2: int, y2: int, flag: int) -> None:
        """Make specific move in tree and move Root there."""
        root_candidate = self.root.child_from_move(x1, y1, x2, y2, flag)
        self.root = Root.from_node(root_candidate)

    def get_value(self) -> float:
        """Retrieve Value from root node."""
        return self.root.extract_value()

    def get_move_tensor(self) -> torch.Tensor:
        """Retrieve move tensor from Root node."""
        return self.root.extract_policy()

    def perform_rollouts(self, num: int) -> list[Node]:
        """Perform number of rollouts and return leaf nodes."""
        nodes = []
        for _ in range(num):
            nodes.append(self.root.rollout())
        return nodes


class MCTSAgent(BaseAgent):
    """Monte Carlo Tree Search Agent."""

    def __init__(self, game: Game) -> None:
        """Initialize MCTS Agent."""
        super().__init__(game)
        self.tree = Tree(self._game, Constants.BATCH_SIZE, 1.5)
        self.search_time = 10000
        self.open_tasks = {}

    def make_move(self, x1: int, y1: int, x2: int, y2: int, flag: int) -> None:
        """Make specific move and propagate to tree."""
        self.tree.make_move(x1, y1, x2, y2, flag)
        return super().make_move(x1, y1, x2, y2, flag)

    def generate_move(
        self, move_tensor: torch.Tensor
    ) -> tuple[
        int | torch.Tensor,
        int | torch.Tensor,
        int | torch.Tensor,
        int | torch.Tensor,
        int | torch.Tensor,
    ]:
        """Generate a move for the current game."""
        x1, y1, index = torch.argmax(move_tensor)
        x2, y2, flag = self._game.move_from_index(index)
        return x1, y1, x2, y2, flag

    def _gather_new(self) -> list[Node]:
        """Perform rollouts and gather the resulting unevaluated nodes."""
        nodes = self.tree.perform_rollouts(Constants.BATCH_SIZE)
        return nodes

    def _collate_inputs(self, nodes: list[Node]) -> torch.Tensor:
        """Generate state tensor minibatch for evaluation send."""
        states = [node.state for node in nodes]
        return torch.cat(states)

    def _apply_outputs(
        self,
        nodes: list[Node],
        values: torch.Tensor,
        move_tensors: torch.Tensor,
    ) -> None:
        """Apply the to the nodes and backpropagate."""
        for i, node in enumerate(nodes):
            node.set(move_tensors[i], values[i])
            node.backpropagate(torch.empty(0))

    def generate_move_tensor(self) -> None:
        """Generate move tensor and value."""
        stop_time = monotonic() + self.search_time
        search_ongoing = True
        while search_ongoing:
            if monotonic() > stop_time:
                search_ongoing = not self.open_tasks
            else:
                new_nodes = self._gather_new()
                leaf_batch = self._collate_inputs(new_nodes)
                send_token = self.mpi.send_leaf_batch(leaf_batch)
                self.open_tasks[send_token] = new_nodes
            while self.mpi.probe_value_batch():
                recv_token, values, move_tensors = self.mpi.recv_value_batch()
                nodes = self.open_tasks[recv_token]
                self._apply_outputs(nodes, values, move_tensors)
                del self.open_tasks[recv_token]
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
