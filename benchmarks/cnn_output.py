import sys

sys.path.append("src/")
sys.path.append("../src/")

from torchview import draw_graph
from cnn.small import SmallCNN
from game.utils import Constants
import torch

device = torch.device("cpu")
model = SmallCNN(128, Constants.state_tensor_shape, Constants.move_tensor_shape, device)

graph = draw_graph(
    model, input_size=(1, *Constants.state_tensor_shape), expand_nested=True
)
graph.visual_graph.render("smallcnn_blocks", format="png")
