import torch
import torch.onnx

from agent import DQN
from snake_env import SnakeEnv

env = SnakeEnv(
    grid_size=[100, 50],
    window_height=512,
    window_width=1024,
    render_stats=True,
)

observation, info = env.reset()
n_actions = env.action_space.n
n_observations = len(observation)

model = DQN(n_observations, n_actions).to("cpu")
model.load_state_dict(
    torch.load("models/i_am_training_this_overnight/14200.pth")
)

dummy_in = torch.randn(1, n_observations)
torch.onnx.export(model, dummy_in, "out.onnx")
