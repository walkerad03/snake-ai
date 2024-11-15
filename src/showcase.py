import torch

from src.agent import DQN
from src.snake_env import SnakeEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")


env = SnakeEnv(
    grid_size=[100, 50],
    window_height=512,
    window_width=1024,
    render_mode="human",
    render_stats=True,
)

observation, info = env.reset()
n_actions = env.action_space.n
n_observations = len(observation)

model = DQN(n_observations, n_actions).to(device)
model.load_state_dict(torch.load("checkpoint.pth", weights_only=True))

while True:
    observation = torch.tensor(
        observation, dtype=torch.float32, device=device
    ).unsqueeze(0)

    with torch.no_grad():
        action = model(observation).max(1).indices.view(1, 1)

    observation, _, terminated, truncated, _ = env.step(action.item())

    if terminated or truncated:
        observation, info = env.reset()
