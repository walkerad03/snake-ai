import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 0}

    def __init__(
        self,
        grid_size: list[int] = [25, 25],
        window_width: int = 512,
        window_height: int = 512,
        render_stats: bool = False,
        render_mode: str = None,
    ) -> None:
        self.grid_size = grid_size

        # Pygame options
        self.window_width = window_width
        self.window_height = window_height

        self.render_stats = render_stats

        assert (
            render_mode is None or render_mode in self.metadata["render_modes"]
        )
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.stats_printout = None

        # Pygame colors
        self.SNAKE_HEAD_COL = (97, 147, 204)
        self.SNAKE_BODY_COL = (69, 105, 145)
        self.FOOD_COL = (204, 97, 99)
        self.BG_COL = (10, 10, 10)

        # Gym related options
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.array([-grid_size[0], -grid_size[0], 0, 0, 0, 0, 0]),
            high=np.array([grid_size[0] - 1, grid_size[1] - 1, 3, 1, 1, 1, 1]),
            dtype=np.int64,
        )

        # Game state tracking
        self.snake_position = np.random.randint(0, self.grid_size, size=2)
        self.snake_direction = np.random.randint(0, 4)
        self.snake_history = np.empty([0, 2])
        self.snake_size = 0

        self.past_observations = []

        self.food_on_board = False
        self.food_position = np.random.randint(0, self.grid_size, size=2)

        self.time_since_food = 0
        self.time_limit = (
            np.sqrt(self.grid_size[0] ** 2 + self.grid_size[1] ** 2) * 1.25
        )

        # Statistics
        self.longest_snake = 0
        self.average_snake_length = 0
        self.length_std = 0
        self.snake_length_history = []

    def _get_observation(self) -> np.array:
        # Calculate surroundings
        grid_size = 19
        surroundings = np.zeros((grid_size, grid_size))
        head_x, head_y = self.snake_position

        offset = grid_size // 2

        for i in range(grid_size):
            for j in range(grid_size):
                check_x = head_x + (i - offset)
                check_y = head_y + (j - offset)

                if (
                    check_x < 0
                    or check_x >= self.grid_size[0]
                    or check_y < 0
                    or check_y >= self.grid_size[1]
                ):
                    surroundings[i, j] = 1
                elif np.any(
                    np.all(self.snake_history == [check_x, check_y], axis=1)
                ):
                    surroundings[i, j] = 1

        flattened_surroundings = surroundings.flatten()

        food_distance = np.linalg.norm(
            self.snake_position - self.food_position
        )

        if food_distance == 0:
            food_relative_position = np.array([0, 0])
        else:

            food_relative_position = (
                self.snake_position - self.food_position
            ) / food_distance

        # Construct and return the observation
        observation = np.concatenate(
            (
                food_relative_position,  # Food relative position
                [self.snake_direction],  # Snake direction
                flattened_surroundings,  # Immediate surroundings
            )
        )

        return observation

    def _get_info(self) -> dict:
        return {}

    def reset(self):
        if self.snake_size != 0:
            self.snake_length_history.append(self.snake_size)

        if len(self.snake_length_history) != 0:
            self.average_snake_length = np.mean(self.snake_length_history)
            self.length_std = np.std(self.snake_length_history)

        self.snake_history = np.empty([0, 2])
        self.snake_size = 0
        self.snake_position = np.random.randint(0, self.grid_size, size=2)
        self.time_since_food = 0

        self.past_observations = []

        self.food_position = np.random.randint(0, self.grid_size, size=2)
        while np.array_equal(self.food_position, self.snake_position):
            self.food_position = np.random.randint(0, self.grid_size, size=2)
        self.food_on_board = True

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action: int):
        self.time_since_food += 1
        reward = -0.01
        done = False

        if self.time_since_food > self.time_limit:
            done = True

        # Trims snake history to the correct size.
        self.snake_history = np.vstack(
            (self.snake_history, self.snake_position)
        )
        if self.snake_history.shape[0] > self.snake_size:
            self.snake_history = np.delete(self.snake_history, 0, 0)

        original_dist = np.linalg.norm(
            self.snake_position - self.food_position
        )

        previous_direction = self.snake_direction

        # Snake movement
        self.snake_direction = action

        if previous_direction == self.snake_direction:
            reward += 0.5

        if self.snake_direction == 0:
            # Move snake down
            self.snake_position[1] -= 1
        elif self.snake_direction == 1:
            # move snake up
            self.snake_position[1] += 1
        elif self.snake_direction == 2:
            # move snake left
            self.snake_position[0] -= 1
        elif self.snake_direction == 3:
            # move snake right
            self.snake_position[0] += 1

        new_dist = np.linalg.norm(self.snake_position - self.food_position)

        if new_dist < original_dist:
            reward += 1

        # Check if snake intersects with its tail
        if np.any(np.all(self.snake_history == self.snake_position, axis=1)):
            reward += -10
            done = True

        # check if snake hits a wall
        if np.any(
            (self.snake_position >= self.grid_size) | (self.snake_position < 0)
        ):
            reward += -10
            done = True

        # Check if snake intersects with food and reward if true
        if np.array_equal(self.food_position, self.snake_position):
            self.food_on_board = False
            self.snake_size += 1

            if self.snake_size > self.longest_snake:
                self.longest_snake = self.snake_size

            reward += 10 + self.snake_size
            self.time_since_food = 0

            if self.snake_size != self.grid_size[0] * self.grid_size[1]:
                self.food_position = np.random.randint(
                    0, self.grid_size, size=2
                )
                while np.array_equal(
                    self.food_position, self.snake_position
                ) or np.any(
                    np.all(self.food_position == self.snake_history, axis=1)
                ):
                    self.food_position = np.random.randint(
                        0, self.grid_size, size=2
                    )
                self.food_on_board = True

        observation = self._get_observation()
        info = self._get_info()

        # TODO: Fix duplicate game state protection.
        """game_state = np.concatenate(
            [observation], [self.snake_history], [self.snake_position]
        )

        self.past_observations.append(game_state)

        seen = set()

        for array in self.past_observations:
            array_tuple = (tuple(array.flatten()), array.shape)
            if array_tuple in seen:
                done = True
                break
            seen.add(array_tuple)
        """

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, done, False, info

    def draw_stat_printout(self):
        base_string = (
            f"Games Played: {len(self.snake_length_history)}\n"
            f"Current Length: {self.snake_size}\n"
            f"Longest Snake: {self.longest_snake}\n"
            f"Average Snake Length: {np.round(self.average_snake_length, 2)}\n"
            f"Length Standard Deviation: {np.round(self.length_std, 2)}"
        )
        lines = base_string.split("\n")
        for i, line in enumerate(lines):
            out_string_t = self.stats_printout.render(
                line, 1, pygame.Color("WHITE")
            )
            self.window.blit(out_string_t, (5, 5 + 15 * i))

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_width, self.window_height)
            )

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        if self.stats_printout is None and self.render_mode == "human":
            self.stats_printout = pygame.font.SysFont("Arial", 13, bold=False)

        canvas = pygame.Surface((self.window_width, self.window_height))

        cell_width = self.window_width / self.grid_size[0]
        cell_height = self.window_height / self.grid_size[1]

        canvas.fill(self.BG_COL)

        for idx, part in enumerate(self.snake_history):
            if idx == len(self.snake_history) - 1:
                head_dir = self.snake_history[idx] - self.snake_position

                # Going left
                if (head_dir == np.array([-1.0, 0.0])).all():
                    left = part[0] * cell_width + (0.4 * cell_width)
                    top = part[1] * cell_height + (0.2 * cell_height)
                    width = cell_width * 0.8
                    height = cell_height * 0.6

                # Going right
                if (head_dir == np.array([1.0, 0.0])).all():
                    left = part[0] * cell_width - (0.4 * cell_width)
                    top = part[1] * cell_height + (0.2 * cell_height)
                    width = cell_width * 0.8
                    height = cell_height * 0.6

                # Going up
                if (head_dir == np.array([0.0, 1.0])).all():
                    left = part[0] * cell_width + (0.2 * cell_width)
                    top = part[1] * cell_height - (0.4 * cell_height)
                    width = cell_width * 0.6
                    height = cell_height * 0.8

                # Going down
                if (head_dir == np.array([0.0, -1.0])).all():
                    left = part[0] * cell_width + (0.2 * cell_width)
                    top = part[1] * cell_height + (0.4 * cell_height)
                    width = cell_width * 0.6
                    height = cell_height * 0.8

                rect = pygame.Rect(left, top, width, height)
                pygame.draw.rect(canvas, self.SNAKE_BODY_COL, rect)

            if idx > 0:
                prev_dir = (
                    self.snake_history[idx] - self.snake_history[idx - 1]
                )

                # Going left
                if (prev_dir == np.array([-1.0, 0.0])).all():
                    left = part[0] * cell_width + (0.2 * cell_width)
                    top = part[1] * cell_height + (0.2 * cell_height)
                    width = cell_width * 0.8
                    height = cell_height * 0.6

                # Going right
                if (prev_dir == np.array([1.0, 0.0])).all():
                    left = part[0] * cell_width
                    top = part[1] * cell_height + (0.2 * cell_height)
                    width = cell_width * 0.8
                    height = cell_height * 0.6

                # Going down
                if (prev_dir == np.array([0.0, 1.0])).all():
                    left = part[0] * cell_width + (0.2 * cell_width)
                    top = part[1] * cell_height
                    width = cell_width * 0.6
                    height = cell_height * 0.8

                # Going up
                if (prev_dir == np.array([0.0, -1.0])).all():
                    left = part[0] * cell_width + (0.2 * cell_width)
                    top = part[1] * cell_height + (0.2 * cell_height)
                    width = cell_width * 0.6
                    height = cell_height * 0.8

                rect = pygame.Rect(left, top, width, height)
                pygame.draw.rect(canvas, self.SNAKE_BODY_COL, rect)

            if idx < len(self.snake_history) - 1:
                prev_dir = (
                    self.snake_history[idx] - self.snake_history[idx + 1]
                )

                # Going left
                if (prev_dir == np.array([-1.0, 0.0])).all():
                    left = part[0] * cell_width + (0.2 * cell_width)
                    top = part[1] * cell_height + (0.2 * cell_height)
                    width = cell_width * 0.8
                    height = cell_height * 0.6

                # Going right
                if (prev_dir == np.array([1.0, 0.0])).all():
                    left = part[0] * cell_width
                    top = part[1] * cell_height + (0.2 * cell_height)
                    width = cell_width * 0.8
                    height = cell_height * 0.6

                # Going down
                if (prev_dir == np.array([0.0, 1.0])).all():
                    left = part[0] * cell_width + (0.2 * cell_width)
                    top = part[1] * cell_height
                    width = cell_width * 0.6
                    height = cell_height * 0.8

                # Going up
                if (prev_dir == np.array([0.0, -1.0])).all():
                    left = part[0] * cell_width + (0.2 * cell_width)
                    top = part[1] * cell_height + (0.2 * cell_height)
                    width = cell_width * 0.6
                    height = cell_height * 0.8

                rect = pygame.Rect(left, top, width, height)
                pygame.draw.rect(canvas, self.SNAKE_BODY_COL, rect)

        head_rect = pygame.Rect(
            self.snake_position[0] * cell_width + (0.2 * cell_width),
            self.snake_position[1] * cell_height + (0.2 * cell_height),
            cell_width * 0.6,
            cell_height * 0.6,
        )
        pygame.draw.rect(canvas, self.SNAKE_HEAD_COL, head_rect)

        food_rect = pygame.Rect(
            self.food_position[0] * cell_width + (0.2 * cell_width),
            self.food_position[1] * cell_height + (0.2 * cell_height),
            cell_width * 0.6,
            cell_height * 0.6,
        )
        pygame.draw.rect(canvas, self.FOOD_COL, food_rect)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())

            if self.render_stats:
                self.draw_stat_printout()

            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata["render_fps"])

        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
