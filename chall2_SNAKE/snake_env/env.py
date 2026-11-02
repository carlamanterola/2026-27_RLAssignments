# Import necessary libraries
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2

# Set the goal length for the snake (used to normalize the length feature)
SNAKE_LEN_GOAL = 60

# Steps allowed without eating an apple before the episode is truncated
STARVATION_LIMIT = 200

# Weight of the distance-shaping reward (apple reward dominates when small)
SHAPING_REWARD = 0.5

# Directions indexed by action: left, right, down, up
DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))

# Observation size: apple delta (2) + danger (4) + rays (4) + direction one-hot (4) + length (1)
OBS_DIM = 15

# Define the size of the game table for gameplay
tableSize = 500
halfTable = int(tableSize / 2)


# Generate a random apple position aligned with the snake's movement grid and not on the body
def random_apple_position(action_size=1, occupied=(), rng=None):
    if rng is None:
        rng = np.random.default_rng()
    step = 10 * action_size
    max_k = (halfTable - step) // step
    while True:
        pos = [
            halfTable + step * int(rng.integers(-max_k, max_k + 1)),
            halfTable + step * int(rng.integers(-max_k, max_k + 1)),
        ]
        if pos not in occupied:
            return pos


# Function to check if the snake head collides with game boundaries
def collision_with_boundaries(snake_head):
    return (
        snake_head[0] >= tableSize or snake_head[0] < 0
        or snake_head[1] >= tableSize or snake_head[1] < 0
    )


# Function to check if the snake collides with itself
def collision_with_self(snake_position):
    return snake_position[0] in snake_position[1:]


# Define custom snake game environment class inheriting from gym's Env class
class SnakeEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10
    }

    def __init__(self, render_mode=None, action_size=1, render_fps=10):
        super(SnakeEnv, self).__init__()
        # Define the action space: 4 discrete actions (left, right, down, up)
        self.action_space = spaces.Discrete(4)

        # Normalized egocentric observation
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(OBS_DIM,),
            dtype=np.float32
        )

        # Initialize game display and reward variables
        self.img = np.zeros((tableSize, tableSize, 3), dtype='uint8')
        self.prev_reward = 0
        self.total_reward = 0
        self.score = 0
        self.max_score = 0
        self.steps_since_apple = 0

        # rendering
        self.render_mode = render_mode
        self.render_fps = render_fps

        # number of cells the snake moves in each step
        self.action_size = action_size

    # Build the normalized observation of the current state
    def _get_obs(self):
        step = 10 * self.action_size
        hx, hy = self.snake_head
        # Tail cell frees up on the next move, so it is not an obstacle
        body = {tuple(p) for p in self.snake_position[:-1]}
        max_n = tableSize // step

        danger, rays = [], []
        for dx, dy in DIRS:
            x, y, n = hx, hy, 0
            while True:
                x += dx * step
                y += dy * step
                n += 1
                if not (0 <= x < tableSize and 0 <= y < tableSize) or (x, y) in body:
                    break
            danger.append(float(n == 1))
            rays.append(min(n / max_n, 1.0))

        ax, ay = self.apple_position
        obs = np.array(
            [
                (ax - hx) / tableSize,
                (ay - hy) / tableSize,
                *danger,
                *rays,
                *np.eye(4)[self.direction],
                min(len(self.snake_position) / SNAKE_LEN_GOAL, 1.0),
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -1.0, 1.0)

    # Step function that updates the environment after an action is taken
    def step(self, action):
        # Calculate previous distance to the apple
        prev_distance = np.linalg.norm(np.array(self.snake_head) - np.array(self.apple_position))

        # Perform the action
        self._take_action(action)

        # Check if snake eats the apple
        ate = self.snake_head == self.apple_position
        self.snake_position.insert(0, list(self.snake_head))
        if ate:
            self.score += 1
            self.apple_position = random_apple_position(
                self.action_size, self.snake_position, self.np_random
            )
            self.steps_since_apple = 0
        else:
            self.snake_position.pop()
            self.steps_since_apple += 1

        # Check for collisions (with boundaries or self) and starvation
        terminated = collision_with_boundaries(self.snake_head) or collision_with_self(self.snake_position)
        truncated = self.steps_since_apple >= STARVATION_LIMIT

        # Reward
        reward = -0.1  # small penalty for each step
        if ate:
            reward += 10
        if terminated:
            reward -= 10
        elif not ate:
            current_distance = np.linalg.norm(np.array(self.snake_head) - np.array(self.apple_position))
            reward += SHAPING_REWARD if current_distance < prev_distance else -SHAPING_REWARD

        info = {"apples": self.score}

        return self._get_obs(), reward, terminated, truncated, info

    # Reset the environment to the initial state
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Reset snake position, spaced according to the movement step size
        step = 10 * self.action_size
        self.snake_position = [
            [halfTable, halfTable],
            [halfTable - step, halfTable],
            [halfTable - 2 * step, halfTable]
        ]

        # Generate random apple position aligned with the movement grid
        self.apple_position = random_apple_position(
            self.action_size, self.snake_position, self.np_random
        )

        # Update max score if needed
        if self.score > self.max_score:
            self.max_score = self.score
            print(f"New maximum score registered: {self.max_score}")

        # Reset score, starvation counter and snake head position
        self.score = 0
        self.steps_since_apple = 0
        self.snake_head = [halfTable, halfTable]

        # Initialize direction of the snake (right)
        self.direction = 1

        # Reset reward values
        self.prev_reward = 0
        self.total_reward = 0

        return self._get_obs(), {}

    # Render the game visually using OpenCV
    def render(self, mode=None):
        if mode is None:
            mode = self.render_mode

        self._update_ui()  # Update the game UI

        if mode == 'human':
            cv2.imshow('Snake Game', self.img)
            cv2.waitKey(int(1000 / self.render_fps))  # in milliseconds
        elif mode == 'rgb_array':
            return self.img.copy()

    # Close the OpenCV windows
    def close(self):
        cv2.destroyAllWindows()

    # Update the UI with the current positions of the snake and the apple
    def _update_ui(self):
        # Clear the game board
        self.img = np.zeros((tableSize, tableSize, 3), dtype='uint8')

        # Draw apple on the board
        cv2.rectangle(
            self.img,
            (self.apple_position[0], self.apple_position[1]),
            (self.apple_position[0] + 10, self.apple_position[1] + 10),
            (0, 0, 255),
            -1
        )

        # Draw snake on the board
        for _i, position in enumerate(self.snake_position):
            cv2.rectangle(
                self.img,
                (position[0], position[1]),
                (position[0] + 10, position[1] + 10),
                (0, 255, 255) if _i == 0 else (0, 255, 0),
                -1
            )

    # Handle actions and update the snake's position
    def _take_action(self, action):
        action = int(action)

        # Avoid direct opposite movements
        if action == 0 and self.direction == 1:
            action = 1
        elif action == 1 and self.direction == 0:
            action = 0
        elif action == 2 and self.direction == 3:
            action = 3
        elif action == 3 and self.direction == 2:
            action = 2

        # Update direction based on the action
        self.direction = action

        # Move the snake head according to the action
        if action == 0:  # Move left
            self.snake_head[0] -= 10 * self.action_size
        elif action == 1:  # Move right
            self.snake_head[0] += 10 * self.action_size
        elif action == 2:  # Move down
            self.snake_head[1] += 10 * self.action_size
        elif action == 3:  # Move up
            self.snake_head[1] -= 10 * self.action_size