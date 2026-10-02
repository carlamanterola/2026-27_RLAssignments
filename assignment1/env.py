import numpy as np
import pygame

class GridworldEnv:
    """3x4 Russell & Norvig Gridworld Environment (Gymnasium Interface).

    This environment represents a classic 3-row by 4-column gridworld. States are
    indexed as single integers s in [0, 11] laid out in row-major order (top to
    bottom, left to right).

    ===========================================================================
    1. STATE LAYOUT & DUAL COORDINATE SYSTEMS
    ===========================================================================
    The grid can be referenced using three complementary representations:
    
    A) Matrix Indexing [r, c] (0-indexed, Top-Left origin):
       - Row r in {0, 1, 2} (Row 0 = Top, Row 2 = Bottom)
       - Col c in {0, 1, 2, 3} (Col 0 = Left, Col 3 = Right)
       
    B) Cartesian Coordinates (x, y) (1-indexed, Bottom-Left origin):
       - x in {1, 2, 3, 4} (Left to Right)
       - y in {1, 2, 3} (Bottom to Top)

    C) Flat State Index s in {0, ..., 11}:
       Formula: s = r * n_cols + c
       Reverse: r = s // 4,  c = s % 4

    ---------------------------------------------------------------------------
    GRID MAP VISUALIZATION:

    Matrix [r, c] / Flat Index (s)        Cartesian (x, y) Labels
    +-------+-------+-------+-------+     +-------+-------+-------+-------+
    | s=0   | s=1   | s=2   | s=3   |     | (1,3) | (2,3) | (3,3) | +1GOAL|
    | [0,0] | [0,1] | [0,2] | [0,3] |     |       |       |       |  (3)  |
    +-------+-------+-------+-------+     +-------+-------+-------+-------+
    | s=4   | s=5   | s=6   | s=7   |     | (1,2) | WALL  | (3,2) | -1 PIT|
    | [1,0] | [1,1] | [1,2] | [1,3] |     |       |  (#)  |       |  (7)  |
    +-------+-------+-------+-------+     +-------+-------+-------+-------+
    | s=8   | s=9   | s=10  | s=11  |     | (1,1) | (2,1) | (3,1) | (4,1) |
    | [2,0] | [2,1] | [2,2] | [2,3] |     | START |       |       |       |
    +-------+-------+-------+-------+     +-------+-------+-------+-------+

    ===========================================================================
    2. SPECIAL STATES
    ===========================================================================
    - Start State    : s = 8  | Matrix [2, 0] | Cartesian (1, 1) [Bottom-Left]
    - Wall Obstacle  : s = 5  | Matrix [1, 1] | Cartesian (2, 2) [Inaccessible]
    - Terminal Goal  : s = 3  | Matrix [0, 3] | Cartesian (4, 3) [Reward: +1.0]
    - Terminal Pit   : s = 7  | Matrix [1, 3] | Cartesian (4, 2) [Reward: -1.0]

    ===========================================================================
    3. ACTION SPACE & DIRECTION MAPPING
    ===========================================================================
    Action space is discrete: A = {0, 1, 2, 3}.
    
    +------------+--------------+------------------+---------------------+
    | Action ID  | Direction    | Matrix Delta     | Cartesian Delta     |
    | (Integer)  | (Human Name) | (dr, dc)         | (dx, dy)            |
    +------------+--------------+------------------+---------------------+
    |     0      | LEFT         | ( 0, -1)         | (-1,  0)            |
    |     1      | DOWN         | (+1,  0)         | ( 0, -1)            |
    |     2      | RIGHT        | ( 0, +1)         | (+1,  0)            |
    |     3      | UP           | (-1,  0)         | ( 0, +1)            |
    +------------+--------------+------------------+---------------------+

    Transition Rules:
    - Bouncing: Attempting to move off the grid boundary or into the Wall (s=5)
      results in staying in the current state (s' = s).
    - Slippery Transitions (is_slippery=True):
        * Intended action execution : 80% probability (0.8)
        * Right-angle slip (left)   : 10% probability (0.1)
        * Right-angle slip (right)  : 10% probability (0.1)
    - Deterministic Transitions (is_slippery=False):
        * Intended action execution : 100% probability (1.0)

    ===========================================================================
    4. GYMNASIUM TRANSITION MODEL (env.P STRUCTURE)
    ===========================================================================
    `env.P` is a nested Python dictionary encoding the full Markov Decision 
    Process (MDP) transition dynamics P(s', r | s, a).

    Lookup Hierarchy:
        env.P[s][a] --> list of transition tuples [(prob, s_prime, reward, done), ...]

    Tuple Fields:
        - prob (float)    : Transition probability P(s' | s, a) in [0.0, 1.0].
                            In deterministic mode, len(env.P[s][a]) == 1 (prob = 1.0).
                            In slippery mode, len(env.P[s][a]) == 3 (0.8, 0.1, 0.1).
        - s_prime (int)   : Destination state index s' in {0, ..., 11}.
        - reward (float)  : Immediate scalar reward r(s, a, s') received upon transition.
        - done (bool)     : Episode termination flag. If True, s_prime is a terminal 
                            state (+1 Goal or -1 Pit) and future V(s') = 0.

    Concrete Code Usage Example:
        >>> # Inspect transition for State 8 (Start) when taking Action 3 (UP):
        >>> env.P[8][3]
        [(1.0, 4, 0.0, False)]  # Move to State 4 (1,2) with 100% prob, 0 reward, non-terminal
    """

    def __init__(self, render_mode=None, is_slippery=False):
        self.n_rows = 3
        self.n_cols = 4
        self.n_states = self.n_rows * self.n_cols
        self.n_actions = 4

        self.start_state = 8  # Bottom-left
        self.wall_state = 5  # Center obstacle
        self.terminal_states = {3: +1.0, 7: -1.0}  # Goal (+1) and Pit (-1)
        self.step_cost = 0.0
        self.is_slippery = is_slippery

        self.P = self._build_P()
        self.s = self.start_state

        # Rendering attributes
        self.render_mode = render_mode
        self.window_size = (self.n_cols * 120, self.n_rows * 120)  # (Width, Height)
        self.window = None
        self.clock = None

    def _build_P(self):
        actions = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
        slips = {0: (1, 3), 1: (0, 2), 2: (1, 3), 3: (0, 2)}

        def get_next_state(r, c, act_idx):
            dr, dc = actions[act_idx]
            nr, nc = r + dr, c + dc
            if (
                nr < 0
                or nr >= self.n_rows
                or nc < 0
                or nc >= self.n_cols
                or (nr == 1 and nc == 1)
            ):
                return r * self.n_cols + c
            return nr * self.n_cols + nc

        P = {s: {a: [] for a in range(self.n_actions)} for s in range(self.n_states)}

        for r in range(self.n_rows):
            for c in range(self.n_cols):
                s = r * self.n_cols + c
                if s == self.wall_state:
                    continue
                if s in self.terminal_states:
                    for a in range(self.n_actions):
                        P[s][a] = [(1.0, s, 0.0, True)]
                    continue

                for a in range(self.n_actions):
                    if self.is_slippery:
                        outcomes = [(a, 0.8), (slips[a][0], 0.1), (slips[a][1], 0.1)]
                    else:
                        outcomes = [(a, 1.0)]

                    trans_map = {}
                    for act_idx, prob in outcomes:
                        s_prime = get_next_state(r, c, act_idx)
                        done = s_prime in self.terminal_states
                        reward = (
                            self.terminal_states[s_prime] if done else self.step_cost
                        )
                        key = (s_prime, reward, done)
                        trans_map[key] = trans_map.get(key, 0.0) + prob

                    P[s][a] = [
                        (prob, s_prime, r_val, d_val)
                        for (s_prime, r_val, d_val), prob in trans_map.items()
                    ]
        return P

    def reset(self, seed=None, options=None, start_state=None):
        """Resets the environment. 
        
        Allows setting a custom initial state via `start_state` parameter
        or standard Gymnasium `options` dictionary.
        use options={'start_state': <number>} like in gymnasium library
        """
        if seed is not None:
            np.random.seed(seed)

        # 1. Determine target state (Direct arg > Gymnasium options dict > Default start_state)
        target_s = self.start_state
        if start_state is not None:
            target_s = start_state
        elif options is not None and "start_state" in options:
            target_s = options["start_state"]

        # 2. Safety Check: Ensure state is valid and not inside a wall
        if target_s < 0 or target_s >= self.n_states:
            raise ValueError(f"Invalid state index {target_s}. Must be between 0 and {self.n_states - 1}.")
        if target_s == self.wall_state:
            raise ValueError(f"State {target_s} is the wall obstacle! Agent cannot spawn there.")

        self.s = target_s

        if self.render_mode == "human":
            self.render()

        return self.s, {}

    def step(self, action):
        transitions = self.P[self.s][action]
        probs = [t[0] for t in transitions]

        idx = np.random.choice(len(transitions), p=probs)
        _, next_state, reward, terminated = transitions[idx]

        self.s = next_state

        if self.render_mode == "human":
            self.render()

        return next_state, reward, terminated, False, {}

    def render(self):
        """Gymnasium-compliant render supporting 'ansi', 'human', and 'rgb_array'."""
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode in ("human", "rgb_array"):
            return self._render_pygame()

    def _render_ansi(self):
        """Renders grid as ASCII characters in terminal."""
        out = ""
        for r in range(self.n_rows):
            row_str = ""
            for c in range(self.n_cols):
                s = r * self.n_cols + c
                if s == self.s:
                    row_str += " A "  # Agent
                elif s == self.wall_state:
                    row_str += " W "  # Wall
                elif s in self.terminal_states:
                    row_str += " G " if self.terminal_states[s] > 0 else " H "
                else:
                    row_str += " . "  # Empty cell
            out += row_str + "\n"
        return out

    def _render_pygame(self):
        """Renders grid using Pygame window matching FrozenLake aesthetics."""
        if self.window is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Gridworld Environment")
                self.window = pygame.display.set_mode(self.window_size)
            else:  # rgb_array
                self.window = pygame.Surface(self.window_size)

        if self.clock is None:
            self.clock = pygame.time.Clock()

        cell_w = self.window_size[0] // self.n_cols
        cell_h = self.window_size[1] // self.n_rows

        # Clear Canvas
        self.window.fill((255, 255, 255))

        # Color Palette
        COLOR_BG = (220, 240, 255)  # Light Blue
        COLOR_WALL = (100, 100, 100)  # Dark Gray
        COLOR_GOAL = (144, 238, 144)  # Light Green
        COLOR_HOLE = (255, 160, 122)  # Light Red / Hole
        COLOR_AGENT = (220, 20, 60)  # Crimson Red

        # Draw Cells
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                s = r * self.n_cols + c
                rect = pygame.Rect(c * cell_w, r * cell_h, cell_w, cell_h)

                if s == self.wall_state:
                    pygame.draw.rect(self.window, COLOR_WALL, rect)
                elif s in self.terminal_states:
                    color = (
                        COLOR_GOAL
                        if self.terminal_states[s] > 0
                        else COLOR_HOLE
                    )
                    pygame.draw.rect(self.window, color, rect)
                else:
                    pygame.draw.rect(self.window, COLOR_BG, rect)

                # Draw Grid Borders
                pygame.draw.rect(self.window, (0, 0, 0), rect, 2)

                # Draw Agent Circle
                if s == self.s:
                    center = (c * cell_w + cell_w // 2, r * cell_h + cell_h // 2)
                    pygame.draw.circle(
                        self.window, COLOR_AGENT, center, min(cell_w, cell_h) // 3
                    )

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(4)  # 4 FPS speed control
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), (1, 0, 2)
            )

    def close(self):
        """Closes Pygame rendering window."""
        if self.window is not None:
            pygame.quit()
            self.window = None  # Reset window handle

import time
import tools

if __name__ == "__main__":
    # 1. custom env with pygame render
    env = GridworldEnv(render_mode="human")
    obs, info = env.reset()

    done = False
    while not done:
        action = np.random.choice(env.n_actions)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    time.sleep(1)
    env.close()
    tools.plot_environment(env)
    

    # 2. custom env with ANSI render
    env = GridworldEnv(render_mode="ansi")
    obs, info = env.reset()

    done = False
    while not done:
        print(env.render())
        time.sleep(0.2)
        action = np.random.choice(env.n_actions)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    tools.plot_environment(env)

    # 3. gymnasium frozenlake with human render
    import gymnasium as gym
    env = gym.make("FrozenLake-v1", render_mode="human")

    obs, info = env.reset()
    terminated = False
    truncated = False
    step_count = 0

    while not (terminated or truncated) and step_count < 10:
        action = env.action_space.sample()  # Take random Gymnasium action
        next_obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1

    env.close()
    tools.plot_environment(env)

    # 4. gymnasium frozenlake with ANSI render
    import gymnasium as gym
    env = gym.make("FrozenLake-v1", render_mode="ansi")

    obs, info = env.reset()
    terminated = False
    truncated = False
    step_count = 0

    while not (terminated or truncated) and step_count < 10:
        print(env.render())
        action = env.action_space.sample()  # Take random Gymnasium action
        next_obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        step_count += 1

    env.close()
    tools.plot_environment(env)
