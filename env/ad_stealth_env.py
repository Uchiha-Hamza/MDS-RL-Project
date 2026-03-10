"""
AdStealthEnv — Custom Gymnasium Environment

A 10×10 grid where an ad agent must stay in the user's viewport
while evading a cycling ad-blocker scanner.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class AdStealthEnv(gym.Env):
    """
    Reinforcement Learning environment for Stealthy Advertisement Placement.

    Observation (6-dim):
        [xa, ya, xt, yt, sigma, tau]
        - (xa, ya): agent position on 10×10 grid
        - (xt, yt): center of 3×3 user viewport
        - sigma:    sector targeted by ad-blocker (0-3)
        - tau:      steps remaining until scan (0 = scanning now)

    Actions (5):
        0=Up, 1=Down, 2=Left, 3=Right, 4=Stay
    """

    metadata = {"render_modes": ["human", "ansi"]}

    # Action constants
    UP, DOWN, LEFT, RIGHT, STAY = 0, 1, 2, 3, 4
    ACTION_NAMES = ["Up", "Down", "Left", "Right", "Stay"]
    DELTAS = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0), 4: (0, 0)}

    def __init__(self, config=None, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # Load config (import here to avoid circular imports)
        if config is None:
            from config import ENV_CONFIG, REWARD_CONFIG
            config = {**ENV_CONFIG, **REWARD_CONFIG}

        # Grid parameters
        self.grid_size = config.get("grid_size", 10)
        self.viewport_size = config.get("viewport_size", 3)
        self.sector_size = config.get("sector_size", 5)
        self.warning_phase = config.get("warning_phase_length", 4)
        self.scan_phase = config.get("scan_phase_length", 1)
        self.viewport_move_interval = config.get("viewport_move_interval", 5)
        self.max_steps = config.get("max_steps", 300)
        self.viewport_pattern = config.get("viewport_pattern", "linear")

        # Rewards
        self.r_in_viewport = config.get("in_viewport", 1.0)
        self.r_near_viewport = config.get("near_viewport", 0.3)
        self.r_step_penalty = config.get("step_penalty", -0.1)
        self.r_caught = config.get("caught_penalty", -10.0)
        self.r_survival_bonus = config.get("survival_bonus", 5.0)
        self.survival_interval = config.get("survival_interval", 50)

        # Blocker cycle length (warning + scan)
        self.cycle_length = self.warning_phase + self.scan_phase  # 5

        # Viewport valid center range (so 3×3 stays in bounds)
        half = self.viewport_size // 2
        self.vp_min = half               # 1
        self.vp_max = self.grid_size - 1 - half  # 8

        # Spaces
        self.action_space = spaces.Discrete(5)
        # State: xa, ya, xt, yt, sigma, tau
        self.observation_space = spaces.Box(
            low=np.array([0, 0, self.vp_min, self.vp_min, 0, 0], dtype=np.float32),
            high=np.array([
                self.grid_size - 1, self.grid_size - 1,
                self.vp_max, self.vp_max,
                3, self.warning_phase
            ], dtype=np.float32),
        )

        # Sector definitions: dict mapping sector_id -> (x_min, x_max, y_min, y_max)
        s = self.sector_size
        self.sectors = {
            0: (0, s - 1, 0, s - 1),          # top-left
            1: (s, 2 * s - 1, 0, s - 1),      # top-right
            2: (0, s - 1, s, 2 * s - 1),       # bottom-left
            3: (s, 2 * s - 1, s, 2 * s - 1),   # bottom-right
        }

        # Internal state
        self._step_count = 0
        self._scroll_direction = 1  # for linear pattern: +1 or -1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Agent starts at random position
        self.agent_x = self.np_random.integers(0, self.grid_size)
        self.agent_y = self.np_random.integers(0, self.grid_size)

        # Viewport starts at center
        self.vp_x = self.grid_size // 2
        self.vp_y = self.grid_size // 2

        # Blocker starts at sector 0, fully warned (tau = warning_phase)
        self.blocker_sector = 0
        self.blocker_timer = self.warning_phase  # countdown: 4, 3, 2, 1, 0(scan)

        self._step_count = 0
        self._scroll_direction = 1

        return self._get_obs(), self._get_info()

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action {action}"

        self._step_count += 1
        reward = self.r_step_penalty
        terminated = False
        truncated = False

        # ── 1. Move agent ────────────────────────────────────────────
        dx, dy = self.DELTAS[action]
        new_x = np.clip(self.agent_x + dx, 0, self.grid_size - 1)
        new_y = np.clip(self.agent_y + dy, 0, self.grid_size - 1)
        self.agent_x = int(new_x)
        self.agent_y = int(new_y)

        # ── 2. Update blocker timer ──────────────────────────────────
        self.blocker_timer -= 1

        # Check if scan phase (timer reaches 0)
        if self.blocker_timer < 0:
            # Scan happens! Check if agent is in the scanned sector
            sx_min, sx_max, sy_min, sy_max = self.sectors[self.blocker_sector]
            if sx_min <= self.agent_x <= sx_max and sy_min <= self.agent_y <= sy_max:
                reward += self.r_caught
                terminated = True

            # Move to next sector, reset timer
            self.blocker_sector = (self.blocker_sector + 1) % 4
            self.blocker_timer = self.warning_phase

        # ── 3. Move viewport ─────────────────────────────────────────
        if self._step_count % self.viewport_move_interval == 0:
            self._move_viewport()

        # ── 4. Compute reward ─────────────────────────────────────────
        if not terminated:
            if self._in_viewport():
                reward += self.r_in_viewport
            elif self._near_viewport():
                reward += self.r_near_viewport

            # Survival bonus
            if self._step_count % self.survival_interval == 0:
                reward += self.r_survival_bonus

        # ── 5. Check truncation ───────────────────────────────────────
        if self._step_count >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _in_viewport(self):
        """Check if agent is inside the 3×3 viewport."""
        half = self.viewport_size // 2
        return (abs(self.agent_x - self.vp_x) <= half and
                abs(self.agent_y - self.vp_y) <= half)

    def _near_viewport(self):
        """Check if agent is within 1 cell of the viewport border."""
        half = self.viewport_size // 2 + 1
        return (abs(self.agent_x - self.vp_x) <= half and
                abs(self.agent_y - self.vp_y) <= half)

    def _move_viewport(self):
        """Move the viewport according to the selected pattern."""
        if self.viewport_pattern == "linear":
            self._move_viewport_linear()
        elif self.viewport_pattern == "random_walk":
            self._move_viewport_random_walk()
        elif self.viewport_pattern == "erratic":
            self._move_viewport_erratic()

    def _move_viewport_linear(self):
        """Smooth vertical scrolling, bouncing at edges."""
        self.vp_y += self._scroll_direction
        if self.vp_y >= self.vp_max:
            self.vp_y = self.vp_max
            self._scroll_direction = -1
        elif self.vp_y <= self.vp_min:
            self.vp_y = self.vp_min
            self._scroll_direction = 1

    def _move_viewport_random_walk(self):
        """Random direction each step."""
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        dx, dy = directions[self.np_random.integers(0, 4)]
        self.vp_x = int(np.clip(self.vp_x + dx, self.vp_min, self.vp_max))
        self.vp_y = int(np.clip(self.vp_y + dy, self.vp_min, self.vp_max))

    def _move_viewport_erratic(self):
        """Teleport to random valid position."""
        self.vp_x = int(self.np_random.integers(self.vp_min, self.vp_max + 1))
        self.vp_y = int(self.np_random.integers(self.vp_min, self.vp_max + 1))

    def _get_obs(self):
        """Return current observation as numpy array."""
        return np.array([
            self.agent_x, self.agent_y,
            self.vp_x, self.vp_y,
            self.blocker_sector, self.blocker_timer
        ], dtype=np.float32)

    def _get_info(self):
        """Return auxiliary info dict."""
        return {
            "step": self._step_count,
            "in_viewport": self._in_viewport(),
            "blocker_sector": self.blocker_sector,
            "blocker_timer": self.blocker_timer,
            "agent_pos": (self.agent_x, self.agent_y),
            "viewport_center": (self.vp_x, self.vp_y),
        }

    def get_danger_sector_bounds(self):
        """Return the bounds of the currently targeted sector."""
        return self.sectors[self.blocker_sector]

    def render(self):
        """Render environment as ASCII grid."""
        if self.render_mode != "ansi":
            return

        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Draw viewport
        half = self.viewport_size // 2
        for dx in range(-half, half + 1):
            for dy in range(-half, half + 1):
                vx, vy = self.vp_x + dx, self.vp_y + dy
                if 0 <= vx < self.grid_size and 0 <= vy < self.grid_size:
                    grid[vy][vx] = 'V'

        # Draw danger sector
        sx_min, sx_max, sy_min, sy_max = self.sectors[self.blocker_sector]
        for x in range(sx_min, sx_max + 1):
            for y in range(sy_min, sy_max + 1):
                if grid[y][x] == 'V':
                    grid[y][x] = '!'  # overlap: viewport + danger
                elif grid[y][x] == '.':
                    if self.blocker_timer == 0:
                        grid[y][x] = 'X'  # scanning now
                    else:
                        grid[y][x] = 'W'  # warning

        # Draw agent
        grid[self.agent_y][self.agent_x] = 'A'

        # Build string
        header = f"Step {self._step_count} | Sector {self.blocker_sector} | Timer {self.blocker_timer}"
        lines = [header, '-' * (self.grid_size * 2)]
        for row in grid:
            lines.append(' '.join(row))
        lines.append(f"Agent: ({self.agent_x},{self.agent_y}) | VP: ({self.vp_x},{self.vp_y})")

        output = '\n'.join(lines)
        print(output)
        return output
