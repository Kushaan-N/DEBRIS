import json
import numpy as np
from pathlib import Path

from hud.environment.robot import RobotBridge, ThreadSimRunner

OBS_DIM = 71

_WELD_NAME_TO_IDX = {
    "weld_ceil_0": 0,
    "weld_ceil_1": 1,
    "weld_ceil_2": 2,
    "weld_ceil_3": 3,
    "weld_ceil_4": 4,
    "weld_wall_L0": 5,
    "weld_wall_L1": 6,
    "weld_wall_R0": 7,
    "weld_wall_R1": 8,
}


class DisasterBridgeV2(RobotBridge):

    def __init__(self, stage: str = "full_chaos", seed: int | None = None):
        meta_path = Path(__file__).parent.parent / "scenes" / "disaster-corridor-v2" / "metadata.json"
        with open(meta_path) as f:
            metadata = json.load(f)

        # -- stage config (curriculum is a direct list) --
        stages = metadata["curriculum"]
        self._stage_config = next(s for s in stages if s["name"] == stage)

        # -- episode-level params --
        self._episode_max_steps = metadata["episode"]["max_steps"]

        # -- reward coefficients from metadata --
        r = metadata["reward"]
        self.R_GOAL                  = r["goal_reached"]
        self.R_COLLISION_EDGE        = r["collision_penalty_on_contact_start"]
        self.R_FIRE                  = r["fire_zone_penalty"]
        self.R_OVERHEAD              = r["overhead_debris_penalty"]
        self.OVERHEAD_RADIUS_XY      = r["overhead_debris_radius_xy"]
        self.OVERHEAD_MIN_HEIGHT     = r["overhead_debris_min_height_above_agent"]
        self.OVERHEAD_MIN_FALL_SPEED = r["overhead_debris_min_fall_speed"]
        self.R_TIME                  = r["time_penalty_per_step"]
        self.R_PROGRESS              = r["progress_weight"]
        self.COLLISION_ENABLED_FROM  = r["collision_penalty_enabled_from_stage"]

        # collision penalties enabled per-stage flag
        self.collision_penalties_enabled = self._stage_config.get("collision_penalties_enabled", True)

        # -- scene constants --
        self.GOAL_POS       = np.array(metadata["goal"]["pos"])
        self.GOAL_THRESHOLD = metadata["goal"]["reach_threshold"]
        self.AGENT_START    = np.array(metadata["agent"]["start_pos"])

        # -- hazard data --
        hz = metadata["hazard_layers"]
        self._fire_zones    = [np.array(z) for z in hz["fire_zones"]["positions"]]
        self._fire_radius   = hz["fire_zones"]["radius"]
        self._weld_schedule = metadata["weld_crumble_schedule"]
        self.PRIMARY_DEBRIS = hz["primary_debris"]["bodies"]
        self.SCATTER_DEBRIS = hz["scatter_debris"]["bodies"]
        self._respawn_delay = hz["primary_debris"]["respawn_delay_steps"]

        self._rng = np.random.default_rng(seed)
        self._last_raw: dict = {}

        self._reset_episode_state()
        self._reset_collapse_state()

        super().__init__(sim_runner=ThreadSimRunner())

    # ------------------------------------------------------------------
    # Internal state helpers

    def _reset_episode_state(self):
        self.step_count               = 0
        self.collision_count          = 0
        self.fire_zone_hits           = 0
        self.prev_dist_to_goal        = float(np.linalg.norm(self.AGENT_START - self.GOAL_POS))
        self.success                  = False
        self.terminated               = False
        self.was_in_contact_last_step = False   # edge detection for collision penalty

    def _reset_collapse_state(self):
        self.broken_welds        = set()
        self.primary_landed_step = [-9999] * 5
        self.scatter_landed_step = [-9999] * 5
        self.ceil_tiles_broken   = [False] * 5
        self.wall_chunks_broken  = [False] * 4

    # ------------------------------------------------------------------
    # RobotBridge interface

    async def reset(self, task_id: str = "", seed: int = 0) -> str:
        self._reset_episode_state()
        self._reset_collapse_state()
        self._rng = np.random.default_rng(seed)

        start_y = self._rng.uniform(-1.5, 1.5)
        agent_start = np.array([-12.0, start_y, 0.9])
        self.prev_dist_to_goal = float(np.linalg.norm(agent_start - self.GOAL_POS))

        if self._sim is not None:
            self._sim.set_body_qpos("agent", self._pos_to_qpos(agent_start))

            for i in range(9):
                self._sim.set_eq_active(i, True)

            n_primary = self._stage_config["active_primary_debris"]
            for i in range(n_primary):
                self._sim.set_body_qpos(self.PRIMARY_DEBRIS[i], self._random_debris_spawn(primary=True))

            n_scatter = self._stage_config["active_scatter_debris"]
            for i in range(n_scatter):
                self._sim.set_body_qpos(self.SCATTER_DEBRIS[i], self._random_debris_spawn(primary=False))
                self._sim.set_body_qvel(self.SCATTER_DEBRIS[i], self._random_scatter_velocity())

        await self._send_observation()
        return (
            "Navigate the collapsing building corridor to the emergency exit "
            "without being struck by falling debris, crumbling walls, or entering fire zones."
        )

    def step(self, action) -> None:
        action = np.clip(action, [-1.5, -1.0, -1.0], [1.5, 1.0, 1.0])
        self._sim.set_ctrl(action)
        self._sim.step(n_substeps=5)
        self._last_raw = self._sim.get_sensor_data()
        self._update_collapse()
        self._respawn_debris()
        obs_dict, _ = self.get_observation()
        self._compute_reward(obs_dict)
        self.step_count += 1
        if self.success or self.step_count >= self._episode_max_steps:
            self.terminated = True

    def get_observation(self) -> tuple[dict, bool]:
        return ({"observation/state": self._build_obs()}, self.terminated)

    def _build_obs(self) -> np.ndarray:
        if self._sim is None:
            return np.zeros(OBS_DIM, dtype=np.float32)

        raw = self._sim.get_sensor_data()
        _z3 = np.zeros(3)

        agent_pos    = raw.get("agent_pos",    _z3)
        agent_vel    = raw.get("agent_vel",    _z3)
        agent_facing = raw.get("agent_facing", np.array([1.0, 0.0, 0.0]))
        goal_rel     = self.GOAL_POS - agent_pos

        # Primary debris: position + all three velocity components
        pdeb_pos = np.stack([raw.get(f"debris_p{i}_pos", _z3) for i in range(5)])  # (5,3)
        pdeb_vel = np.stack([raw.get(f"debris_p{i}_vel", _z3) for i in range(5)])  # (5,3)
        pdeb_vz  = pdeb_vel[:, 2]   # [27:32] vertical — falling indicator
        pdeb_vx  = pdeb_vel[:, 0]   # [52:57] lateral x — landing zone predictor
        pdeb_vy  = pdeb_vel[:, 1]   # [57:62] lateral y — landing zone predictor

        # Scatter debris
        sdeb_pos = np.stack([raw.get(f"debris_s{i}_pos", _z3) for i in range(5)])  # (5,3)
        sdeb_vel = np.stack([raw.get(f"debris_s{i}_vel", _z3) for i in range(5)])  # (5,3)
        sdeb_vz  = sdeb_vel[:, 2]

        # Ceiling broken flags
        ceil_flags = np.array([float(self.ceil_tiles_broken[i]) for i in range(5)])

        # Fire zone flag
        fire_flag = float(self._agent_in_fire_zone(agent_pos))

        # Collision flag
        collision_flag = float(raw.get("contact_agent", 0.0) > 0.1)

        # Time remaining normalized: 1.0 at episode start, 0.0 at timeout
        time_remaining = float(1.0 - self.step_count / max(self._episode_max_steps, 1))

        # Heading to goal angle: 0 = facing goal, π = facing away
        goal_dir = goal_rel[:2]
        goal_dir_norm = goal_dir / (np.linalg.norm(goal_dir) + 1e-6)
        facing_xy = agent_facing[:2]
        facing_xy_norm = facing_xy / (np.linalg.norm(facing_xy) + 1e-6)
        heading_angle = float(np.arccos(np.clip(np.dot(facing_xy_norm, goal_dir_norm), -1.0, 1.0)))

        flat = np.concatenate([
            agent_pos,            # [0:3]
            agent_vel,            # [3:6]
            agent_facing,         # [6:9]
            goal_rel,             # [9:12]
            pdeb_pos.flatten(),   # [12:27]
            pdeb_vz,              # [27:32]
            sdeb_pos.flatten(),   # [32:47]
            sdeb_vz,              # [47:52]
            pdeb_vx,              # [52:57]
            pdeb_vy,              # [57:62]
            [time_remaining],     # [62]
            [heading_angle],      # [63]
            ceil_flags,           # [64:69]
            [fire_flag],          # [69]
            [collision_flag],     # [70]   total = 71
        ])
        return flat.astype(np.float32)

    def _agent_in_fire_zone(self, agent_pos: np.ndarray) -> bool:
        axy = agent_pos[:2]
        return any(np.linalg.norm(axy - fz[:2]) < self._fire_radius for fz in self._fire_zones)

    def result(self) -> dict:
        return {
            "score":           float(self.success),
            "success":         self.success,
            "collision_count": self.collision_count,
            "fire_zone_hits":  self.fire_zone_hits,
            "steps":           self.step_count,
        }

    # ------------------------------------------------------------------
    # Reward

    def _compute_reward(self, obs: dict) -> float:
        flat      = obs["observation/state"]
        agent_pos = flat[0:3]
        fire_flag = float(flat[69])
        collision = float(flat[70])

        dist     = float(np.linalg.norm(self.GOAL_POS - agent_pos))
        progress = self.prev_dist_to_goal - dist
        self.prev_dist_to_goal = dist

        reward = 0.0

        # 1. Progress shaping (always on)
        reward += self.R_PROGRESS * progress

        # 2. Goal reached (always on)
        if dist < self.GOAL_THRESHOLD:
            reward += self.R_GOAL
            self.success = True

        # 3. Collision penalty — EDGE DETECTION ONLY (fires on first frame of contact)
        if self.collision_penalties_enabled:
            currently_in_contact = collision > 0.5
            if currently_in_contact and not self.was_in_contact_last_step:
                reward += self.R_COLLISION_EDGE
                self.collision_count += 1
            self.was_in_contact_last_step = currently_in_contact

        # 4. Fire zone penalty (per step)
        if fire_flag > 0.5:
            reward += self.R_FIRE
            self.fire_zone_hits += 1

        # 5. Overhead debris cone penalty
        if self.collision_penalties_enabled:
            pdeb_pos = flat[12:27].reshape(5, 3)
            pdeb_vz  = flat[27:32]
            for i in range(5):
                height_above = float(pdeb_pos[i, 2] - agent_pos[2])
                xy_dist      = float(np.linalg.norm(pdeb_pos[i, :2] - agent_pos[:2]))
                if (height_above > self.OVERHEAD_MIN_HEIGHT
                        and xy_dist < self.OVERHEAD_RADIUS_XY
                        and pdeb_vz[i] < -self.OVERHEAD_MIN_FALL_SPEED):
                    reward += self.R_OVERHEAD
                    break   # at most one overhead penalty per step

        # 6. Time penalty
        reward += self.R_TIME
        return float(reward)

    # ------------------------------------------------------------------
    # Collapse wave

    def _update_collapse(self):
        if self._sim is None or not self._stage_config["crumble_enabled"]:
            return

        agent_x = float(self._last_raw.get("agent_pos", np.zeros(3))[0])

        for entry in self._weld_schedule:
            weld_name = entry["weld"]   # key is "weld" not "name"
            if weld_name in self.broken_welds:
                continue
            if self.step_count >= entry["step"] or agent_x >= entry["agent_x_trigger"]:
                idx = _WELD_NAME_TO_IDX[weld_name]
                self._sim.set_eq_active(idx, False)
                self.broken_welds.add(weld_name)
                if weld_name.startswith("weld_ceil_"):
                    self.ceil_tiles_broken[int(weld_name[-1])] = True
                elif weld_name.startswith("weld_wall_"):
                    suffix = weld_name[len("weld_wall_"):]
                    chunk_idx = {"L0": 0, "L1": 1, "R0": 2, "R1": 3}.get(suffix, -1)
                    if chunk_idx >= 0:
                        self.wall_chunks_broken[chunk_idx] = True

    # ------------------------------------------------------------------
    # Debris respawn

    def _respawn_debris(self):
        if self._sim is None:
            return

        n_primary = self._stage_config["active_primary_debris"]
        for i in range(n_primary):
            pos = self._last_raw.get(f"debris_p{i}_pos", np.zeros(3))
            if pos[2] < 0.5 and self.primary_landed_step[i] == -9999:
                self.primary_landed_step[i] = self.step_count
            if (self.primary_landed_step[i] != -9999
                    and self.step_count - self.primary_landed_step[i] >= self._respawn_delay):
                self._sim.set_body_qpos(self.PRIMARY_DEBRIS[i], self._random_debris_spawn(primary=True))
                self.primary_landed_step[i] = -9999

        n_scatter     = self._stage_config["active_scatter_debris"]
        scatter_delay = max(1, self._respawn_delay // 2)
        for i in range(n_scatter):
            pos = self._last_raw.get(f"debris_s{i}_pos", np.zeros(3))
            if pos[2] < 0.5 and self.scatter_landed_step[i] == -9999:
                self.scatter_landed_step[i] = self.step_count
            if (self.scatter_landed_step[i] != -9999
                    and self.step_count - self.scatter_landed_step[i] >= scatter_delay):
                self._sim.set_body_qpos(self.SCATTER_DEBRIS[i], self._random_debris_spawn(primary=False))
                self._sim.set_body_qvel(self.SCATTER_DEBRIS[i], self._random_scatter_velocity())
                self.scatter_landed_step[i] = -9999

    # ------------------------------------------------------------------
    # Private helpers

    def _pos_to_qpos(self, pos: np.ndarray) -> np.ndarray:
        """13-element body state: [x,y,z, qw,qx,qy,qz, vx,vy,vz, wx,wy,wz]"""
        return np.array([
            pos[0], pos[1], pos[2],
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ], dtype=np.float64)

    def _random_debris_spawn(self, primary: bool = True) -> np.ndarray:
        x = float(self._rng.uniform(-11.0, 12.0))
        y = float(self._rng.uniform(-2.8, 2.8))
        z = float(self._rng.uniform(6.0, 9.0) if primary else self._rng.uniform(4.5, 7.0))
        u = self._rng.uniform(size=3)
        qx = np.sqrt(1 - u[0]) * np.sin(2 * np.pi * u[1])
        qy = np.sqrt(1 - u[0]) * np.cos(2 * np.pi * u[1])
        qz = np.sqrt(u[0])     * np.sin(2 * np.pi * u[2])
        qw = np.sqrt(u[0])     * np.cos(2 * np.pi * u[2])
        return np.array([x, y, z, qw, qx, qy, qz, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def _random_scatter_velocity(self) -> np.ndarray:
        """6-element velocity state: [vx, vy, vz, wx, wy, wz]"""
        vx = float(self._rng.uniform(-3.0, 3.0))
        vy = float(self._rng.uniform(-4.0, 4.0))
        vz = float(self._rng.uniform(-1.0, 0.5))
        wx = float(self._rng.uniform(-5.0, 5.0))
        wy = float(self._rng.uniform(-5.0, 5.0))
        wz = float(self._rng.uniform(-5.0, 5.0))
        return np.array([vx, vy, vz, wx, wy, wz], dtype=np.float64)
