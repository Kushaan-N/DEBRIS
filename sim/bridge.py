"""
sim/bridge.py — MuJoCo adapter for standalone training (no Newton/gizmo/HUD).

DisasterBridgeV2 calls:
    self._sim.set_body_qpos(body_name, qpos_array)   # 13-elem [x,y,z,qw,qx,qy,qz,vx,vy,vz,wx,wy,wz]
    self._sim.set_body_qvel(body_name, qvel_array)   # 6-elem  [vx,vy,vz,wx,wy,wz]
    self._sim.set_ctrl(action_array)
    self._sim.step(n_substeps=N)
    self._sim.get_sensor_data() -> dict[str, np.ndarray]
    self._sim.set_eq_active(weld_idx, active: bool)
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np

SCENES_DIR = (
    Path(os.environ.get("DEBRIS_ROOT", str(Path(__file__).resolve().parents[1])))
    / "scenes"
)


class SimBridge:
    """
    Wraps MuJoCo CPU physics for standalone training.

    Loads scene.xml directly via mujoco.MjModel, runs mj_step, and exposes
    the interface DisasterBridgeV2 expects — no Newton/gizmo/HUD required.

    Note on MuJoCo 3.x conventions:
      - Free-joint qpos: [x,y,z, qw,qx,qy,qz]
      - Free-joint qvel: [wx,wy,wz, vx,vy,vz]  (angular first, then linear)
      - eq_active lives on mjData (not mjModel)
    """

    def __init__(self, scene_id: str, settle_steps: int = 50, viewer: bool = False):
        scene_path = SCENES_DIR / scene_id / "scene.xml"
        if not scene_path.exists():
            raise FileNotFoundError(f"Scene not found: {scene_path}")

        self._model = mujoco.MjModel.from_xml_path(str(scene_path))
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)

        self._viewer = None
        if viewer or os.environ.get("WORLDSIM_VIEWER") == "1":
            try:
                from mujoco import viewer as _mj_viewer
                self._viewer = _mj_viewer.launch_passive(self._model, self._data)
                print("[SimBridge] Viewer launched")
            except Exception as e:
                print(f"[SimBridge] Viewer not available: {e}")

        # Sensor name → (adr, dim) — built once for O(1) lookup in get_sensor_data
        self._sensor_map: dict[str, tuple[int, int]] = {}
        for i in range(self._model.nsensor):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if name:
                self._sensor_map[name] = (
                    int(self._model.sensor_adr[i]),
                    int(self._model.sensor_dim[i]),
                )

        # Body name → (qpos_adr, dof_adr) for free joints
        self._free_joint_map: dict[str, tuple[int, int]] = {}
        m = self._model
        for jnt_id in range(m.njnt):
            if m.jnt_type[jnt_id] == mujoco.mjtJoint.mjJNT_FREE:
                body_id = m.jnt_bodyid[jnt_id]
                body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body_id)
                if body_name:
                    self._free_joint_map[body_name] = (
                        int(m.jnt_qposadr[jnt_id]),
                        int(m.jnt_dofadr[jnt_id]),
                    )

        # Let the scene settle briefly (weld constraint forces, geom overlap)
        for _ in range(settle_steps):
            mujoco.mj_step(self._model, self._data)

        self.step_count = 0

    # ------------------------------------------------------------------
    # Interface used by DisasterBridgeV2

    def get_sensor_data(self) -> dict[str, np.ndarray]:
        """Read all named sensors. Returns dict[name → float32 array (or scalar)]."""
        mujoco.mj_forward(self._model, self._data)
        out: dict[str, np.ndarray] = {}
        for name, (adr, dim) in self._sensor_map.items():
            arr = self._data.sensordata[adr : adr + dim].astype(np.float32)
            out[name] = arr.copy() if dim > 1 else float(arr[0])
        return out

    def set_body_qpos(self, body_name: str, qpos: np.ndarray) -> None:
        """Set the free-joint state of a named body.

        qpos layout (13 elements):
            [0:3]   xyz position
            [3:7]   quaternion wxyz
            [7:10]  linear velocity vx,vy,vz
            [10:13] angular velocity wx,wy,wz
        """
        entry = self._free_joint_map.get(body_name)
        if entry is None:
            return
        qadr, dadr = entry
        self._data.qpos[qadr : qadr + 7] = qpos[:7]       # xyz + quat
        self._data.qvel[dadr : dadr + 3]  = qpos[10:13]   # angular vel (MuJoCo: angular first)
        self._data.qvel[dadr + 3 : dadr + 6] = qpos[7:10] # linear vel
        mujoco.mj_forward(self._model, self._data)

    def set_body_qvel(self, body_name: str, qvel: np.ndarray) -> None:
        """Set the velocity of a named body.

        qvel layout (6 elements): [vx,vy,vz, wx,wy,wz]
        """
        entry = self._free_joint_map.get(body_name)
        if entry is None:
            return
        _, dadr = entry
        self._data.qvel[dadr + 3 : dadr + 6] = qvel[:3]  # linear vel
        self._data.qvel[dadr : dadr + 3]      = qvel[3:6] # angular vel
        mujoco.mj_forward(self._model, self._data)

    def set_ctrl(self, action: np.ndarray) -> None:
        """Set actuator controls (clipped to nu)."""
        nu = self._model.nu
        self._data.ctrl[:nu] = action[:nu]

    def step(self, n_substeps: int = 1) -> None:
        """Advance the simulation by n_substeps."""
        for _ in range(n_substeps):
            mujoco.mj_step(self._model, self._data)
        self.step_count += n_substeps
        if self._viewer is not None:
            self._viewer.sync()

    def set_eq_active(self, weld_idx: int, active: bool) -> None:
        """Toggle a weld equality constraint.

        In MuJoCo 3.x, eq_active is on mjData (not mjModel).
        Setting it takes effect on the next mj_step call.
        """
        if weld_idx < self._model.neq:
            self._data.eq_active[weld_idx] = 1 if active else 0
