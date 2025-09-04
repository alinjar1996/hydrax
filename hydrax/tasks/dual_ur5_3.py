import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task

class DUAL_UR5(Task):
    
    def __init__(self) -> None:
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/dual_ur5/scene.xml")
        super().__init__(
            mj_model,
            trace_sites=["tcp_0", "tcp_1"],
        )

        self.model = mjx.put_model(mj_model)
             
        # Get the site and body ids
        self.tcp_id_0 = mj_model.site("tcp_0").id
        self.tcp_id_1 = mj_model.site("tcp_1").id

        # Get target positions from the model
        self.target_pos_0 = mj_model.body("ball").pos.copy()
        self.target_pos_1 = mj_model.body("ball").pos.copy()

        self.init_joint_angle = jnp.array([1.5, -1.8, 1.75, -1.25, -1.6, 0, -1.5, -1.8, 1.75, -1.25, -1.6, 0])

        # Print debug info
        print(f"Model loaded with {mj_model.nu} actuators")
        print(f"Target 0 position: {self.target_pos_0}")
        print(f"Target 1 position: {self.target_pos_1}")

    def _get_eef_pos(self, state: mjx.Data, arm: int) -> jax.Array:
        """Get the End Effector Frame (EEF) position for specified arm."""
        tcp_id = self.tcp_id_0 if arm == 0 else self.tcp_id_1
        return state.site_xpos[tcp_id]

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """Minimal running cost - just position tracking and control penalty."""
        # Get end effector positions
        pos_0 = self._get_eef_pos(state, 0)
        pos_1 = self._get_eef_pos(state, 1)
        
        # Position costs (distance to targets)
        position_cost_0 = jnp.sum(jnp.square(pos_0 - self.target_pos_0))
        position_cost_1 = jnp.sum(jnp.square(pos_1 - self.target_pos_1))
        
        # Control penalty (smoothness)
        control_cost = jnp.sum(jnp.square(control))
        
        # Simple weighted sum
        total_cost = 100.0 * (position_cost_0 + position_cost_1) + 0.1 * control_cost
        
        return total_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        return self.running_cost(state, jnp.zeros(self.model.nu))