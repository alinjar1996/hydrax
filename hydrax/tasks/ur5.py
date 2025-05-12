import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task


class UR5(Task):
    """Standup task for the Unitree G1 humanoid."""

    def __init__(self) -> None:
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/ur5/scene.xml")
        super().__init__(
            mj_model,
            trace_sites=["tcp"],
        )
             
        # Get the hande and tcp ids
        self.hande_id = mj_model.body(name="hande").id
        self.tcp_id = mj_model.site(name="tcp").id

        self.init_joint_angle = jnp.array([1.5, -1.8, 1.75, -1.25, -1.6, 0])

        self.target_pos =  mj_model.body(name="target_1").pos
        self.target_quat =  mj_model.body(name="target_1").quat

        for i in range(mj_model.nu):
                print(mj_model.joint(i).name)
        

    def _get_eef_quat(self, state: mjx.Data) -> jax.Array:
        """Get the End Effector Frame (EEF) rotation."""
        eef_quat = state.xquat[self.hande_id]

        return eef_quat

    def _get_eef_pos(self, state: mjx.Data) -> jax.Array:
        """Get the End Effector Frame (EEF) position."""
        tcp_pos = state.site_xpos[self.tcp_id]

        return tcp_pos
    
    def quaternion_distance(self, q1, q2):
        dot_product = jnp.abs(jnp.dot(q1, q2))
        dot_product = jnp.clip(dot_product, -1.0, 1.0)
        return 2 * jnp.arccos(dot_product)

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ) applied from t=1 to T-1."""
        # Get quaternion of end-effector
        quat = self._get_eef_quat(state)  # Shape (4,)
        # Orientation cost using dot product of quaternions
        
        orientation_cost = self.quaternion_distance(quat, self.target_quat)  # scalar

        # Position cost
        pos = self._get_eef_pos(state)  # Shape (3,)
        
        target_pos = jnp.broadcast_to(self.target_pos, pos.shape)
        
        position_cost = jnp.sum(jnp.square(pos - self.target_pos), axis = -1)  # scalar
        
        # jax.debug.print("eef_pos: {}", pos)
        # print("target_pos", self.target_pos)
        

        # Weighted sum
        return 10.0 * position_cost + 1 * orientation_cost 

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        #print(self.model.nu)
        return self.running_cost(state, jnp.zeros(self.model.nu))
        