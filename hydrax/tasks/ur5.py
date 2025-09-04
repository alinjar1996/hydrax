import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task

from functools import partial


class UR5(Task):
    """Standup task for the Unitree G1 humanoid."""

    def __init__(self) -> None:
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/ur5/scene.xml")
        super().__init__(
            mj_model,
            trace_sites=["tcp"],
        )

        self.model = mjx.put_model(mj_model)
             
        # Get the hande and tcp ids
        self.hande_id = mj_model.body(name="hande").id
        self.tcp_id = mj_model.site(name="tcp").id

        self.init_joint_angle = jnp.array([1.5, -1.8, 1.75, -1.25, -1.6, 0])

        self.target_pos =  mj_model.body(name="target_1").pos
        self.target_quat =  mj_model.body(name="target_1").quat

        self.mjx_data = mujoco.MjData(mj_model)

        self.mjx_data = mjx.put_data(mj_model, self.mjx_data)


        self.geom_ids = jnp.array([
                    mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, f'robot_{i}')
                    for i in range(10)
                ])
        

        self.mask = jnp.any(jnp.isin(self.mjx_data.contact.geom, self.geom_ids), axis=1)
        

        #self.collision = self.mjx_data.contact.dist[self.mask]
        
        # for i in range(len(self.geom_ids)):
        #     print("geom_ids", self.geom_ids[i])
        
        # jax.debug.print("contact.geom shape: {}", self.mjx_data.contact.geom.shape)
		
        #jax.debug.print("contact.geom: {}", self.mjx_data.contact.geom)
        #jax.debug.print("self.geom_ids[:] {}",self.geom_ids)
        
        


        # for i in range(mj_model.nu):
        #         print(mj_model.joint(i).name)
        

    def _get_eef_quat(self, state: mjx.Data) -> jax.Array:
        """Get the End Effector Frame (EEF) rotation."""
        eef_quat = state.xquat[self.hande_id]

        return eef_quat

    def _get_eef_pos(self, state: mjx.Data) -> jax.Array:
        """Get the End Effector Frame (EEF) position."""
        tcp_pos = state.site_xpos[self.tcp_id]

        return tcp_pos
    
    def _quaternion_distance(self, q1, q2):
        dot_product = jnp.abs(jnp.dot(q1, q2))
        dot_product = jnp.clip(dot_product, -1.0, 1.0)

        return 2 * jnp.arccos(dot_product)
    
    @partial(jax.jit, static_argnums=(0,))
    def collision_cost(self) -> jax.Array:

        self.mjx_data = jax.jit(mjx.forward)(self.model, self.mjx_data)

        self.mjx_data = jax.jit(mjx.step)(self.model, self.mjx_data)

        collision = self.mjx_data.contact.dist[self.mask]

        # jax.debug.print("self.mask: {}", self.mask)

        #jax.debug.print("contact.geom shape: {}", self.mjx_data.contact.geom.shape)

        y = 0.005

        collision = collision.T

        # jax.debug.print("collision: {}", collision)
		
        # g = -collision[:, 1:]+collision[:, :-1]-y*collision[:, :-1]
		
        # cost_c = jnp.sum(jnp.max(g.reshape(g.shape[0], g.shape[1], 1), axis=-1, initial=0)) + jnp.sum(collision < 0)

        g = -collision[1:]+collision[:-1]-y*collision[:-1]
		
        cost_c = jnp.sum(jnp.max(g.reshape(g.shape[0], 1), axis=-1, initial=0)) + jnp.sum(collision < 0)
        
        return cost_c

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ) applied from t=1 to T-1."""
        # Get quaternion of end-effector
        quat = self._get_eef_quat(state)  # Shape (4,)
        # Orientation cost using dot product of quaternions
        
        orientation_cost = self._quaternion_distance(quat, self.target_quat)  # scalar

        # Position cost
        pos = self._get_eef_pos(state)  # Shape (3,)
        
        #target_pos = jnp.broadcast_to(self.target_pos, pos.shape)
        
        position_cost = jnp.sum(jnp.square(pos - self.target_pos))  # scalar
        
        #jax.debug.print("mask sum: {}", jnp.sum(self.mask))

        collision_cost = self.collision_cost()

        # jax.debug.print("state.qvel {}", state.qvel)
        # jax.debug.print("state.qpos {}", state.qpos)

        # Weighted sum
        return 50.0 * position_cost + 5.0 * orientation_cost + 0.0 * collision_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        print("model actuators", self.model.nu)
        return self.running_cost(state, jnp.zeros(self.model.nu))
        