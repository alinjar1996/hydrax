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
            trace_sites=["hande", "tcp"],
        )

        # Get the hande and tcp ids
        self.hande_id = mj_model.body(name="hande").id
        self.tcp_id = mj_model.site(name="tcp").id

    def _get_eef_rot(self, state: mjx.Data) -> jax.Array:
        """Get the End Effector Frame (EEF) rotation."""
        eef_rot = state.xquat[self.hande_id]

        return eef_rot

    def _get_eef_pos(self, state: mjx.Data) -> jax.Array:
        """Get the End Effector Frame (EEF) position."""
        tcp_pos = state.site_xpos[self.tcp_id]

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        orientation_cost = jnp.sum(jnp.square(self._get_eef_rot(state)))

        return 100.0 * orientation_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        return self.running_cost(state, jnp.zeros(self.model.nu))

    # def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
    #     """Randomize the friction parameters."""
    #     n_geoms = self.model.geom_friction.shape[0]
    #     multiplier = jax.random.uniform(rng, (n_geoms,), minval=0.5, maxval=2.0)
    #     new_frictions = self.model.geom_friction.at[:, 0].set(
    #         self.model.geom_friction[:, 0] * multiplier
    #     )
    #     return {"geom_friction": new_frictions}

    # def domain_randomize_data(
    #     self, data: mjx.Data, rng: jax.Array
    # ) -> Dict[str, jax.Array]:
    #     """Randomly perturb the measured base position and velocities."""
    #     rng, q_rng, v_rng = jax.random.split(rng, 3)
    #     q_err = 0.001 * jax.random.normal(q_rng, (7,))
    #     v_err = 0.001 * jax.random.normal(v_rng, (6,))

    #     qpos = data.qpos.at[0:7].set(data.qpos[0:7] + q_err)
    #     qvel = data.qvel.at[0:6].set(data.qvel[0:6] + v_err)

    #     return {"qpos": qpos, "qvel": qvel}
