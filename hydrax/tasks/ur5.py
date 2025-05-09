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

        return tcp_pos

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        orientation_cost = jnp.sum(jnp.square(self._get_eef_rot(state)))

        return 100.0 * orientation_cost + 100.0 * jnp.sum(jnp.square(self._get_eef_pos(state)))

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        return self.running_cost(state, jnp.zeros(self.model.nu))