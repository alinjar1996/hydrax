import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task

from functools import partial

import numpy as np

class DUAL_UR5(Task):
    

    def __init__(self) -> None:
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/dual_ur5/scene.xml")
        
        # Pass the regular MuJoCo model to the parent class
        super().__init__(
            mj_model,
            trace_sites=["tcp_0", "tcp_1"],
        )

        self.mj_model = mj_model  # Regular MuJoCo model
        self.data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = 0.1
        self.nu = self.mj_model.nu
        
        # Create MJX model and data
        self.mjx_model = mjx.put_model(self.mj_model)
        self.mjx_data = mjx.put_data(self.mj_model, self.data)
        self.mjx_data = jax.jit(mjx.forward)(self.mjx_model, self.mjx_data)
        self.jit_step = jax.jit(mjx.step)
        self.jit_forward = jax.jit(mjx.forward)
             
        # # Set timestep
        
        self.mj_model.opt.timestep = 0.1
        self.mjx_model = self.mjx_model.replace(opt=self.mjx_model.opt.replace(timestep=0.1))

        # Get the hand and tcp ids using regular model
        self.hande_id_0 = self.mj_model.body(name="hande_0").id
        self.tcp_id_0 = self.mj_model.site(name="tcp_0").id
        self.hande_id_1 = self.mj_model.body(name="hande_1").id
        self.tcp_id_1 = self.mj_model.site(name="tcp_1").id

        self.init_joint_angle = jnp.array([1.5, -1.8, 1.75, -1.25, -1.6, 0, -1.5, -1.8, 1.75, -1.25, -1.6, 0])

        # Get target positions from MJX data
        self.target_pos_0 = self.mjx_data.xpos[self.mj_model.body(name="target_0").id]
        self.target_rot_0 = self.mjx_data.xquat[self.mj_model.body(name="target_0").id].copy()
        self.target_0 = jnp.concatenate([self.target_pos_0, self.target_rot_0])

        # self.target_pos_2 = self.mjx_data.xpos[self.mj_model.body(name="ball").id]
        # self.target_rot_2 = self.mjx_data.xquat[self.mj_model.body(name="ball").id].copy()
        # self.target_2 = jnp.concatenate([self.target_pos_2, self.target_rot_2])

        # Create joint masks
        joint_names_pos = list()
        joint_names_vel = list()
        for i in range(self.mj_model.njnt):
            joint_type = self.mj_model.jnt_type[i]
            n_pos = 7 if joint_type == mujoco.mjtJoint.mjJNT_FREE else 4 if joint_type == mujoco.mjtJoint.mjJNT_BALL else 1
            n_vel = 6 if joint_type == mujoco.mjtJoint.mjJNT_FREE else 3 if joint_type == mujoco.mjtJoint.mjJNT_BALL else 1
            for _ in range(n_pos):
                joint_names_pos.append(mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i))
            for _ in range(n_vel):
                joint_names_vel.append(mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i))
        
        robot_joints = np.array(['shoulder_pan_joint_1', 'shoulder_lift_joint_1', 'elbow_joint_1', 'wrist_1_joint_1', 'wrist_2_joint_1', 'wrist_3_joint_1',
                        'shoulder_pan_joint_2', 'shoulder_lift_joint_2', 'elbow_joint_2', 'wrist_1_joint_2', 'wrist_2_joint_2', 'wrist_3_joint_2'])
        self.joint_mask_pos = np.isin(np.array(joint_names_pos), robot_joints)
        self.joint_mask_vel = np.isin(np.array(joint_names_vel), robot_joints)

        # Create geom IDs for collision detection
        self.geom_ids = []
                
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name is not None and (
                name.startswith('robot') 
                or
                name.startswith('object') 
                # or 
                # name.startswith('cupboard')
                # name.startswith('target')
            ):  
                print(f"Found geom: id={i}, name='{name}'")
                self.geom_ids.append(i)

        self.geom_ids_all = np.array(self.geom_ids)
        # self.mask = jnp.any(jnp.isin(self.mjx_data.contact.geom, self.geom_ids_all), axis=1)
        
        
        self.hande_id_0 = self.model.body(name="hande_0").id
        self.tcp_id_0 = self.model.site(name="tcp_0").id
        self.hande_id_1 = self.model.body(name="hande_1").id
        self.tcp_id_1 = self.model.site(name="tcp_1").id
        self.tray_site_0_id = self.model.site(name="tray_site_0").id
        self.tray_site_1_id = self.model.site(name="tray_site_1").id
        self.tray_0_id = self.model.body(name="target_0").id
        self.tray_1_id = self.model.body(name="target_1").id
        self.tray_mocap_idx = self.model.body_mocapid[self.model.body(name='tray_mocap').id]

        # Use MJX data instead of MuJoCo data
        target0_body_id = self.mj_model.body(name="target_0").id
        target1_body_id = self.mj_model.body(name="target_1").id
        tray_target_body_id = self.mj_model.body(name="tray_mocap_target").id

        # Body targets
        self.target_pos_0 = self.mjx_data.xpos[target0_body_id]
        self.target_rot_0 = self.mjx_data.xquat[target0_body_id]
        self.target_0 = jnp.concatenate([self.target_pos_0, self.target_rot_0])

        self.target_pos_1 = self.mjx_data.xpos[target1_body_id]
        self.target_rot_1 = self.mjx_data.xquat[target1_body_id]
        self.target_1 = jnp.concatenate([self.target_pos_1, self.target_rot_1])

        # Mocap target
        mocap_id = self.mj_model.body_mocapid[tray_target_body_id]
        self.target_pos_2 = self.mjx_data.mocap_pos[mocap_id]
        self.target_rot_2 = self.mjx_data.mocap_quat[mocap_id]
        self.target_2 = jnp.concatenate([self.target_pos_2, self.target_rot_2])

                # Get references for both arms
        # self.target_pos_0 = self.data.xpos[self.model.body(name="target_0").id]
        # self.target_rot_0 = self.data.xquat[self.model.body(name="target_0").id].copy()
        # self.target_0 = jnp.concatenate([self.target_pos_0, self.target_rot_0])

        # self.target_pos_1 = self.data.xpos[self.model.body(name="target_1").id]
        # self.target_rot_1 = self.data.xquat[self.model.body(name="target_1").id].copy()
        # self.target_1 = jnp.concatenate([self.target_pos_1, self.target_rot_1])

        # self.target_pos_2 = self.data.mocap_pos[self.model.body_mocapid[self.model.body(name='tray_mocap_target').id]]
        # self.target_rot_2 = self.data.mocap_quat[self.model.body_mocapid[self.model.body(name='tray_mocap_target').id]] 
        # self.target_2 = jnp.concatenate([self.target_pos_2, self.target_rot_2])


    # def _get_eef_quat(self, state: mjx.Data) -> jax.Array:
    #     """Get the End Effector Frame (EEF) rotation."""
    #     eef_quat = state.xquat[self.hande_id_0]
    #     return eef_quat

    # def _get_eef_pos(self, state: mjx.Data) -> jax.Array:
    #     """Get the End Effector Frame (EEF) position."""
    #     tcp_pos = state.site_xpos[self.tcp_id_0]
    #     return tcp_pos
    
    # def _quaternion_distance(self, q1, q2):
    #     dot_product = jnp.abs(jnp.dot(q1, q2))
    #     dot_product = jnp.clip(dot_product, -1.0, 1.0)
    #     return 2 * jnp.arccos(dot_product)
    
    # @partial(jax.jit, static_argnums=(0,))
    # def collision_cost(self, data: mjx.Data):
    #     collision = data.contact.dist

    #     # Compute collision cost for pick
    #     y = 0.15  # Higher y implies stricter condition on g to be positive
    #     collision_pick = collision[self.mask]
    #     collision_pick = collision_pick.T

    #     # print("collision_pick", jnp.shape(collision_pick))
    #     # jax.debug.print("collision_pick {}", collision_pick)

    #     g = -collision_pick[1:] + (1 - y) * collision_pick[:-1]
    #     # cost_c_pick = jnp.sum(jnp.max(g.reshape(g.shape[0], 1), axis=-1, initial=0)) + jnp.sum(collision_pick < 0)
    #     # cost_c_pick = jnp.sum(jnp.maximum(g, 0)) + jnp.sum(collision_pick < 0)

    #     cost_c_pick = jnp.sum(collision_pick)

    #     print("cost_c_pick", cost_c_pick)


    #     # Compute collision cost for move
    #     collision_move = collision[self.mask_move]
    #     collision_move = collision_move.T
    #     g = -collision_move[1:] + (1 - y) * collision_move[:-1]
    #     # cost_c_move = jnp.sum(jnp.max(g.reshape(g.shape[0], 1), axis=-1, initial=0)) + jnp.sum(collision_move < 0)

    #     cost_c_move = jnp.sum(collision_move)



    #     return cost_c_pick, cost_c_move

    #@partial(jax.jit, static_argnums=(0,))
    def _normalize_vec(self, vec, eps: float = 1e-6):

        normalized_vec = vec/jnp.linalg.norm(vec + eps)

        return normalized_vec
       

    # def _distance_to_upright(self, state: mjx.Data) -> jax.Array:
    #     """Get a measure of distance to the upright position."""
    #     theta = state.qpos[0] - jnp.pi
    #     # jax.debug.print("theta {}", theta)
    #     theta_err = jnp.array([jnp.cos(theta) - 1, jnp.sin(theta)])
    #     # jax.debug.print("theta_err {}", theta_err)
    #     return jnp.sum(jnp.square(theta_err))
    
    #@partial(jax.jit, static_argnums=(0,))
    def _initial_state_dist_cost(self, state: mjx.Data) -> jax.Array:
        cost_theta = jnp.linalg.norm(state.qpos[self.joint_mask_pos] - self.init_joint_angle)
        return cost_theta
    
    #@partial(jax.jit, static_argnums=(0,))
    def _eef_pos_cost(self, eef_0, eef_1) -> jax.Array:
        cost_eef_pos = jnp.linalg.norm(eef_0[2] - eef_1[2])

        return cost_eef_pos
    
    #@partial(jax.jit, static_argnums=(0,))
    def _eef_vel_cost(self, eef_0, eef_1, eef_vel_lin_0, eef_vel_lin_1) -> jax.Array:
        rel_pos = eef_0[:3] - eef_1[:3]        # Shape (Batch, 3)
        rel_vel = eef_vel_lin_0 - eef_vel_lin_1 # Shape (Batch, 3)
        dot_products = jnp.sum(rel_pos * rel_vel, axis=-1)  # Shape (Batch,)
        cost_eef_vel = jnp.linalg.norm(dot_products)

        return cost_eef_vel
    
    #@partial(jax.jit, static_argnums=(0,))
    def _pick_cost(self, eef_0, eef_1, target_0, target_1):
        # Move end effectors to pick positions
        cost_g_0 = jnp.linalg.norm(eef_0[:3] - target_0[:3], axis=1)
        cost_g_1 = jnp.linalg.norm(eef_1[:3] - target_1[:3], axis=1)
        cost_g = (jnp.sum(cost_g_0) + jnp.sum(cost_g_1))/2

		# Move end effectors to pick orientation

        eef_0_quat_normalized = self._normalize_vec(eef_0[3:])
        eef_1_quat_normalized = self._normalize_vec(eef_1[3:])
        target_0_quat_normalized = self._normalize_vec(target_0[3:])
        target_1_quat_normalized = self._normalize_vec(target_1[3:])

		# dot_product = jnp.abs(jnp.dot(eef_0[3:]/jnp.linalg.norm(eef_0[3:], axis=1).reshape(1, self.num).T, target_0[3:]/jnp.linalg.norm(target_0[3:])))
        dot_product = jnp.abs(jnp.dot(eef_0_quat_normalized, target_0_quat_normalized))
        dot_product = jnp.clip(dot_product, -1.0, 1.0)
        cost_r_0 = 2 * jnp.arccos(dot_product)

		# dot_product = jnp.abs(jnp.dot(eef_1[:, 3:]/jnp.linalg.norm(eef_1[:, 3:], axis=1).reshape(1, self.num).T, target_1[3:]/jnp.linalg.norm(target_1[3:])))
        dot_product = jnp.abs(jnp.dot(eef_1_quat_normalized, target_1_quat_normalized))
        dot_product = jnp.clip(dot_product, -1.0, 1.0)
        cost_r_1 = 2 * jnp.arccos(dot_product)

        cost_r_pick = (jnp.sum(cost_r_0) + jnp.sum(cost_r_1))/2

        return cost_g, cost_r_pick
    
    #@partial(jax.jit, static_argnums=(0,))
    def _move_cost(self, eef_0, eef_1, target_0, target_1):
        #Keep end-effectors in contact with tray
        cost_g_0_move = jnp.linalg.norm(eef_0[:3] - target_0[:3])
        cost_g_1_move = jnp.linalg.norm(eef_1[:3] - target_1[:3])
        cost_g_move = (jnp.sum(cost_g_0_move) + jnp.sum(cost_g_1_move))/2 

        eef_0_quat_normalized = self._normalize_vec(eef_0[3:])
        eef_1_quat_normalized = self._normalize_vec(eef_1[3:])
        target_0_quat_normalized = self._normalize_vec(target_0[3:])
        target_1_quat_normalized = self._normalize_vec(target_1[3:])

		# Move end effectors to pick orientation
        # dot_product = jnp.abs(jnp.dot(eef_0[:, 3:]/jnp.linalg.norm(eef_0[:, 3:], axis=1).reshape(1, self.num).T, (target_0_rot/jnp.linalg.norm(target_0_rot, axis=1).reshape(1, self.num).T).T))
        dot_product = jnp.abs(jnp.dot(eef_0_quat_normalized, target_0_quat_normalized))
        dot_product = jnp.clip(dot_product, -1.0, 1.0)
        cost_r_0 = (2 * jnp.arccos(dot_product))
        
        # dot_product = jnp.abs(jnp.dot(eef_1[:, 3:]/jnp.linalg.norm(eef_1[:, 3:], axis=1).reshape(1, self.num).T, (target_1_rot/jnp.linalg.norm(target_1_rot, axis=1).reshape(1, self.num).T).T))
        dot_product = jnp.abs(jnp.dot(eef_1_quat_normalized, target_1_quat_normalized))
        dot_product = jnp.clip(dot_product, -1.0, 1.0)
        cost_r_1 = (2 * jnp.arccos(dot_product))

        cost_r_move = (jnp.sum(cost_r_0) + jnp.sum(cost_r_1))/2

        return cost_g_move, cost_r_move
    
    #@partial(jax.jit, static_argnums=(0,))
    def _tray_cost(self, tray, target_2):
        # Move tray to target position
        cost_g_tray = jnp.linalg.norm(tray[:3] - target_2[:3])
        tray_quat_normalized = self._normalize_vec(tray[3:])
        target_2_normalized = self._normalize_vec(target_2[3:])
        
        # dot_product = jnp.abs(jnp.dot(tray[:, 3:]/jnp.linalg.norm(tray[:, 3:], axis=1).reshape(1, self.num).T, target_2[3:]/jnp.linalg.norm(target_2[3:])))
        dot_product = jnp.abs(jnp.dot(tray_quat_normalized, target_2_normalized))
        dot_product = jnp.clip(dot_product, -1.0, 1.0)
        cost_r_tray = 2 * jnp.arccos(dot_product)
        cost_r_tray = jnp.sum(cost_r_tray)

        return cost_g_tray, cost_r_tray

    

    @partial(jax.jit, static_argnums=(0,))
    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ) applied from t=1 to T-1."""
        # Get quaternion of end-effector
        theta = state.qpos[self.joint_mask_pos]
        # collision_cost_pick, collision_cost_move = self.collision_cost(state)
        cost_theta = self._initial_state_dist_cost(state)

        # First arm end-effector 
        eef_pos_0 = state.site_xpos[self.tcp_id_0]
        eef_rot_0 = state.xquat[self.hande_id_0]   
        eef_0 = jnp.concatenate([eef_pos_0, eef_rot_0])
        
        # Second arm end-effector
        eef_pos_1 = state.site_xpos[self.tcp_id_1]
        eef_rot_1 = state.xquat[self.hande_id_1]    
        eef_1 = jnp.concatenate([eef_pos_1, eef_rot_1])

        # Helper functions for Jacobian computation
        def get_site_pos0(qpos):
            new_data = state.replace(qpos=qpos)
            new_data = mjx.forward(self.mjx_model, new_data)
            return new_data.site_xpos[self.tcp_id_0]

        def get_site_rot0(qpos):
            new_data = state.replace(qpos=qpos)
            new_data = mjx.forward(self.mjx_model, new_data)
            return new_data.xquat[self.hande_id_0]

        def get_site_pos1(qpos):
            new_data = state.replace(qpos=qpos)
            new_data = mjx.forward(self.mjx_model, new_data)
            return new_data.site_xpos[self.tcp_id_1]

        def get_site_rot1(qpos):
            new_data = state.replace(qpos=qpos)
            new_data = mjx.forward(self.mjx_model, new_data)
            return new_data.xquat[self.hande_id_1]

        # Compute Jacobians using JAX's automatic differentiation
        jacp0 = jax.jacfwd(get_site_pos0)(state.qpos)
        jacr0 = jax.jacfwd(get_site_rot0)(state.qpos)
        jacp1 = jax.jacfwd(get_site_pos1)(state.qpos)
        jacr1 = jax.jacfwd(get_site_rot1)(state.qpos)

        # # Analytic Jacobians from MJX
        # jacp0, jacr0 = mjx.jac_site(self.mjx_model, state, self.tcp_id_0)
        # jacp1, jacr1 = mjx.jac_site(self.mjx_model, state, self.tcp_id_1)

        # print("jacp0", jnp.shape(jacp0))
        # print("jacp1", jnp.shape(jacp1))
        # print("jacr0", jnp.shape(jacr0))
        # print("jacr1", jnp.shape(jacr1))

        # print("self.joint_mask_pos", self.joint_mask_pos)

        # Compute EEF velocities
        eef_vel_lin_0 = jacp0[:, self.joint_mask_pos] @ state.qvel[self.joint_mask_vel]
        eef_vel_ang_0 = jacr0[:, self.joint_mask_pos] @ state.qvel[self.joint_mask_vel]
        eef_vel_lin_1 = jacp1[:, self.joint_mask_pos] @ state.qvel[self.joint_mask_vel]
        eef_vel_ang_1 = jacr1[:, self.joint_mask_pos] @ state.qvel[self.joint_mask_vel]

        cost_eef_pos = self._eef_pos_cost(eef_0, eef_1)
        cost_eef_vel = self._eef_vel_cost(eef_0, eef_1, eef_vel_lin_0, eef_vel_lin_1)

        cost_eef_pick_pos, cost_eef_pick_rot = self._pick_cost(eef_0, eef_1, self.target_0, self.target_1)
        eef_cost_move_pos, eef_cost_move_rot = self._move_cost(eef_0, eef_1, self.target_0, self.target_1)

        # self._tray_cost(tray, self.target_2)




        cost_weights = {
            'collision': 15,
			'theta': 0.01,
			'z-axis': 1.1,
            'velocity': 0.02,

            'position': 2.0,
            'orientation_pick': 0.5,

            'distance': 2.0,
            'position_tray': 11.0,
            'orientation_tray': 4.9,
            'position_move':0.2,
            'orientation_move': 2,

            # 'pick': 0,
            # 'move': 0
        }

        cost = (
			# cost_weights['collision']*cost_c +
			cost_weights['theta']*cost_theta +
			cost_weights['z-axis']*cost_eef_pos +
			cost_weights['velocity']*cost_eef_vel +

            cost_weights['position']*cost_eef_pick_pos +
			cost_weights['orientation_pick']*cost_eef_pick_rot 

			 

			# cost_task_weights['pick']*cost_weights['position']*cost_g +
			# cost_task_weights['pick']*cost_weights['orientation_pick']*cost_r_pick +

			# cost_task_weights['move']*cost_weights['distance']*cost_dist +
			# cost_task_weights['move']*cost_weights['position_tray']*cost_g_tray +
			# cost_task_weights['move']*cost_weights['orientation_tray']*cost_r_tray +
			# cost_task_weights['move']*cost_weights['position_move']*cost_g_move +
			# cost_task_weights['move']*cost_weights['orientation_move']*cost_r_move 
		)	


        return cost

    #@partial(jax.jit, static_argnums=(0,))
    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        return self.running_cost(state, jnp.zeros(self.nu))