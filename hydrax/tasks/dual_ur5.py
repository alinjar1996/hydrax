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
        super().__init__(
            mj_model,
            trace_sites=["tcp_0", "tcp_1"],
        )

        self.model = mjx.put_model(mj_model)
             
        # Get the hande and tcp ids
        self.tcp_id_0 = mj_model.site(name="tcp_0").id
        self.hande_id_0 = mj_model.body(name="hande_0").id
        self.tcp_id_1 = mj_model.site(name="tcp_1").id
        self.hande_id_1 = mj_model.body(name="hande_1").id


        # self.hande_id = mj_model.body(name="hande").id
        # self.tcp_id = mj_model.site(name="tcp").id

        # self.init_joint_angle = jnp.array([1.5, -1.8, 1.75, -1.25, -1.6, 0])
        self.init_joint_angle = jnp.array([1.5, -1.8, 1.75, -1.25, -1.6, 0, -1.5, -1.8, 1.75, -1.25, -1.6, 0])

        # self.target_pos =  mj_model.body(name="target_1").pos
        # self.target_quat =  mj_model.body(name="target_1").quat


        mj_data = mujoco.MjData(mj_model)

        # self.mjx_data = mjx.put_data(mj_model, self.mjx_data)
        mujoco.mj_forward(mj_model, mj_data)

        self.target_pos_0 = mj_data.xpos[mj_model.body(name="target_0").id]
        self.target_rot_0 = mj_data.xquat[mj_model.body(name="target_0").id].copy()
        self.target_0 = jnp.concatenate([self.target_pos_0, self.target_rot_0])

        self.target_pos_2 = mj_data.xpos[mj_model.body(name="ball").id]
        self.target_rot_2 = mj_data.xquat[mj_model.body(name="ball").id].copy()
        self.target_2 = jnp.concatenate([self.target_pos_2, self.target_rot_2])


        joint_names_pos = list()
        joint_names_vel = list()
        for i in range(mj_model.njnt):
            joint_type = mj_model.jnt_type[i]
            n_pos = 7 if joint_type == mujoco.mjtJoint.mjJNT_FREE else 4 if joint_type == mujoco.mjtJoint.mjJNT_BALL else 1
            n_vel = 6 if joint_type == mujoco.mjtJoint.mjJNT_FREE else 3 if joint_type == mujoco.mjtJoint.mjJNT_BALL else 1
            for _ in range(n_pos):
                joint_names_pos.append(mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i))
            for _ in range(n_vel):
                joint_names_vel.append(mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i))
        robot_joints = np.array(['shoulder_pan_joint_1', 'shoulder_lift_joint_1', 'elbow_joint_1', 'wrist_1_joint_1', 'wrist_2_joint_1', 'wrist_3_joint_1',
						'shoulder_pan_joint_2', 'shoulder_lift_joint_2', 'elbow_joint_2', 'wrist_1_joint_2', 'wrist_2_joint_2', 'wrist_3_joint_2'])
        self.joint_mask_pos = np.isin(np.array(joint_names_pos), robot_joints)
        self.joint_mask_vel = np.isin(np.array(joint_names_vel), robot_joints)


        self.geom_ids = []
        for i in range(mj_model.ngeom):
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name is not None and (
				name.startswith('robot') 
				or
				name.startswith('object') 
				# name.startswith('target')
			):  
                # print(f"Found geom: id={i}, name='{name}'")
                self.geom_ids.append(i)

        self.geom_ids_all = np.array(self.geom_ids)
        self.mask = jnp.any(jnp.isin(mj_data.contact.geom, self.geom_ids_all), axis=1)

        ball_geom_id = np.array([mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, 'ball')])
        wall_geom_id = np.array([
				mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, 'ball'),
				mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, 'wall_0'),
				mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, 'ball_pick'),
			])
		# self.geom_ids_all = np.concatenate([self.geom_ids_all, target_geom_id])
        ball_mask = ~jnp.any(jnp.isin(mj_data.contact.geom, ball_geom_id), axis=1)
        wall_mask = jnp.all(jnp.isin(mj_data.contact.geom, wall_geom_id), axis=1)
        wall_mask = jnp.logical_or(ball_mask, wall_mask)
        self.mask_move = jnp.logical_and(ball_mask, self.mask)
        self.mask_move = jnp.logical_or(wall_mask, self.mask_move)
        

        
        
        

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
    def collision_cost(self, data: mjx.Data):
        
        data = jax.jit(mjx.forward)(self.model, data)

        data = jax.jit(mjx.step)(self.model, data)

        collision = data.contact.dist[self.mask]

        # Compute collision cost for pick
        y = 0.15 # Higher y implies stricter condition on g to be positive
        collision_pick = collision[self.mask]

        collision_pick = collision_pick.T

        # g = -collision_pick[:, 1:]+(1 - y)*collision_pick[:, :-1]
        g = -collision_pick[1:]+(1 - y)*collision_pick[:-1]

        cost_c_pick = jnp.sum(jnp.max(g.reshape(g.shape[0], 1), axis=-1, initial=0)) + jnp.sum(collision_pick < 0)
        # cost_c_pick = jnp.sum(jnp.maximum(g, 0)) + jnp.sum(collision_pick < 0)


		# Compute collision cost for move
        y = 0.15 # Higher y implies stricter condition on g to be positive
        collision_move = collision[self.mask_move]
        collision_move = collision_move.T
        # g = -collision_move[:, 1:]+(1 - y)*collision_move[:, :-1]
        g = -collision_move[1:]+(1 - y)*collision_move[:-1]
        # cost_c_move = jnp.sum(jnp.maximum(g, 0)) + jnp.sum(collision_move < 0)
        cost_c_move = jnp.sum(jnp.max(g.reshape(g.shape[0], 1), axis=-1, initial=0)) + jnp.sum(collision_move < 0)

        # if task == "pick":
        #     self.cost_weights['pick'] = 1
        #     self.cost_weights['move'] = 0
        #     self.ball_pick_init = self.target_2
        # elif task == 'move':
        #     self.cost_weights['distance'] = 50.0
        #     self.cost_weights['pick'] = 0
        #     self.cost_weights['move'] = 1
        #     self.ball_pick_init[2] = 2

        # return cost_weights['pick']*cost_weights['collision']*cost_c_pick + cost_weights['move']*cost_weights['collision']*cost_c_move
        return cost_c_pick, cost_c_move
    
    @partial(jax.jit, static_argnums=(0,))
    def initial_state_dist_cost(self, theta):

        cost_theta = jnp.linalg.norm(theta - self.init_joint_angle)

        return cost_theta
    
    @partial(jax.jit, static_argnums=(0,))
    def eef_cost(self, eef_0, eef_1, eef_vel_lin_0, eef_vel_lin_1):

        #EEF Y Z at same level
        cost_eef_pos = jnp.linalg.norm(eef_0[1:3] - eef_1[1:3])
        rel_pos = eef_0[:3] - eef_1[:3] 
        #EEF relative velocity perpendicular to the line of contacts       
        rel_vel = eef_vel_lin_0 - eef_vel_lin_1 
        dot_products = jnp.sum(rel_pos * rel_vel, axis=-1)  
        cost_eef_vel = jnp.linalg.norm(dot_products)

        # Move end effectors to correct orientation
        target_rot_0 = jnp.array([0.183, -0.683, -0.683, 0.183])
        dot_product = jnp.abs(jnp.dot(eef_0[3:]/jnp.linalg.norm(eef_0[3:]), target_rot_0/jnp.linalg.norm(target_rot_0)))
        dot_product = jnp.clip(dot_product, -1.0, 1.0)
        cost_r_0 = 2 * jnp.arccos(dot_product)
        target_rot_1 = jnp.array([0.183, -0.683, 0.683, -0.183])
        dot_product = jnp.abs(jnp.dot(eef_1[3:]/jnp.linalg.norm(eef_1[3:]), target_rot_1/jnp.linalg.norm(target_rot_1)))
        dot_product = jnp.clip(dot_product, -1.0, 1.0)
        cost_r_1 = 2 * jnp.arccos(dot_product)
        cost_eef_r = (jnp.sum(cost_r_0) + jnp.sum(cost_r_1))/2

        return cost_eef_pos, cost_eef_vel, cost_eef_r
    
    @partial(jax.jit, static_argnums=(0,))
    def dist_eef_obj_target_cost(self, eef_0, eef_1):
        center_pos = (eef_0[:3] + eef_1[:3])/2 + jnp.array([0, 0, 0.05])
        obj_goal = center_pos - self.target_0[:3]
        obj_goal_dist = jnp.linalg.norm(obj_goal)

        # Approach the ball with some offset
        distances = jnp.linalg.norm(eef_0[:3] - eef_1[:3])
        cost_dist = jnp.sum((distances - 0.19)**2)

		# Distance between center point between two eef and object with the offset
        center_point = (eef_0[:3]+eef_1[:3])/2
        center_obj_dist = jnp.linalg.norm(center_point - (self.target_2[:3]-jnp.array([0, 0, 0.05])))
        eef_obj_dist = jnp.sum(center_obj_dist)
		# Push ball to target location
        obj_goal_dist = jnp.sum(obj_goal_dist)

        return cost_dist, eef_obj_dist, obj_goal_dist
    



    
    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ) applied from t=1 to T-1."""

        
        # Get quaternion of end-effector
        theta = state.qpos[self.joint_mask_pos]

        # jax.debug.print("theta {}", theta)
        collision_cost_pick, collision_cost_move = self.collision_cost(state)

        initial_state_dist_cost = self.initial_state_dist_cost(theta)

        # First arm end-effector 
        eef_pos_0 = state.site_xpos[self.tcp_id_0]
        eef_rot_0 = state.xquat[self.hande_id_0]   
        eef_0 = jnp.concatenate([eef_pos_0, eef_rot_0])
		
		# Second arm end-effector
        eef_pos_1 = state.site_xpos[self.tcp_id_1]
        eef_rot_1 = state.xquat[self.hande_id_1]    
        eef_1 = jnp.concatenate([eef_pos_1, eef_rot_1])

        # Compute Jacobians using the current MJX API
        def get_site_pos0(qpos):
			# Create new data with updated qpos
            new_data = state.replace(qpos=qpos)
			# Forward kinematics
            new_data = mjx.forward(self.model, new_data)
            return new_data.site_xpos[self.tcp_id_0]
        def get_site_rot0(qpos):
            new_data = state.replace(qpos=qpos)
            new_data = mjx.forward(self.model, new_data)
            return new_data.xquat[self.hande_id_0]
        def get_site_pos1(qpos):
            new_data = state.replace(qpos=qpos)
            new_data = mjx.forward(self.model, new_data)
            return new_data.site_xpos[self.tcp_id_1]
        def get_site_rot1(qpos):
            new_data = state.replace(qpos=qpos)
            new_data = mjx.forward(self.model, new_data)
            return new_data.xquat[self.hande_id_1]
        # Compute Jacobians using JAX's automatic differentiation
        jacp0 = jax.jacfwd(get_site_pos0)(state.qpos)
        jacr0 = jax.jacfwd(get_site_rot0)(state.qpos)
        jacp1 = jax.jacfwd(get_site_pos1)(state.qpos)
        jacr1 = jax.jacfwd(get_site_rot1)(state.qpos)

		# Compute EEF velocities
        eef_vel_lin_0 = jacp0[:, self.joint_mask_pos] @ state.qvel[self.joint_mask_vel]
        eef_vel_ang_0 = jacr0[:, self.joint_mask_pos] @ state.qvel[self.joint_mask_vel]
        eef_vel_lin_1 = jacp1[:, self.joint_mask_pos] @ state.qvel[self.joint_mask_vel]
        eef_vel_ang_1 = jacr1[:, self.joint_mask_pos] @ state.qvel[self.joint_mask_vel]

        eef_cost_pos, eef_cost_vel, eef_cost_rot = self.eef_cost(eef_0, eef_1, eef_vel_lin_0, eef_vel_lin_1)

        cost_dist, eef_obj_dist, obj_goal_dist = self.dist_eef_obj_target_cost(eef_0, eef_1)

        cost_weights = {
            'collision': 500,
			'theta': 0.3,
			'z-axis': 10.0,
            'velocity': 0.1,
            'distance': 7.0,

            'allign': 2.0,
            'orientation': 7.0,
            'eef_to_obj': 10.0,
            'obj_to_targ': 1.0,

            'pick': 1.0,
            'move': 0.0
        }

        cost = (
			cost_weights['pick']*cost_weights['collision']*collision_cost_pick +
			cost_weights['move']*cost_weights['collision']*collision_cost_move +
			cost_weights['theta']*initial_state_dist_cost +
			cost_weights['velocity']*eef_cost_vel +
			cost_weights['z-axis']*eef_cost_pos +
            cost_weights['orientation']*eef_cost_rot +
			cost_weights['distance']*cost_dist +
			cost_weights['pick']*cost_weights['eef_to_obj']*eef_obj_dist +
			cost_weights['move']*cost_weights['obj_to_targ']*obj_goal_dist
		)

        # Weighted sum
        #return 50.0 * position_cost + 5.0 * orientation_cost + 0.0 * collision_cost + 1.0*eef_cost
        return cost

        # return theta

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        #print(self.model.nu)
        return self.running_cost(state, jnp.zeros(self.model.nu))
        