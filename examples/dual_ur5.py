import argparse

import evosax
import mujoco

from hydrax.algs import CEM, MPPI, PredictiveSampling, Evosax
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.dual_ur5 import DUAL_UR5

import jax.numpy as jnp

import jax

import numpy as np

"""
Run an interactive simulation of UR5 manipulator target tracking task.
"""

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run an interactive simulation of UR5 manipulator target tracking task."
)

parser.add_argument(
    "--iterations",
    type=int,
    default=1,
    help="Number of CEM iterations.",
)

subparsers = parser.add_subparsers(
    dest="algorithm", help="Sampling algorithm (choose one)"
)
subparsers.add_parser("cem", help="Cross Entropy Method")
subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
subparsers.add_parser("ps", help="Predictive Sampling")
subparsers.add_parser("evosax", help="EvoSax")


args = parser.parse_args()

# Define the task (cost and dynamics)
task = DUAL_UR5()

# Set up the controller
if args.algorithm == "cem" or args.algorithm is None:
    print("Running Cross Entropy Method")
    ctrl = CEM(
        task,
        num_samples=100,
        num_elites=50,
        sigma_start=0.2,
        sigma_min=0.05,
        explore_fraction=0.5,
        plan_horizon=0.2,
        spline_type="linear",
        num_knots=5,
        iterations=args.iterations,
    )
elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(
            task,
            num_samples=100,
            noise_level=0.02,
            temperature=10.0,
            plan_horizon=0.2,
            spline_type="linear",
            num_knots=10,
            iterations=args.iterations,
        )
elif args.algorithm == "ps":
    print("Running Predictive Sampling")
    ctrl = PredictiveSampling(
            task,
            num_samples=100,
            noise_level=0.2,
            plan_horizon=0.2,
            spline_type="zero",
            num_knots=4,
            iterations=args.iterations,
        )
elif args.algorithm == "evosax":
    print("Running Evosax")
    ctrl = Evosax(
        task,
        evosax.Sep_CMA_ES,
        num_samples=100,
        elite_ratio=0.5,
        plan_horizon=0.2,
        spline_type="zero",
        num_knots=4,
    )       
else:
    parser.error("Invalid algorithm")

# Define the model used for simulation
mj_model = task.mj_model
mj_model.opt.timestep = 0.01
mj_model.opt.iterations = 1
mj_model.opt.ls_iterations = 5
#mj_model.opt.o_solimp = [0.9, 0.95, 0.001, 0.5, 2]
#mj_model.opt.enableflags = mujoco.mjtEnableBit.mjENBL_OVERRIDE


# Set the initial state
mj_data = mujoco.MjData(mj_model)

#jax.debug.print("mj_data.ctrl[:] {}", mj_data.ctrl[:])

# Create joint masks
joint_names_pos = []
joint_names_vel = []
for i in range(mj_model.njnt):
    joint_type = mj_model.jnt_type[i]
    n_pos = 7 if joint_type == mujoco.mjtJoint.mjJNT_FREE else 4 if joint_type == mujoco.mjtJoint.mjJNT_BALL else 1
    n_vel = 6 if joint_type == mujoco.mjtJoint.mjJNT_FREE else 3 if joint_type == mujoco.mjtJoint.mjJNT_BALL else 1
    for _ in range(n_pos):
        joint_names_pos.append(mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i))
    for _ in range(n_vel):
        joint_names_vel.append(mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i))

robot_joints = np.array([
    'shoulder_pan_joint_1', 'shoulder_lift_joint_1', 'elbow_joint_1', 
    'wrist_1_joint_1', 'wrist_2_joint_1', 'wrist_3_joint_1',
    'shoulder_pan_joint_2', 'shoulder_lift_joint_2', 'elbow_joint_2', 
    'wrist_1_joint_2', 'wrist_2_joint_2', 'wrist_3_joint_2'
])

joint_mask_pos = jnp.array(np.isin(np.array(joint_names_pos), robot_joints))
joint_mask_vel = jnp.array(np.isin(np.array(joint_names_vel), robot_joints))

# theta = state.qpos[joint_mask_pos]

mj_data.qpos[:] = jnp.zeros_like(mj_data.qpos)

mj_data.qpos[joint_mask_pos] = task.init_joint_angle
# mj_data.qpos[:12] = task.init_joint_angle



run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=20,
    show_traces=True,
    max_traces=5,
)
