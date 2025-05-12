import argparse

import mujoco

from hydrax.algs import CEM, MPPI, PredictiveSampling
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.ur5 import UR5

import jax.numpy as jnp

import jax

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


args = parser.parse_args()

# Define the task (cost and dynamics)
task = UR5()

# Set up the controller
if args.algorithm == "cem" or args.algorithm is None:
    print("Running Cross Entropy Method")
    ctrl = CEM(
        task,
        num_samples=100,
        num_elites=20,
        sigma_start=0.2,
        sigma_min=0.05,
        explore_fraction=0.5,
        plan_horizon=0.2,
        spline_type="zero",
        num_knots=4,
        iterations=args.iterations,
    )
elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(
            task,
            num_samples=100,
            noise_level=0.2,
            temperature=0.1,
            plan_horizon=0.2,
            spline_type="zero",
            num_knots=4,
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
else:
    parser.error("Invalid algorithm")

# Define the model used for simulation
mj_model = task.mj_model
mj_model.opt.timestep = 0.01
mj_model.opt.iterations = 1
mj_model.opt.ls_iterations = 5
mj_model.opt.o_solimp = [0.9, 0.95, 0.001, 0.5, 2]
mj_model.opt.enableflags = mujoco.mjtEnableBit.mjENBL_OVERRIDE

# Set the initial state
mj_data = mujoco.MjData(mj_model)
jax.debug.print("mj_data.ctrl[:] {}", mj_data.ctrl[:])

mj_data.qpos[:] = jnp.zeros_like(mj_data.qpos)
mj_data.qpos[:6] = task.init_joint_angle



run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=20,
    show_traces=True,
    max_traces=5,
)
