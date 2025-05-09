import argparse

import mujoco

from hydrax.algs import CEM
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.ur5 import UR5

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

args = parser.parse_args()

# Define the task (cost and dynamics)
task = UR5()

# Set up the controller
ctrl = CEM(
    task,
    num_samples=512,
    num_elites=20,
    sigma_start=0.2,
    sigma_min=0.05,
    explore_fraction=0.5,
    plan_horizon=0.6,
    spline_type="zero",
    num_knots=4,
    iterations=args.iterations,
)

# Define the model used for simulation
mj_model = task.mj_model
mj_model.opt.timestep = 0.01
mj_model.opt.iterations = 10
mj_model.opt.ls_iterations = 50
mj_model.opt.o_solimp = [0.9, 0.95, 0.001, 0.5, 2]
mj_model.opt.enableflags = mujoco.mjtEnableBit.mjENBL_OVERRIDE

# Set the initial state
mj_data = mujoco.MjData(mj_model)
mj_data.qpos[:] = task.reference[0]
initial_knots = task.reference[: ctrl.num_knots, 7:]

if args.show_reference:
    reference = task.reference
else:
    reference = None

run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=100,
    show_traces=False,
    reference=reference,
    reference_fps=task.reference_fps,
    initial_knots=initial_knots,
)
