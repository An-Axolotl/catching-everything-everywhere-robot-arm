#!/usr/bin/env bash
# Run a trained catching policy on the real xArm6 + LEAP hand.
#
# Expects a training run folder laid out the way train.py saves it:
#   isaacgymenvs/runs/<RUN_NAME>/Overhand.yaml        task config of that run
#   isaacgymenvs/runs/<RUN_NAME>/nn/<CKPT>            policy weights (default CatchXArm.pth)
#
# Start roscore and the sim2real nodes first (see README), then:
#   RUN_NAME=my_run ./scripts/run_real.sh
#
# Isaac Gym runs one CPU env that mirrors the real robot state; actions are sent over ROS.
set -e
: "${RUN_NAME:?set RUN_NAME to the folder name under isaacgymenvs/runs/}"
CKPT=${CKPT:-CatchXArm.pth}
cd "$(dirname "$0")/../isaacgymenvs"
IS_REAL=True python train.py task=Overhand test=True +real=True +run_name="$RUN_NAME" \
  checkpoint="$CKPT" num_envs=1 headless=${HEADLESS:-False} force_render=True \
  wandb_activate=False pipeline=cpu sim_device=cpu "$@"
