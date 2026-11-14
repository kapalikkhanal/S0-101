"""MjLab task registration (loaded through the ``mjlab.tasks`` entry point).

Tasks:
  So101-Reach        native goal-conditioned reaching policy (train + play), Menagerie servos
  So101-Reach-Real   same task with the servo model identified on the real arm
                     (configs/servo_model_fit.json) + randomisation around it; used to
                     fine-tune the policy for hardware

The nvidia/so101-reach adapter is an evaluation-only environment
(``so101_reach.tasks.eval_env_cfg.make_eval_env_cfg(controller="nvidia")``) because the
checkpoint is an inference-only ONNX actor that cannot be loaded by the RSL-RL runner.
"""

from mjlab.tasks.registry import register_mjlab_task

from so101_reach.tasks.reach_env_cfg import make_reach_env_cfg
from so101_reach.tasks.rl_cfg import so101_reach_ppo_runner_cfg
from so101_reach.tasks.runner import So101ReachRunner

register_mjlab_task(
  task_id="So101-Reach",
  env_cfg=make_reach_env_cfg(),
  play_env_cfg=make_reach_env_cfg(play=True),
  rl_cfg=so101_reach_ppo_runner_cfg(),
  runner_cls=So101ReachRunner,
)

register_mjlab_task(
  task_id="So101-Reach-Real",
  env_cfg=make_reach_env_cfg(actuator_model="identified"),
  play_env_cfg=make_reach_env_cfg(play=True, actuator_model="identified"),
  rl_cfg=so101_reach_ppo_runner_cfg(),
  runner_cls=So101ReachRunner,
)
