"""RSL-RL runner that also checkpoints the curriculum stage and exports ONNX on save."""

from __future__ import annotations

import json

from mjlab.rl.exporter_utils import attach_metadata_to_onnx
from mjlab.rl.runner import MjlabOnPolicyRunner

CMD = "ee_target"


class So101ReachRunner(MjlabOnPolicyRunner):
  def save(self, path: str, infos=None):
    cmd = self.env.unwrapped.command_manager.get_term(CMD)
    infos = {
      **(infos or {}),
      "so101_curriculum": {"stage_idx": int(cmd.stage_idx), "stage_start_step": int(cmd.stage_start_step)},
    }
    super().save(path, infos)
    # ONNX next to the checkpoint, with this task's interface contract as metadata (MjLab's
    # generic metadata helper assumes an action term named "joint_pos").
    from so101_reach.checkpoints.policy_io import native_contract

    policy_dir, filename, onnx_path = self._get_export_paths(path)
    try:
      self.export_policy_to_onnx(str(policy_dir), filename)
      attach_metadata_to_onnx(str(onnx_path), {"policy_contract": json.dumps(native_contract(path))})
    except Exception as e:
      print(f"[WARN] ONNX export failed (training continues): {e}")

  def load(self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None):
    infos = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
    cur = (infos or {}).get("so101_curriculum")
    cmd = self.env.unwrapped.command_manager.get_term(CMD)
    # Only restore into a training env (multi-stage); play/eval envs pin their own stage.
    if cur is not None and len(cmd.stages) > 1:
      cmd.set_stage(cur["stage_idx"])
      cmd.stage_start_step = cur.get("stage_start_step", cmd.stage_start_step)
      print(f"[INFO] Restored curriculum stage {cmd.stage.name} (entered at env step {cmd.stage_start_step})")
    return infos
