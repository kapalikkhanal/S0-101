"""Persist unresolved real payloads across restarts."""
import json
from so101_reach.sim import so101
STATE=so101.PROJECT_ROOT/'logs/manipulation/held_payload.json'

def require_empty(confirmed_empty=False):
  if confirmed_empty:STATE.unlink(missing_ok=True)
  elif STATE.exists():
    raise RuntimeError('A payload may still be held. Complete recovery first. Only after physically verifying an empty gripper use --empty-gripper to clear this record.')

def record_payload(data,log_dir):
  STATE.parent.mkdir(parents=True,exist_ok=True)
  tmp=STATE.with_suffix('.tmp')
  tmp.write_text(json.dumps({'log':str(log_dir),'attachment':data,'state':'possibly_held'},default=lambda a:a.tolist() if hasattr(a,'tolist') else str(a),indent=2)+'\n')
  tmp.replace(STATE)
