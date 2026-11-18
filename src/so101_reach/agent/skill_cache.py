"""Persistent skill cache: fine-motion parameters distilled from successful LLM phases.

The pipeline tries local control first (cached parameters, else a scripted default) and
calls the LLM only when the local attempt observably fails. A verified LLM success is
distilled into a compact, object-relative parameter set and stored here, so the next
similar scene replays it with zero tokens.

Entries are keyed by a scene signature with no absolute positions: phase, object name,
object size (quantized), and for `place` also the target's name/size/container flag.
Values are poses relative to the measured object (same retargeting idea as
`real/grasp_replay.py`), so they transfer across object positions.

File: ``data/agent_cache/skills.json`` (git-ignored, like ``data/grasp_demos``).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # src/so101_reach/agent -> repo root
DEFAULT_DIR = PROJECT_ROOT / "data" / "agent_cache"
SIZE_QUANT_MM = 5.0  # key quantization
SIZE_TOL_MM = 10.0  # per-dimension match tolerance
STALE_AFTER_FAILURES = 2  # consecutive replay failures before the entry is skipped


def _norm_name(name: str) -> str:
  return " ".join(name.lower().split())


def _quant_size_mm(size_m) -> list[float]:
  return [round(1000 * float(v) / SIZE_QUANT_MM) * SIZE_QUANT_MM for v in size_m]


def make_key(phase: str, obj_name: str, obj_size, *, target_name: str | None = None,
             target_size=None, container: bool | None = None) -> dict:
  """Scene signature for one fine-motion phase. `obj_size`/`target_size` in metres."""
  key = {"phase": phase, "object": _norm_name(obj_name), "size_mm": _quant_size_mm(obj_size)}
  if phase == "place":
    key.update({"target": _norm_name(target_name or ""), "target_size_mm": _quant_size_mm(target_size or (0, 0, 0)),
                "container": bool(container)})
  return key


def _sizes_match(a: list[float], b: list[float]) -> bool:
  return len(a) == len(b) and all(abs(x - y) <= SIZE_TOL_MM for x, y in zip(a, b, strict=True))


def _key_matches(key: dict, other: dict) -> bool:
  if key["phase"] != other["phase"] or key["object"] != other["object"]:
    return False
  if not _sizes_match(key["size_mm"], other["size_mm"]):
    return False
  if key["phase"] == "place":
    return (key["target"] == other["target"] and key["container"] == other["container"]
            and _sizes_match(key["target_size_mm"], other["target_size_mm"]))
  return True


class SkillCache:
  """Load/save ``skills.json``; look up, record and score entries.

  An entry: {"key": ..., "params": ..., "stats": {replays, successes, failures,
  consecutive_failures, created, last_used, source, tokens_phase}}.
  """

  def __init__(self, cache_dir: Path | str = DEFAULT_DIR):
    self.path = Path(cache_dir) / "skills.json"
    self.entries: list[dict] = []
    if self.path.exists():
      try:
        self.entries = json.loads(self.path.read_text())
      except (json.JSONDecodeError, OSError):
        self.entries = []  # a corrupt cache must never break a run; it will be rewritten

  def _save(self) -> None:
    self.path.parent.mkdir(parents=True, exist_ok=True)
    self.path.write_text(json.dumps(self.entries, indent=1) + "\n")

  def lookup(self, key: dict) -> dict | None:
    """The matching non-stale entry, or None."""
    for e in self.entries:
      if _key_matches(key, e["key"]) and e["stats"].get("consecutive_failures", 0) < STALE_AFTER_FAILURES:
        return e
    return None

  def record(self, key: dict, params: dict, tokens_phase: int, source: str = "llm") -> dict:
    """Store (or overwrite) the entry for `key` with freshly distilled `params`."""
    now = round(time.time(), 1)
    stats = {"replays": 0, "successes": 0, "failures": 0, "consecutive_failures": 0,
             "created": now, "last_used": now, "source": source, "tokens_phase": int(tokens_phase)}
    for e in self.entries:
      if _key_matches(key, e["key"]):
        stats["created"] = e["stats"].get("created", now)
        e.update(key=key, params=params, stats=stats)
        self._save()
        return e
    e = {"key": key, "params": params, "stats": stats}
    self.entries.append(e)
    self._save()
    return e

  def hit(self, entry: dict, success: bool) -> None:
    """Report the outcome of one replay of `entry` (updates stats and staleness)."""
    s = entry["stats"]
    s["replays"] = s.get("replays", 0) + 1
    s["last_used"] = round(time.time(), 1)
    if success:
      s["successes"] = s.get("successes", 0) + 1
      s["consecutive_failures"] = 0
    else:
      s["failures"] = s.get("failures", 0) + 1
      s["consecutive_failures"] = s.get("consecutive_failures", 0) + 1
    self._save()

  def summary(self) -> str:
    n = len(self.entries)
    stale = sum(1 for e in self.entries if e["stats"].get("consecutive_failures", 0) >= STALE_AFTER_FAILURES)
    return f"{n} cached skill(s)" + (f" ({stale} stale)" if stale else "")
