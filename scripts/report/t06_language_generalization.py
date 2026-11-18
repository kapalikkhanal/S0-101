"""Test 6 - language generalization: free-text instruction -> pick-and-place plan.

The task-level advantage a VLA should have over the RL/scripted stack: understanding varied
instructions. Each seeded scene has 3 coloured objects and the white box. Instruction families:

  direct       "Put the red cube into the white box."
  paraphrase   same goal, different wording ("Could you drop the red cube in the box?")
  synonym      attribute / shape synonyms ("the crimson block")
  spatial      by position ("the object closest to the robot")
  multi        several objects ("put all the objects in the box", "the A and the B")
  stack        "stack the A on the B"
  negation     "everything except the A"

Planners:
  vla_<model>  Agent.plan (one VLM call: instruction + measured objects + head image)
  keyword      rule-based parser over exact object names (the non-learning baseline)

A plan is correct if its (pick, place_on) steps equal the ground truth (order-insensitive where
the order does not matter). Results: results/report/language.jsonl (resumable).

  python scripts/report/t06_language_generalization.py --models qwen2.5vl-tools:7b --per-family 20
"""

from __future__ import annotations

import argparse
import base64
import re
import time

import cv2
import numpy as np

from common import BOX, COLORS, DEFAULT_BASE_URL, RESULTS, TZ, append, controller, done_keys, make_llm_agent
from so101_reach.agent.arm import AgentArm
from so101_reach.agent.perception import SimPerception, annotate
from so101_reach.agent.sim_world import OpenBox, SimWorld, WorldCfg
from so101_reach.real.mouse_real import SafeTargets
from so101_reach.real.robot import SimRealPlant
from so101_reach.sim import so101
from so101_reach.sim.grasp_scene import ObjectCfg

FAMILIES = ("direct", "paraphrase", "synonym", "spatial", "multi", "stack", "negation")
SHAPES = {"cube": ("box", (0.016, 0.016, 0.016)), "ball": ("sphere", (0.016,)), "cylinder": ("cylinder", (0.013, 0.018))}
COLOR_SYN = {"red": ["crimson", "scarlet"], "blue": ["navy", "azure"], "green": ["emerald", "lime"],
             "yellow": ["golden", "lemon-coloured"], "orange": ["tangerine"], "purple": ["violet", "lilac"]}
SHAPE_SYN = {"cube": ["block", "square block"], "ball": ["sphere", "round thing"], "cylinder": ["can", "tube"]}
SPOTS = [(0.19, -0.10), (0.24, -0.02), (0.20, 0.06), (0.26, -0.12), (0.17, 0.0), (0.25, 0.08)]


def scene(seed: int):
  rng = np.random.default_rng(seed)
  colors = list(rng.choice(list(COLORS), 3, replace=False))
  shapes = [str(rng.choice(list(SHAPES))) for _ in range(3)]
  spots = [SPOTS[i] for i in rng.choice(len(SPOTS), 3, replace=False)]
  objs = {}
  for c, s, (x, y) in zip(colors, shapes, spots):
    mj, size = SHAPES[s]
    hz = size[-1] if mj != "sphere" else size[0]
    objs[f"{c} {s}"] = ObjectCfg(mj, size, (x, y, TZ + hz + 0.001), mass=0.03, friction=(1.0, 0.005, 0.0001), rgba=COLORS[c])
  return WorldCfg(TZ, objs, [], [OpenBox(BOX, (0.33, -0.19), (0.12, 0.16), 0.045)])


def instruction(family: str, labels: list[str], xy: dict, rng) -> tuple[str, list[tuple[str, str]], bool]:
  """(text, ground-truth steps, ordered?)"""
  a, b, c = list(rng.permutation(labels))
  if family == "direct":
    return f"Put the {a} into the {BOX}.", [(a, BOX)], True
  if family == "paraphrase":
    t = rng.choice([f"Could you grab the {a} and drop it in the {BOX}?", f"I need the {a} inside the {BOX}, please.",
                    f"The {a} belongs in the {BOX} - tidy it away.", f"Move the {a} over to the {BOX} and leave it there."])
    return str(t), [(a, BOX)], True
  if family == "synonym":
    col, shp = a.split()
    return f"Put the {rng.choice(COLOR_SYN[col])} {rng.choice(SHAPE_SYN[shp])} in the box.", [(a, BOX)], True
  if family == "spatial":
    kind = rng.choice(["closest", "farthest", "leftmost", "rightmost"])
    key = {"closest": lambda l: xy[l][0], "farthest": lambda l: -xy[l][0],
           "leftmost": lambda l: -xy[l][1], "rightmost": lambda l: xy[l][1]}[kind]
    tgt = min(labels, key=key)
    phrase = {"closest": "the object closest to the robot", "farthest": "the object farthest from the robot",
              "leftmost": "the leftmost object (from the robot's point of view)",
              "rightmost": "the rightmost object (from the robot's point of view)"}[kind]
    return f"Put {phrase} into the {BOX}.", [(tgt, BOX)], True
  if family == "multi":
    if rng.uniform() < 0.5:
      return f"Put all the objects into the {BOX}.", [(l, BOX) for l in labels], False
    return f"Put the {a} and the {b} in the {BOX}.", [(a, BOX), (b, BOX)], False
  if family == "stack":
    return f"Stack the {a} on top of the {b}.", [(a, b)], True
  if family == "negation":
    return f"Put everything except the {a} into the {BOX}.", [(b, BOX), (c, BOX)], False
  raise ValueError(family)


def keyword_plan(text: str, labels: list[str]) -> list[tuple[str, str]]:
  """Rule-based parser: exact object names, 'all'/'everything', 'except', 'on top of'."""
  t = text.lower()
  named = sorted([l for l in labels if l in t], key=t.index)
  if re.search(r"\bstack\b|\bon top of\b", t) and len(named) >= 2:
    return [(named[0], named[1])]
  if "except" in t:
    excl = [l for l in labels if l in t.split("except", 1)[1]]
    return [(l, BOX) for l in labels if l not in excl]
  if re.search(r"\ball\b|\beverything\b|\bevery\b", t):
    return [(l, BOX) for l in labels]
  return [(l, BOX) for l in named]


def correct(plan, truth, ordered: bool) -> bool:
  plan = [(p.lower().strip(), q.lower().strip()) for p, q in plan]
  return plan == truth if ordered else sorted(plan) == sorted(truth)


def head_b64(per, arm, width: int = 640) -> str:
  img = annotate(per.views()["head"], [], arm.geom(arm.q6()[0])["gp"], TZ, "head camera", axes=True)
  h = int(img.shape[0] * width / img.shape[1])
  return base64.b64encode(cv2.imencode(".jpg", cv2.resize(img, (width, h)), [cv2.IMWRITE_JPEG_QUALITY, 70])[1]).decode()


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--models", nargs="*", default=["qwen2.5vl-tools:7b"])
  p.add_argument("--per-family", type=int, default=20)
  p.add_argument("--base-url", default=DEFAULT_BASE_URL)
  args = p.parse_args()
  path = RESULTS / "language.jsonl"
  done = done_keys(path, ("planner", "family", "k"))
  planners = ["keyword"] + [f"vla_{m}" for m in args.models]
  for k in range(args.per_family):
    for fam in FAMILIES:
      if all((pl, fam, k) in done for pl in planners):
        continue
      seed = 400_000 + 1000 * FAMILIES.index(fam) + k
      cfg = scene(seed)
      world = SimWorld(cfg)
      world.set_arm({**so101.HOME_JOINT_POS, **so101.FIXED_JOINT_TARGETS, "wrist_roll": world.jmap.to_rad("wrist_roll", 2048)})
      plant = SimRealPlant(world, world.jmap, world.fcal)
      plant.engage()
      for _ in range(10):
        plant.wait()
      arm = AgentArm(plant, controller(0.5), SafeTargets(), TZ)
      per = SimPerception(world)
      labels = list(cfg.objects)
      xy = {l: cfg.objects[l].pos[:2] for l in labels}
      text, truth, ordered = instruction(fam, labels, xy, np.random.default_rng(seed))
      dets = [d for n in labels + [BOX] for d in per.find(n)[:1]]
      for i, d in enumerate(dets):
        d.id = i
      objects_text = "\n".join(d.describe(TZ) for d in dets)
      img = head_b64(per, arm)
      for pl in planners:
        if (pl, fam, k) in done:
          continue
        t0 = time.time()
        if pl == "keyword":
          plan, n_tok = keyword_plan(text, labels), 0
        else:
          agent = make_llm_agent(arm, per, RESULTS / "agent_logs" / "language" / f"{pl.replace(':', '_')}_{fam}_{k}",
                                 pl[4:], args.base_url)
          try:
            steps = agent.plan(text, objects_text, img)
            plan = [(str(s.get("pick", "")), str(s.get("place_on", ""))) for s in steps if isinstance(s, dict)]
          except Exception as e:  # malformed output counts as a wrong plan
            plan = [("ERROR", str(e)[:80])]
          n_tok = agent.total_usage["in"] + agent.total_usage["out"]
        ok = correct(plan, truth, ordered)
        row = {"planner": pl, "family": fam, "k": k, "seed": seed, "instruction": text, "truth": truth, "plan": plan,
               "correct": ok, "seconds": round(time.time() - t0, 2), "tokens": n_tok}
        append(path, row)
        print(f"[t06] {pl:18s} {fam:10s} {k:2d} {'OK ' if ok else 'BAD'} {text[:60]:60s} -> {plan}", flush=True)
      world.close_renderers()


if __name__ == "__main__":
  main()
