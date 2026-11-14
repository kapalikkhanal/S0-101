"""Run a LeRobot CLI with retries on servo bus reads/writes, and S/Space episode control.

1. LeRobot gives up on the first corrupted status packet (num_retry=0), which crashes
   teleop/record on an occasional noisy reply. This forces at least RETRIES retries.
2. For `record`, episodes no longer start on their own. Between episodes the follower
   keeps following the leader (nothing saved) until you press a key in this terminal:
       S       start recording an episode
       Space   stop and save the episode
       <-      discard the episode being recorded (back to waiting)
       Esc     save what was recorded and quit
3. SO101_GRIP_FORCE=<percent> (set by record.sh --grip-force): limits how hard the follower's gripper
   squeezes, as % of the servo's maximum torque (LeRobot's default and the upper limit here is 50%).

Usage: python lerobot_run.py {record|teleoperate|replay} [lerobot args...]
"""
import functools
import os
import sys
import threading
import time

from lerobot.motors.motors_bus import MotorsBus

RETRIES = 3
RED, GREEN, YELLOW, BOLD, RESET = "\033[1;41;97m", "\033[1;42;97m", "\033[1;33m", "\033[1m", "\033[0m"


def _with_retries(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        kwargs["num_retry"] = max(kwargs.get("num_retry", 0), RETRIES)
        return fn(*args, **kwargs)
    return wrapper


for name in ("_read", "_write", "_sync_read"):
    setattr(MotorsBus, name, _with_retries(getattr(MotorsBus, name)))


GRIP_FORCE_MAX = 50  # % - LeRobot sets the gripper's Max_Torque_Limit to 50% to avoid burning the servo


def patch_grip_force(pct):
    """After LeRobot configures the follower, set the gripper's live torque limit (RAM register, takes effect
    immediately, back to the 50% default after a power cycle). Closing on an object, the servo pushes with at
    most this torque, so the object is held instead of crushed."""
    from lerobot.robots.so_follower.so_follower import SOFollower

    pct = float(pct)
    if not 1 <= pct <= GRIP_FORCE_MAX:
        sys.exit(f"grip force must be between 1 and {GRIP_FORCE_MAX} (% of servo max torque), got {pct:g}")
    orig = SOFollower.configure

    @functools.wraps(orig)
    def configure(self):
        orig(self)
        value = int(round(pct * 10))  # register unit: 0.1 % of max torque
        self.bus.write("Torque_Limit", "gripper", value, normalize=False)
        got = self.bus.read("Torque_Limit", "gripper", normalize=False)
        print(f"{YELLOW}gripper force limited to {got / 10:g}% of max torque{RESET}", flush=True)

    SOFollower.configure = configure


if os.environ.get("SO101_GRIP_FORCE"):
    patch_grip_force(os.environ["SO101_GRIP_FORCE"])


def init_keyboard_listener():
    """Read keys from this terminal (works locally and over SSH; pynput needs X focus)."""
    import atexit
    import select
    import termios
    import tty

    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False, "recording": False}
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)  # keys arrive immediately without Enter, no echo; Ctrl+C still works
    atexit.register(termios.tcsetattr, fd, termios.TCSADRAIN, old)

    def handle(key):
        if key == "esc":
            print(f"\n{YELLOW}[Esc] stopping, saving recorded episodes...{RESET}", flush=True)
            events["stop_recording"] = True
            events["exit_early"] = True
        elif events["recording"]:
            if key == " ":
                events["exit_early"] = True
            elif key == "left":
                events["rerecord_episode"] = True
                events["exit_early"] = True
        elif key in ("s", "S"):
            events["exit_early"] = True

    def reader():
        while True:
            ch = os.read(fd, 1).decode(errors="ignore")
            if ch == "\x1b":  # Esc alone, or the start of an arrow-key sequence
                if select.select([fd], [], [], 0.05)[0]:
                    seq = os.read(fd, 2).decode(errors="ignore")
                    handle("left" if seq == "[D" else "other")
                else:
                    handle("esc")
            else:
                handle(ch)

    threading.Thread(target=reader, daemon=True).start()
    return None, events


def _rec_timer(events, ep, stop):
    t0 = time.time()
    while not stop.wait(0.5):
        print(f"\r{RED} ● RECORDING episode {ep} {RESET}  {time.time() - t0:5.1f}s   "
              f"(Space = stop & save, <- = discard)", end="", flush=True)


def patch_record(rec):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    orig_loop = rec.record_loop

    @functools.wraps(orig_loop)
    def record_loop(*args, **kwargs):
        dataset = kwargs.get("dataset")
        events = kwargs["events"]
        if dataset is None:  # LeRobot's timed reset phase: unused, waiting happens below
            return orig_loop(*args, **kwargs)

        ep = dataset.num_episodes
        print(f"\n{BOLD}--- Episode {ep}: press S to start recording "
              f"(arm is live, NOT recording). Esc = quit ---{RESET}", flush=True)
        events["exit_early"] = False
        orig_loop(*args, **dict(kwargs, dataset=None, control_time_s=10**9))
        events["exit_early"] = False
        if events["stop_recording"]:
            return

        events["recording"] = True
        stop = threading.Event()
        timer = threading.Thread(target=_rec_timer, args=(events, ep, stop), daemon=True)
        t0 = time.time()
        timer.start()
        try:
            orig_loop(*args, **kwargs)
        finally:
            stop.set()
            timer.join()
            events["recording"] = False
        secs = time.time() - t0
        if events["rerecord_episode"]:
            print(f"\n{YELLOW} ■ STOPPED - episode {ep} DISCARDED ({secs:.1f}s){RESET}", flush=True)
        else:
            print(f"\n{GREEN} ■ STOPPED - saving episode {ep} ({secs:.1f}s)... {RESET}", flush=True)

    orig_save = LeRobotDataset.save_episode

    @functools.wraps(orig_save)
    def save_episode(self, *args, **kwargs):
        buf = self.episode_buffer
        if not kwargs.get("episode_data") and (buf is None or buf.get("size", 0) == 0):
            return  # quit while waiting: nothing to save
        orig_save(self, *args, **kwargs)
        print(f"{GREEN} ✔ Saved. Dataset now has {self.num_episodes} episodes. {RESET}", flush=True)

    rec.record_loop = record_loop
    rec.init_keyboard_listener = init_keyboard_listener
    LeRobotDataset.save_episode = save_episode


if __name__ == "__main__":
    cmd = sys.argv.pop(1)
    if cmd == "record":
        from lerobot.scripts import lerobot_record

        patch_record(lerobot_record)
        main = lerobot_record.main
    elif cmd == "teleoperate":
        from lerobot.scripts.lerobot_teleoperate import main
    elif cmd == "replay":
        from lerobot.scripts.lerobot_replay import main
    else:
        sys.exit(f"unknown command {cmd!r}")
    main()
