"""Minimal Feetech STS3215 bus access for the SO-101 follower (scservo_sdk, protocol 0).

Kept independent of LeRobot so it can live in this project's environment; it uses the
LeRobot calibration file (homing offsets are already stored in the servos' EEPROM by
`lerobot-calibrate`, so Present_Position is offset-corrected).

Register addresses (STS3215 control table):
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import scservo_sdk as scs

# name: (address, length)
REG = {
  "Firmware_Major": (0, 1),
  "Min_Position_Limit": (9, 2),
  "Max_Position_Limit": (11, 2),
  "Max_Torque_Limit": (16, 2),
  "P_Coefficient": (21, 1),
  "D_Coefficient": (22, 1),
  "I_Coefficient": (23, 1),
  "Homing_Offset": (31, 2),
  "Operating_Mode": (33, 1),
  "Torque_Enable": (40, 1),
  "Acceleration": (41, 1),
  "Goal_Position": (42, 2),
  "Goal_Velocity": (46, 2),
  "Torque_Limit": (48, 2),
  "Lock": (55, 1),
  "Present_Position": (56, 2),
  "Present_Velocity": (58, 2),
  "Present_Load": (60, 2),
  "Present_Voltage": (62, 1),
  "Present_Temperature": (63, 1),
  "Moving": (66, 1),
  "Present_Current": (69, 2),
}
# Sign-magnitude encodings: register -> sign bit.
SIGN_BIT = {"Homing_Offset": 11, "Present_Velocity": 15, "Goal_Velocity": 15, "Present_Load": 10}

TICKS_PER_REV = 4096
from so101_reach.settings import FOLLOWER_CALIB as DEFAULT_CALIB
JOINT_ORDER = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def decode_sign(value: int, bit: int) -> int:
  return -(value & ((1 << bit) - 1)) if value & (1 << bit) else value


@dataclass(frozen=True)
class MotorCalib:
  id: int
  drive_mode: int
  homing_offset: int
  range_min: int
  range_max: int


def load_calibration(path: Path = DEFAULT_CALIB) -> dict[str, MotorCalib]:
  raw = json.loads(Path(path).read_text())
  return {name: MotorCalib(**raw[name]) for name in JOINT_ORDER}


class FeetechBus:
  def __init__(self, port: str = "/dev/ttyACM0", baudrate: int = 1_000_000, calib: dict[str, MotorCalib] | None = None):
    self.port = scs.PortHandler(port)
    self.ph = scs.PacketHandler(0)
    self.baudrate = baudrate
    self.calib = calib or load_calibration()
    self.ids = {name: c.id for name, c in self.calib.items()}

  def connect(self) -> None:
    if not self.port.openPort():
      raise ConnectionError(f"Cannot open {self.port.port_name}")
    if not self.port.setBaudRate(self.baudrate):
      raise ConnectionError(f"Cannot set baud rate {self.baudrate}")

  def close(self) -> None:
    self.port.closePort()

  def __enter__(self) -> FeetechBus:
    self.connect()
    return self

  def __exit__(self, *exc) -> None:
    self.close()

  def _check(self, result: int, error: int, what: str) -> None:
    """Communication failures raise. Servo status flags (overload, overheat, ...) only warn:
    the reply still carries valid data and the servo keeps reporting the flag until the
    condition clears, so raising would block every later access."""
    if result != scs.COMM_SUCCESS:
      raise IOError(f"{what}: {self.ph.getTxRxResult(result)}")
    if error:
      import warnings

      warnings.warn(f"{what}: servo status {self.ph.getRxPacketError(error)}", stacklevel=3)

  # Single-servo access.

  def ping(self, motor_id: int) -> int | None:
    model, result, error = self.ph.ping(self.port, motor_id)
    return model if result == scs.COMM_SUCCESS else None

  def read(self, name: str, reg: str) -> int:
    addr, n = REG[reg]
    fn = self.ph.read1ByteTxRx if n == 1 else self.ph.read2ByteTxRx
    val, result, error = fn(self.port, self.ids[name], addr)
    self._check(result, error, f"read {reg} of {name}")
    return decode_sign(val, SIGN_BIT[reg]) if reg in SIGN_BIT else val

  def write(self, name: str, reg: str, value: int) -> None:
    addr, n = REG[reg]
    fn = self.ph.write1ByteTxRx if n == 1 else self.ph.write2ByteTxRx
    result, error = fn(self.port, self.ids[name], addr, int(value))
    self._check(result, error, f"write {reg} of {name}")

  # Synchronous multi-servo access (one bus transaction).

  def sync_read(self, reg: str, names: tuple[str, ...] = JOINT_ORDER) -> dict[str, int]:
    addr, n = REG[reg]
    group = scs.GroupSyncRead(self.port, self.ph, addr, n)
    for name in names:
      group.addParam(self.ids[name])
    result = group.txRxPacket()
    if result != scs.COMM_SUCCESS:
      raise IOError(f"sync_read {reg}: {self.ph.getTxRxResult(result)}")
    out = {}
    for name in names:
      if not group.isAvailable(self.ids[name], addr, n):
        raise IOError(f"sync_read {reg}: no data from {name}")
      v = group.getData(self.ids[name], addr, n)
      out[name] = decode_sign(v, SIGN_BIT[reg]) if reg in SIGN_BIT else v
    return out

  def sync_write(self, reg: str, values: dict[str, int]) -> None:
    addr, n = REG[reg]
    group = scs.GroupSyncWrite(self.port, self.ph, addr, n)
    for name, v in values.items():
      v = int(v)
      data = [v & 0xFF] if n == 1 else [v & 0xFF, (v >> 8) & 0xFF]
      group.addParam(self.ids[name], data)
    result = group.txPacket()
    if result != scs.COMM_SUCCESS:
      raise IOError(f"sync_write {reg}: {self.ph.getTxRxResult(result)}")
