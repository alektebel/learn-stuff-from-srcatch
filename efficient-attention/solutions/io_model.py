"""
A two-level memory hierarchy you can count — GIVEN. Nothing to implement here.

Why this file exists
--------------------
FlashAttention is not a better formula. It computes *exactly* the same function
as standard attention; every number it produces is the number standard attention
produces. Its entire contribution is that it moves fewer bytes between the GPU's
big slow memory (HBM) and the small fast memory on each streaming multiprocessor
(SRAM).

On a machine with a GPU you would measure that with a profiler. There is no GPU
here, and more importantly a profiler tells you what happened without telling
you why. So instead we make the memory hierarchy explicit and *charge* for every
transfer:

    dev = Device(sram_bytes=64 * 1024)
    Qh = dev.hbm(Q, "Q")                    # a tensor living in HBM
    with dev.scope() as sram:               # everything loaded here occupies SRAM
        q = sram.load(Qh, rows=slice(0, 32))   # HBM -> SRAM, charged
        sram.store(Oh, out, rows=slice(0, 32)) # SRAM -> HBM, charged
    # leaving the scope frees the SRAM again
    print(dev.bytes_read, dev.bytes_written)

Two rules are enforced, not merely suggested:

  1. Reading or writing HBM through anything other than `load`/`store` raises.
     A block of memory you did not pay for is the exact mistake this file
     exists to prevent.
  2. Exceeding the SRAM budget raises SRAMOverflow. This is what forces block
     sizes to be *derived* from the budget rather than picked to look tidy —
     the same constraint that produces FlashAttention's B_c = ceil(M / 4d).

The numbers this produces are not a simulation of a real GPU: no DRAM
scheduling, no coalescing, no latency hiding, no warps. It counts bytes crossing
one boundary. That single number is what the FlashAttention analysis is about,
and getting it to come out at O(N^2 d^2 / M) instead of O(N^2) in your own code
is the exercise.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = ["Device", "HBMTensor", "SRAM", "SRAMOverflow", "HBMViolation"]


class SRAMOverflow(RuntimeError):
    """Raised when live SRAM exceeds the device budget: your blocks are too big."""


class HBMViolation(RuntimeError):
    """Raised when an HBM tensor is touched without going through load/store."""


class HBMTensor:
    """An array that lives in HBM. Reads and writes must be charged for.

    The underlying buffer is deliberately awkward to reach (`_buf`) so that
    accidentally doing `Qh[3] @ Kh.T` — which on real hardware would be a
    round trip to HBM per element — is a loud error rather than a silent one.
    """

    __slots__ = ("_buf", "name", "device")

    def __init__(self, buf: np.ndarray, name: str, device: "Device"):
        self._buf = buf
        self.name = name
        self.device = device

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._buf.shape

    @property
    def dtype(self):
        return self._buf.dtype

    @property
    def nbytes(self) -> int:
        return int(self._buf.nbytes)

    def __len__(self) -> int:
        return self._buf.shape[0]

    def __getitem__(self, item):
        raise HBMViolation(
            f"tensor {self.name!r} lives in HBM — indexing it directly hides the "
            "transfer this exercise is about. Use sram.load(tensor, rows=...) "
            "inside a `with dev.scope() as sram:` block.")

    def __setitem__(self, item, value):
        raise HBMViolation(
            f"tensor {self.name!r} lives in HBM — assign to it with "
            "sram.store(tensor, value, rows=...) so the write is charged.")

    def __array__(self, dtype=None, copy=None):
        raise HBMViolation(
            f"tensor {self.name!r} lives in HBM and cannot be handed to numpy "
            "wholesale. Load the block you need.")

    def to_numpy(self) -> np.ndarray:
        """Escape hatch for tests and printing: read the whole thing, uncharged.

        Only ever call this *outside* the algorithm you are measuring.
        """
        return self._buf.copy()


class SRAM:
    """The fast scratchpad, valid inside a `with device.scope() as sram:` block."""

    def __init__(self, device: "Device"):
        self.device = device
        self._live: List[int] = []

    # -- transfers -------------------------------------------------------
    def load(self, tensor: HBMTensor, rows: Optional[slice] = None,
             cols: Optional[slice] = None) -> np.ndarray:
        """Copy a block HBM -> SRAM. Charges bytes_read, occupies SRAM."""
        block = tensor._buf[rows if rows is not None else slice(None)]
        if cols is not None:
            block = block[:, cols]
        block = np.ascontiguousarray(block)
        self.device.bytes_read += block.nbytes
        self._charge(block.nbytes)
        return block

    def store(self, tensor: HBMTensor, value: np.ndarray,
              rows: Optional[slice] = None, cols: Optional[slice] = None) -> None:
        """Copy SRAM -> HBM. Charges bytes_written."""
        value = np.asarray(value)
        rows = slice(None) if rows is None else rows
        target = tensor._buf[rows] if cols is None else tensor._buf[rows, cols]
        if target.shape != value.shape:
            raise ValueError(
                f"store into {tensor.name!r} expected shape {target.shape}, "
                f"got {value.shape}")
        if cols is None:
            tensor._buf[rows] = value
        else:
            tensor._buf[rows, cols] = value
        self.device.bytes_written += value.nbytes

    # -- scratch ---------------------------------------------------------
    def alloc(self, shape, dtype=np.float64, fill: float = 0.0) -> np.ndarray:
        """On-chip scratch: occupies SRAM, no HBM traffic."""
        arr = np.full(shape, fill, dtype=dtype)
        self._charge(arr.nbytes)
        return arr

    def keep(self, array: np.ndarray) -> np.ndarray:
        """Declare an array already in registers/SRAM (e.g. a matmul result).

        Products like S = q @ k.T are materialised on chip; charging them keeps
        the SRAM budget honest.
        """
        self._charge(array.nbytes)
        return array

    # -- accounting ------------------------------------------------------
    def _charge(self, nbytes: int) -> None:
        self._live.append(nbytes)
        self.device._occupy(nbytes)

    def _release_all(self) -> None:
        for nbytes in self._live:
            self.device._release(nbytes)
        self._live.clear()


class _Scope:
    def __init__(self, device: "Device"):
        self.device = device
        self.sram = SRAM(device)

    def __enter__(self) -> SRAM:
        return self.sram

    def __exit__(self, *exc) -> bool:
        self.sram._release_all()
        return False


class Device:
    """A toy GPU: unlimited HBM, `sram_bytes` of fast on-chip memory.

    Default budget of 64 KB is roughly an A100's 164 KB shared memory per SM
    scaled down so that interesting block sizes appear at sequence lengths small
    enough to run in numpy. What matters is that it is *finite*.
    """

    def __init__(self, sram_bytes: int = 64 * 1024, name: str = "toy-gpu"):
        self.sram_bytes = int(sram_bytes)
        self.name = name
        self.reset()

    # -- lifecycle -------------------------------------------------------
    def reset(self) -> None:
        self.bytes_read = 0
        self.bytes_written = 0
        self.live_sram = 0
        self.peak_sram = 0
        self.counters: Dict[str, int] = {}
        self.allocations: List[Tuple[str, Tuple[int, ...], int]] = []

    def hbm(self, array: np.ndarray, name: str = "?") -> HBMTensor:
        """Place an array in HBM. Allocation itself moves no bytes, but the
        footprint is recorded — that is how the checker sees whether you
        materialised an N x N matrix."""
        array = np.asarray(array)
        self.allocations.append((name, tuple(array.shape), int(array.nbytes)))
        return HBMTensor(array.copy(), name, self)

    def hbm_zeros(self, shape, dtype=np.float64, name: str = "?") -> HBMTensor:
        return self.hbm(np.zeros(shape, dtype=dtype), name)

    def scope(self) -> _Scope:
        return _Scope(self)

    # -- counters --------------------------------------------------------
    def count(self, key: str, n: int = 1) -> None:
        """Tally something other than bytes (rescaling ops, blocks visited...)."""
        self.counters[key] = self.counters.get(key, 0) + int(n)

    @property
    def total_bytes(self) -> int:
        return self.bytes_read + self.bytes_written

    @property
    def largest_allocation(self) -> int:
        """Elements in the biggest HBM tensor allocated. An implementation that
        never materialises the score matrix keeps this at O(N*d)."""
        return max((int(np.prod(shape)) for _, shape, _ in self.allocations),
                   default=0)

    @property
    def hbm_footprint(self) -> int:
        return sum(nbytes for _, _, nbytes in self.allocations)

    # -- internal --------------------------------------------------------
    def _occupy(self, nbytes: int) -> None:
        self.live_sram += nbytes
        self.peak_sram = max(self.peak_sram, self.live_sram)
        if self.live_sram > self.sram_bytes:
            raise SRAMOverflow(
                f"needed {self.live_sram} bytes of SRAM but the budget is "
                f"{self.sram_bytes}. Block sizes must be derived from the "
                "budget: that constraint is where FlashAttention's "
                "B_c = ceil(M / 4d) comes from.")

    def _release(self, nbytes: int) -> None:
        self.live_sram -= nbytes

    def report(self) -> str:
        from common import human_bytes

        lines = [
            f"{self.name}: SRAM budget {human_bytes(self.sram_bytes)}",
            f"  HBM read     {human_bytes(self.bytes_read)}",
            f"  HBM written  {human_bytes(self.bytes_written)}",
            f"  HBM total    {human_bytes(self.total_bytes)}",
            f"  peak SRAM    {human_bytes(self.peak_sram)}",
            f"  largest HBM tensor {self.largest_allocation} elements",
        ]
        for key, value in sorted(self.counters.items()):
            lines.append(f"  {key:<12} {value}")
        return "\n".join(lines)
