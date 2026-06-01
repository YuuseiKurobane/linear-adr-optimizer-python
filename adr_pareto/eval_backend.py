from __future__ import annotations

import os
import platform
import stat
import struct
import subprocess
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

from .candidates import key_of
from .config import SearchConfig
from .models import Candidate, Point


REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER_ENV = "ADR_SIMULATOR_HELPER"

CMD_CONFIGURE = 1
CMD_EVALUATE_MANY = 2
CMD_SAFETY_MANY = 3
CMD_DR_SUMMARY_MANY = 4
CMD_CLOSE = 255


class RustBackendError(RuntimeError):
    pass


def normalize(values: Iterable[float]) -> list[float]:
    out = [float(value) for value in values]
    total = sum(out)
    if total <= 0:
        raise ValueError(f"Cannot normalize {out}")
    return [value / total for value in out]


def parse_dr(value: float) -> float:
    dr = float(value)
    if dr > 1.0:
        dr /= 100.0
    if not 0.0 < dr < 1.0:
        raise ValueError(
            f"DR must be between 0 and 1, or 0 and 100 percent; got {value}"
        )
    return dr


def resolve_helper_path() -> Path:
    env_path = os.environ.get(HELPER_ENV)
    if env_path:
        path = Path(env_path)
        if not path.exists():
            raise RustBackendError(f"{HELPER_ENV} does not exist: {path}")
        return path

    system = platform.system().lower()
    machine = platform.machine().lower()
    exe = "adr-simulator-helper.exe" if system == "windows" else "adr-simulator-helper"
    artifact_dir = None
    if system == "windows":
        artifact_dir = "adr-simulator-helper-windows-x86_64"
    elif system == "darwin" and machine in {"arm64", "aarch64"}:
        artifact_dir = "adr-simulator-helper-macos-aarch64"
    elif system == "darwin" and machine in {"x86_64", "amd64"}:
        artifact_dir = "adr-simulator-helper-macos-x86_64"
    elif system == "linux" and machine in {"x86_64", "amd64"}:
        artifact_dir = "adr-simulator-helper-linux-x86_64"

    candidates = [
        REPO_ROOT / "rust" / "target" / "release" / exe,
        REPO_ROOT / "rust" / "target" / "debug" / exe,
        REPO_ROOT / "helper" / exe,
    ]
    if artifact_dir is not None:
        candidates.append(REPO_ROOT / "helper" / artifact_dir / exe)
    for path in candidates:
        if path.exists():
            if os.name != "nt":
                path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            return path

    searched = "\n".join(str(path) for path in candidates)
    raise RustBackendError(
        "No adr-simulator-helper binary found. Build it with:\n"
        "  cd rust\n"
        "  cargo build --release --bin adr-simulator-helper\n"
        f"Or set {HELPER_ENV}. Searched:\n{searched}"
    )


class RustEvalLib:
    def __init__(self, eval_config: dict[str, Any], threads: int) -> None:
        self.helper_path = resolve_helper_path()
        self._proc = self._start_process()
        self.threads = 0
        self._configure(eval_config, threads)

    def evaluate_many(
        self,
        candidates: list[Candidate],
        weight: float,
        seed: int,
    ) -> list[list[float | int]]:
        payload = bytearray()
        payload.append(CMD_EVALUATE_MANY)
        payload += struct.pack("<fQ", float(weight), int(seed))
        _pack_candidates(payload, candidates)
        response = self._request(payload)
        offset, count = _unpack_u32(response, 0)
        rows: list[list[float | int]] = []
        for _ in range(count):
            values = struct.unpack_from("<fffddiddd", response, offset)
            offset += struct.calcsize("<fffddiddd")
            rows.append(list(values))
        _ensure_consumed(response, offset)
        return rows

    def safety_many(
        self,
        candidates: list[Candidate],
        s_max: float,
        max_checks: int,
    ) -> list[list[float | int]]:
        payload = bytearray()
        payload.append(CMD_SAFETY_MANY)
        payload += struct.pack("<fi", float(s_max), int(max_checks))
        _pack_candidates(payload, candidates)
        response = self._request(payload)
        offset, count = _unpack_u32(response, 0)
        rows: list[list[float | int]] = []
        for _ in range(count):
            values = struct.unpack_from("<fffiiiffff", response, offset)
            offset += struct.calcsize("<fffiiiffff")
            rows.append(list(values))
        _ensure_consumed(response, offset)
        return rows

    def dr_summary_many(
        self,
        candidates: list[Candidate],
        start_weight: float,
        prune_weight: float,
    ) -> list[list[float | int]]:
        payload = bytearray()
        payload.append(CMD_DR_SUMMARY_MANY)
        payload += struct.pack("<ff", float(start_weight), float(prune_weight))
        _pack_candidates(payload, candidates)
        response = self._request(payload)
        offset, count = _unpack_u32(response, 0)
        rows: list[list[float | int]] = []
        for _ in range(count):
            values = struct.unpack_from("<fffqffff", response, offset)
            offset += struct.calcsize("<fffqffff")
            rows.append(list(values))
        _ensure_consumed(response, offset)
        return rows

    def close(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is None:
            return
        self._proc = None
        try:
            if proc.poll() is None:
                frame = _frame(bytes([CMD_CLOSE]))
                assert proc.stdin is not None
                proc.stdin.write(frame)
                proc.stdin.flush()
                _read_response(proc)
        except Exception:
            pass
        finally:
            try:
                proc.terminate()
            except Exception:
                pass

    def __del__(self) -> None:
        self.close()

    def _start_process(self) -> subprocess.Popen[bytes]:
        kwargs: dict[str, Any] = {}
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            kwargs["startupinfo"] = startupinfo
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        return subprocess.Popen(
            [str(self.helper_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.helper_path.parent),
            **kwargs,
        )

    def _configure(self, eval_config: dict[str, Any], threads: int) -> None:
        payload = bytearray()
        payload.append(CMD_CONFIGURE)
        for value in eval_config["fsrs6_weights"]:
            payload += struct.pack("<f", float(value))
        payload += struct.pack(
            "<iii",
            int(eval_config["days"]),
            int(eval_config["deck_size"]),
            int(eval_config["new_cards_per_day"]),
        )
        for key in (
            "initial_rating_prob",
            "initial_cost",
            "review_rating_prob_given_success",
            "review_cost",
        ):
            for value in eval_config[key]:
                payload += struct.pack("<f", float(value))
        payload += struct.pack("<I", max(0, int(threads)))
        response = self._request(payload)
        _, self.threads = _unpack_u32(response, 0)

    def _request(self, payload: bytes | bytearray) -> bytes:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            stderr = b""
            if proc is not None and proc.stderr is not None:
                stderr = proc.stderr.read()
            raise RustBackendError(
                "adr-simulator-helper is not running"
                + (f": {stderr.decode(errors='replace')}" if stderr else "")
            )
        assert proc.stdin is not None
        proc.stdin.write(_frame(bytes(payload)))
        proc.stdin.flush()
        return _read_response(proc)


def make_eval_lib(row: dict, config: SearchConfig) -> RustEvalLib:
    usage = row["button_usage"]
    return RustEvalLib(
        {
            "fsrs6_weights": [float(v) for v in row["fsrs6_weights"]],
            "days": int(config.days),
            "deck_size": int(config.deck_size),
            "new_cards_per_day": int(config.learn_limit),
            "initial_rating_prob": normalize(usage["first_rating_prob"]),
            "initial_cost": [float(v) for v in usage["learn_costs"]],
            "review_rating_prob_given_success": normalize(usage["review_rating_prob"]),
            "review_cost": [float(v) for v in usage["review_costs"]],
        },
        threads=config.threads,
    )


def evaluate_raw(
    eval_lib,
    candidates: list[Candidate],
    weight: float,
    seed: int,
) -> list[Point]:
    rows = eval_lib.evaluate_many(candidates, float(weight), int(seed))
    return [
        Point(
            flat=float(row[0]),
            s_multi=float(row[1]),
            d_multi=float(row[2]),
            total_average_memorized=float(row[3]),
            total_cost=float(row[4]),
            total_iters=int(row[5]),
            memorized_fraction=float(row[6]),
            memorized_cards=float(row[7]),
            memorized_per_minute=float(row[8]),
        )
        for row in rows
    ]


def evaluate_search(
    eval_lib,
    candidates: list[Candidate],
    weight: float,
    seed: int,
    config: SearchConfig,
) -> list[Point]:
    from .safety import attach_safety

    return attach_safety(
        eval_lib, evaluate_raw(eval_lib, candidates, weight, seed), config
    )


def evaluate_fixed_points(
    eval_lib,
    specs: list[tuple[float, Candidate]],
    weight: float,
    seed: int,
) -> list[tuple[float, Point]]:
    raw = evaluate_raw(eval_lib, [candidate for _, candidate in specs], weight, seed)
    return [
        (
            dr,
            replace(
                point, dr_samples=1, dr_p10=dr, dr_mean=dr, dr_p90=dr, dr_spread=0.0
            ),
        )
        for (dr, _), point in zip(specs, raw, strict=True)
    ]


def attach_dr_summary(
    eval_lib,
    points: list[Point],
    weight: float,
    config: SearchConfig,
) -> list[Point]:
    if not points:
        return points
    unique = {point.key: (point.flat, point.s_multi, point.d_multi) for point in points}
    rows = eval_lib.dr_summary_many(
        list(unique.values()),
        float(weight),
        float(config.dr_prune_weight),
    )
    summaries = {
        key_of((flat, s_multi, d_multi)): row
        for row in rows
        for flat, s_multi, d_multi in [row[:3]]
    }
    return [
        replace(
            point,
            dr_samples=int(summaries[point.key][3]),
            dr_p10=float(summaries[point.key][4]),
            dr_mean=float(summaries[point.key][5]),
            dr_p90=float(summaries[point.key][6]),
            dr_spread=float(summaries[point.key][7]),
        )
        for point in points
    ]


def _pack_candidates(payload: bytearray, candidates: list[Candidate]) -> None:
    payload += struct.pack("<I", len(candidates))
    for flat, s_multi, d_multi in candidates:
        payload += struct.pack("<fff", float(flat), float(s_multi), float(d_multi))


def _frame(payload: bytes) -> bytes:
    return struct.pack("<I", len(payload)) + payload


def _read_response(proc: subprocess.Popen[bytes]) -> bytes:
    assert proc.stdout is not None
    length_raw = _read_exact(proc.stdout, 4)
    length = struct.unpack("<I", length_raw)[0]
    frame = _read_exact(proc.stdout, length)
    if not frame:
        raise RustBackendError("adr-simulator-helper returned an empty frame")
    status = frame[0]
    payload = frame[1:]
    if status == 0:
        return payload
    if status == 1:
        if len(payload) < 4:
            raise RustBackendError("adr-simulator-helper returned a malformed error")
        error_len = struct.unpack_from("<I", payload, 0)[0]
        error = payload[4 : 4 + error_len].decode("utf-8", errors="replace")
        raise RustBackendError(error)
    raise RustBackendError(f"adr-simulator-helper returned unknown status {status}")


def _read_exact(stream, length: int) -> bytes:
    chunks = []
    remaining = length
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise RustBackendError("adr-simulator-helper closed its output pipe")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _unpack_u32(payload: bytes, offset: int) -> tuple[int, int]:
    return offset + 4, struct.unpack_from("<I", payload, offset)[0]


def _ensure_consumed(payload: bytes, offset: int) -> None:
    if offset != len(payload):
        raise RustBackendError(
            f"adr-simulator-helper response had {len(payload) - offset} trailing bytes"
        )
