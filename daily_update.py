# daily_update.py
from __future__ import annotations

import sys
import os
from pathlib import Path
from datetime import datetime
import subprocess

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "update.log"
MAX_LOG_BYTES = 2 * 1024 * 1024  # 2MB

PY = sys.executable  # venv python nếu bạn chạy: .\.venv\Scripts\python.exe daily_update.py

def rotate_log_if_needed():
    try:
        if LOG.exists() and LOG.stat().st_size > MAX_LOG_BYTES:
            bak = LOG.with_suffix(".log.1")
            try:
                bak.unlink()
            except Exception:
                pass
            LOG.replace(bak)
    except Exception:
        pass

def write_line(s: str):
    s = str(s).rstrip("\n")
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {s}\n"
    # In console
    try:
        print(line, end="")
    except Exception:
        print(line.encode("ascii", "replace").decode("ascii"), end="")
    # Ghi log
    try:
        with LOG.open("a", encoding="utf-8", errors="replace") as f:
            f.write(line)
    except Exception:
        pass

def run_step(name: str, cmd: list[str]) -> int:
    write_line(f"[RUN] {name} …")
    write_line(f"CMD: {' '.join(cmd)}")

    # >>> BUỘC UTF-8 CHO TIẾN TRÌNH CON <<<
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,  # <<< QUAN TRỌNG
    )
    if proc.stdout:
        write_line("--- STDOUT ---")
        write_line(proc.stdout)
    if proc.stderr:
        write_line("--- STDERR ---")
        write_line(proc.stderr)
    write_line(f"[RET] {name} = {proc.returncode}\n")
    return proc.returncode

def main() -> int:
    rotate_log_if_needed()

    ret_all = 0
    fails = []

    r = run_step("etl_itinerary", [PY, "-m", "src.etl", str(ROOT / "data" / "itinerary_dataset.csv")])
    if r != 0: ret_all, fails = 1, fails + ["etl_itinerary"]

    r = run_step("build_tfidf", [PY, "-m", "src.build_index"])
    if r != 0: ret_all, fails = 1, fails + ["build_tfidf"]

    r = run_step("build_nn", [PY, str(ROOT / "build_nn.py")])
    if r != 0: ret_all, fails = 1, fails + ["build_nn"]

    r = run_step("etl_qna", [PY, "-m", "src.etl_qna", str(ROOT / "data" / "qna_places.csv")])
    if r != 0: ret_all, fails = 1, fails + ["etl_qna"]

    r = run_step("build_qna_nn", [PY, str(ROOT / "build_qna_nn.py")])
    if r != 0: ret_all, fails = 1, fails + ["build_qna_nn"]

    if ret_all == 0:
        write_line("Pipeline OK")
    else:
        write_line(f"Pipeline FAILED — xem chi tiết: {LOG}")

    return ret_all

if __name__ == "__main__":
    sys.exit(main())
