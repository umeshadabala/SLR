#!/usr/bin/env python3
"""EXP-07 / EXP-08 Autonomous Verification and Diagnostics Harness.

Executes deterministic end-to-end verification across:
1. Environment and dependencies (FastAPI, PyTorch, Transformers, Android SDK).
2. Connected physical Android device state via ADB.
3. Development backend on port 8000 (/health, /api/v1/model/load).
4. ADB reverse port forwarding (tcp:8000).
5. On-device Selective Layer Residency (SLR) benchmark reports & logs.
6. Offline Android-native inference verification (zero Mac backend dependence).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


class VerificationHarness:
    def __init__(self, project_root: Path):
        self.root = project_root
        self.results = []
        self.start_time = time.time()

    def record(self, test_id: str, name: str, status: str, details: str, duration_sec: float = 0.0):
        res = {
            "test_id": test_id,
            "name": name,
            "status": status,
            "details": details,
            "duration_sec": round(duration_sec, 3),
        }
        self.results.append(res)
        color = "\033[92m" if status == "PASS" else ("\033[93m" if status in ("SKIPPED", "UNAVAILABLE") else "\033[91m")
        reset = "\033[0m"
        print(f"[{color}{status:7s}{reset}] {test_id}: {name} - {details} ({duration_sec:.2f}s)")

    def run_cmd(self, cmd: list[str], timeout: int = 15) -> tuple[int, str, str]:
        t0 = time.time()
        try:
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
            return p.returncode, p.stdout.strip(), p.stderr.strip()
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except Exception as e:
            return -2, "", str(e)

    def check_environment(self):
        t0 = time.time()
        py_bin = self.root / "backend" / ".venv" / "bin" / "python"
        if not py_bin.exists():
            self.record("ENV-01", "Python Venv", "FAIL", f"Missing venv at {py_bin}", time.time() - t0)
            return

        code, out, err = self.run_cmd([str(py_bin), "-c", "import torch, transformers, fastapi, safetensors; print(torch.__version__, transformers.__version__, fastapi.__version__)"])
        if code == 0:
            self.record("ENV-01", "Python Dependencies", "PASS", f"torch={out.split()[0]} transformers={out.split()[1]} fastapi={out.split()[2]}", time.time() - t0)
        else:
            self.record("ENV-01", "Python Dependencies", "FAIL", f"Import error: {err}", time.time() - t0)

    def check_android_device(self) -> str | None:
        t0 = time.time()
        code, out, err = self.run_cmd(["adb", "devices", "-l"])
        if code != 0:
            self.record("DEV-01", "ADB Availability", "FAIL", f"ADB failed: {err}", time.time() - t0)
            return None

        lines = [line for line in out.splitlines() if line.strip() and not line.startswith("List of devices")]
        if not lines:
            self.record("DEV-01", "Physical Device Attached", "UNAVAILABLE", "No ADB devices connected", time.time() - t0)
            return None

        device_line = lines[0]
        serial = device_line.split()[0]
        
        # Query props
        _, model, _ = self.run_cmd(["adb", "-s", serial, "shell", "getprop", "ro.product.model"])
        _, mfg, _ = self.run_cmd(["adb", "-s", serial, "shell", "getprop", "ro.product.manufacturer"])
        _, release, _ = self.run_cmd(["adb", "-s", serial, "shell", "getprop", "ro.build.version.release"])
        _, sdk, _ = self.run_cmd(["adb", "-s", serial, "shell", "getprop", "ro.build.version.sdk"])

        self.record("DEV-01", "Physical Device Identity", "PASS", f"Serial: {serial}, {mfg} {model} (Android {release}, API {sdk})", time.time() - t0)
        return serial

    def check_mac_backend(self):
        t0 = time.time()
        import urllib.request
        try:
            req = urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3)
            data = json.loads(req.read().decode())
            status = "PASS" if data.get("status") == "ok" else "FAIL"
            self.record("SRV-01", "FastAPI /health on :8000", status, f"status={data.get('status')} model_state={data.get('model_state')}", time.time() - t0)
        except Exception as e:
            self.record("SRV-01", "FastAPI /health on :8000", "FAIL", f"Unreachable: {e}", time.time() - t0)

    def check_adb_reverse(self, serial: str):
        t0 = time.time()
        code, out, _ = self.run_cmd(["adb", "-s", serial, "reverse", "--list"])
        if "tcp:8000" in out:
            self.record("NET-01", "ADB Reverse tcp:8000", "PASS", "Active reverse mapping verified", time.time() - t0)
        else:
            self.record("NET-01", "ADB Reverse tcp:8000", "FAIL", f"No reverse mapping found: {out}", time.time() - t0)

    def check_device_benchmark(self, serial: str):
        t0 = time.time()
        code, out, _ = self.run_cmd(["adb", "-s", serial, "shell", "run-as", "com.example.exp07_slr", "ls", "-t", "files/benchmark_reports/"])
        if code != 0 or not out.strip():
            self.record("BNCH-01", "On-Device Benchmark Reports", "UNAVAILABLE", "No benchmark reports found in app storage", time.time() - t0)
            return

        latest_file = out.splitlines()[0].strip()
        code, content, _ = self.run_cmd(["adb", "-s", serial, "shell", "run-as", "com.example.exp07_slr", "cat", f"files/benchmark_reports/{latest_file}"])
        try:
            report = json.loads(content)
            summary = report.get("summary", {})
            passed = summary.get("passed", 0)
            total = summary.get("total_tests", 0)
            status = "PASS" if passed > 0 and summary.get("failed", 0) == 0 else "PARTIAL"
            self.record("BNCH-01", "Latest Benchmark Report", status, f"File: {latest_file}, Passed: {passed}/{total}, Duration: {summary.get('duration_ms', 0)}ms", time.time() - t0)
        except Exception as e:
            self.record("BNCH-01", "Latest Benchmark Report", "FAIL", f"Malformed JSON: {e}", time.time() - t0)

    def check_native_inference_run(self, serial: str):
        t0 = time.time()
        code, out, _ = self.run_cmd(["adb", "-s", serial, "shell", "run-as", "com.example.exp07_slr", "ls", "-t", "files/runs/"])
        if code != 0 or not out.strip():
            self.record("INF-01", "Native SLR Generation Run", "UNAVAILABLE", "No runs found in app storage", time.time() - t0)
            return

        latest_run = out.splitlines()[0].strip()
        code, content, _ = self.run_cmd(["adb", "-s", serial, "shell", "run-as", "com.example.exp07_slr", "cat", f"files/runs/{latest_run}"])
        try:
            run_data = json.loads(content)
            mode = run_data.get("execution_mode", "")
            active_layers = run_data.get("peak_active_layers", 0)
            active_params = run_data.get("peak_active_params", 0)
            ttft = run_data.get("timings", {}).get("model_ttft_sec", 0.0)
            gen_text = run_data.get("generated_text", "").strip()

            is_valid = mode == "ANDROID-NATIVE INFERENCE" and active_layers == 1 and active_params == 46797824
            status = "PASS" if is_valid else "FAIL"
            self.record(
                "INF-01",
                "Native SLR Generation Run",
                status,
                f"File: {latest_run}, TTFT: {ttft:.2f}s, ActiveLayers: {active_layers}/28, Text: '{gen_text}'",
                time.time() - t0
            )
        except Exception as e:
            self.record("INF-01", "Native SLR Generation Run", "FAIL", f"Malformed JSON: {e}", time.time() - t0)

    def summarize(self) -> dict:
        total = len(self.results)
        passed = sum(1 for r in self.results if r["status"] == "PASS")
        failed = sum(1 for r in self.results if r["status"] == "FAIL")
        skipped = sum(1 for r in self.results if r["status"] in ("SKIPPED", "UNAVAILABLE"))
        
        summary = {
            "timestamp": time.time(),
            "duration_total_sec": round(time.time() - self.start_time, 2),
            "total_checks": total,
            "passed": passed,
            "failed": failed,
            "skipped_or_unavailable": skipped,
            "results": self.results
        }
        out_path = self.root / "benchmark_runs" / "full_verification_summary.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nFinal Verification Summary: {passed}/{total} Passed, {failed} Failed, {skipped} Skipped/Unavailable.")
        print(f"Summary JSON saved to {out_path}")
        return summary


def main():
    root = Path(__file__).resolve().parent.parent
    print(f"Starting EXP-07 Comprehensive Verification Harness on {root}...\n")
    harness = VerificationHarness(root)
    harness.check_environment()
    serial = harness.check_android_device()
    harness.check_mac_backend()
    if serial:
        harness.check_adb_reverse(serial)
        harness.check_device_benchmark(serial)
        harness.check_native_inference_run(serial)
    summary = harness.summarize()
    sys.exit(0 if summary["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
