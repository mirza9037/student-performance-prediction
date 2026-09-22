"""Launch and stop a real local Streamlit server, retaining verification evidence."""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Check readiness/HTML and always stop only this script's child server."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    log_path = report_dir / "streamlit_smoke.log"
    result = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "port": port}
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py",
                                    "--server.headless", "true", "--server.address", "127.0.0.1",
                                    "--server.port", str(port), "--browser.gatherUsageStats", "false"],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Streamlit exited early; inspect {log_path}")
                try:
                    with urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=2) as response:
                        health = response.read().decode()
                        if response.status == 200 and health.strip() == "ok":
                            break
                except (URLError, TimeoutError):
                    pass
                time.sleep(.5)
            else:
                raise RuntimeError("Streamlit did not become healthy within 60 seconds.")
            with urlopen(f"http://127.0.0.1:{port}/", timeout=5) as response:
                html = response.read().decode()
                assert response.status == 200 and "<html" in html.lower()
            result.update({"health": health, "html_served": True})
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            result["server_stopped"] = process.poll() is not None
    log_text = log_path.read_text(encoding="utf-8")
    if "Traceback (most recent call last)" in log_text:
        raise RuntimeError(f"Server exception found; inspect {log_path}")
    (report_dir / "server_smoke.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
