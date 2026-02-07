from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def test_no_hardcoded_user_response_literals_in_runtime_paths() -> None:
    source_root = ROOT / "alphonse" / "agent"
    patterns = [
        re.compile(r"_message_result\(\s*['\"]"),
        re.compile(r"reply_text\s*=\s*['\"]"),
    ]
    violations: list[str] = []

    for path in source_root.rglob("*.py"):
        if "safe_fallbacks.py" in str(path):
            continue
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        for idx, line in enumerate(lines, start=1):
            if any(pattern.search(line) for pattern in patterns):
                violations.append(f"{path.relative_to(ROOT)}:{idx}:{line.strip()}")

    assert violations == [], "Hardcoded runtime user-response literals detected:\n" + "\n".join(violations)
