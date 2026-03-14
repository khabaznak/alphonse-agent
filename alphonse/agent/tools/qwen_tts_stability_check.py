from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from typing import Any

from alphonse.agent.tools.local_audio_output import LocalAudioOutputSpeakTool

_REFERENCE_LINES = [
    "Hola Alex, soy Alphonse y estoy listo para ayudarte.",
    "Hello Alex, this is Alphonse. I am ready to help.",
    "Hoy avanzamos con calma y claridad.",
    "Today we move forward calmly and clearly.",
    "Estoy aqui para ayudarte con lo que sigue.",
    "I am here to help with what comes next.",
    "Hagamos esto paso a paso.",
    "Let's do this one step at a time.",
    "Todo va bien, continuemos.",
    "Everything is going well. Let's continue.",
]

_TIER_PROFILES: dict[str, dict[str, str]] = {
    "stable": {
        "ALPHONSE_TTS_BACKEND": "qwen",
        "ALPHONSE_QWEN_TTS_MODEL": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "ALPHONSE_QWEN_TTS_DEVICE_MAP": "cpu",
        "ALPHONSE_QWEN_TTS_DTYPE": "float32",
    },
    "balanced": {
        "ALPHONSE_TTS_BACKEND": "qwen",
        "ALPHONSE_QWEN_TTS_MODEL": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "ALPHONSE_QWEN_TTS_DEVICE_MAP": "auto",
        "ALPHONSE_QWEN_TTS_DTYPE": "float16",
    },
    "strict": {
        "ALPHONSE_TTS_BACKEND": "qwen",
        "ALPHONSE_QWEN_TTS_MODEL": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "ALPHONSE_QWEN_TTS_DEVICE_MAP": "auto",
        "ALPHONSE_QWEN_TTS_DTYPE": "float16",
        "ALPHONSE_QWEN_TTS_INSTRUCT": "Speak calmly, warm, concise, low excitement, clear articulation.",
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m alphonse.agent.tools.qwen_tts_stability_check")
    parser.add_argument("--tier", choices=sorted(_TIER_PROFILES.keys()), default="balanced")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--speaker", default="Ryan")
    parser.add_argument("--language", default="Auto")
    parser.add_argument("--p95-budget-seconds", type=float, default=10.0)
    parser.add_argument("--blocking", action="store_true", default=False)
    return parser.parse_args()


def _apply_profile(*, tier: str, speaker: str, language: str) -> None:
    profile = _TIER_PROFILES[tier]
    for key, value in profile.items():
        os.environ[key] = value
    os.environ["ALPHONSE_QWEN_TTS_SPEAKER"] = str(speaker or "Ryan").strip() or "Ryan"
    os.environ["ALPHONSE_QWEN_TTS_LANGUAGE"] = str(language or "Auto").strip() or "Auto"


def _p95(samples: list[float]) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    idx = max(0, min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1)))))
    return ordered[idx]


def main() -> int:
    args = _parse_args()
    runs = max(1, int(args.runs or 10))
    _apply_profile(tier=args.tier, speaker=args.speaker, language=args.language)

    tool = LocalAudioOutputSpeakTool()
    latencies: list[float] = []
    failures: list[dict[str, Any]] = []

    for i in range(runs):
        text = _REFERENCE_LINES[i % len(_REFERENCE_LINES)]
        started = time.perf_counter()
        result = tool.execute(text=text, blocking=bool(args.blocking))
        elapsed = time.perf_counter() - started
        latencies.append(elapsed)
        exc = result.get("exception") if isinstance(result, dict) else None
        if isinstance(exc, dict):
            failures.append(
                {
                    "index": i + 1,
                    "code": str(exc.get("code") or ""),
                    "message": str(exc.get("message") or ""),
                }
            )

    median_latency = statistics.median(latencies) if latencies else 0.0
    p95_latency = _p95(latencies)
    passed = (len(failures) == 0) and (p95_latency <= float(args.p95_budget_seconds or 10.0))
    report = {
        "tier": args.tier,
        "runs": runs,
        "speaker": os.getenv("ALPHONSE_QWEN_TTS_SPEAKER"),
        "language": os.getenv("ALPHONSE_QWEN_TTS_LANGUAGE"),
        "blocking": bool(args.blocking),
        "median_seconds": round(float(median_latency), 3),
        "p95_seconds": round(float(p95_latency), 3),
        "p95_budget_seconds": float(args.p95_budget_seconds or 10.0),
        "successes": runs - len(failures),
        "failures": len(failures),
        "passed": passed,
        "errors": failures,
    }
    print(json.dumps(report, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
