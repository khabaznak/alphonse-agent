#!/usr/bin/env python3
from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
import sys
import argparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path


def main() -> None:
    args = _parse_args()
    db_path = Path(args.db).expanduser().resolve() if args.db else resolve_nervous_system_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT created_at, user_text, reason, metadata
        FROM capability_gaps
        WHERE status = 'open'
        ORDER BY datetime(created_at) DESC
        LIMIT 500
        """
    ).fetchall()

    clusters: dict[str, dict[str, object]] = defaultdict(
        lambda: {"count": 0, "aliases": set(), "examples": []}
    )

    for row in rows:
        metadata = _parse_json(row["metadata"])
        proposed = str(metadata.get("proposed_intent") or "").strip().lower()
        if not proposed:
            key = str(row["reason"] or "unknown").strip().lower()
        else:
            key = proposed
        cluster = clusters[key]
        cluster["count"] = int(cluster["count"]) + 1
        aliases = metadata.get("proposed_intent_aliases")
        if isinstance(aliases, list):
            for alias in aliases:
                normalized = str(alias or "").strip().lower()
                if normalized:
                    cast_set = cluster["aliases"]
                    if isinstance(cast_set, set):
                        cast_set.add(normalized)
        examples = cluster["examples"]
        text = str(row["user_text"] or "").strip()
        if isinstance(examples, list) and text and text not in examples and len(examples) < 3:
            examples.append(text)

    print(f"db={db_path}")
    print(f"open_gaps={len(rows)}")
    print()
    for key, cluster in sorted(clusters.items(), key=lambda kv: int(kv[1]["count"]), reverse=True):
        print(f"- candidate={key} count={cluster['count']}")
        aliases = sorted(cluster["aliases"]) if isinstance(cluster["aliases"], set) else []
        if aliases:
            print(f"  aliases={aliases}")
        examples = cluster["examples"] if isinstance(cluster["examples"], list) else []
        for example in examples:
            print(f"  ex={example}")


def _parse_json(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show open capability-gap intent candidates.")
    parser.add_argument(
        "--db",
        help="Optional absolute or relative path to nerve-db sqlite file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
