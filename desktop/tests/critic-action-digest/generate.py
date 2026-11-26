"""Generate disposable data with the production A2UI projectors."""
import json
from pathlib import Path

from alphonse.agent_v2.interfaces.private_provider_a2ui import calendar_agenda_surface, gmail_summary_surface


LABEL = "Gmail label Family/Spring workshop"
ZONE = "America/Mexico_City"
BODIES = {
    2: ("Original registration deadline", "Registration closes November 29."),
    7: ("Packing [updated] <latest>", "Bring a plain apron. **Punctuation** stays literal here."),
    8: ("Registration deadline extended", "Registration now closes December 3 at 17:00 Mexico City time. This replaces the November 29 deadline."),
    11: ("Check-in instructions", "Please bring proof of registration. The office opens at 09:00. The organizer did not specify a timezone."),
}
MAIL = {
    "messages": [
        {"subject": BODIES.get(index, ("Workshop bulletin", ""))[0],
         "excerpt": BODIES.get(index, ("", ""))[1],
         "sender": "Workshop office <office@example.invalid>",
         "received_at": "2026-11-25T18:00:00+00:00"}
        for index in range(1, 13)
    ],
    "status": "partial",
    "lookback_days": 30,
}
SYNTHESIS = json.dumps({
    "summary": "### What changed\n\n- **Nov 29** was superseded. [2](#source-2)\n- The office time has *no stated time zone*. [11](#source-11)",
    "actions": [
        {"date": None, "when": "", "title": "Pack a plain apron", "detail": "", "sources": [7]},
        {"date": None, "when": "Dec 3\n17:00 · Mexico City", "title": "Complete workshop registration",
         "detail": "Use the **revised deadline**.", "sources": [8, 2]},
        {"date": None, "when": "", "title": "Bring proof of registration",
         "detail": "Office opens at **09:00**. Timezone unspecified.", "sources": [11]},
    ],
}, ensure_ascii=False)
CALENDAR = {
    "window": ["2026-12-03T00:00:00-06:00", "2026-12-05T00:00:00-06:00"],
    "timezone": ZONE,
    "status": "complete",
    "events": [
        {"title": "Robotics check-in", "start": "2026-12-03T14:30:00+00:00",
         "end": "2026-12-03T15:00:00+00:00", "all_day": False, "end_unspecified": False},
        {"title": "Workshop setup", "start": "2026-12-03T19:00:00+00:00",
         "end": "2026-12-03T20:00:00+00:00", "all_day": False, "end_unspecified": True},
        {"title": "Book fair [extended]", "start": "2026-12-04", "end": "2026-12-06",
         "all_day": True, "end_unspecified": False},
        {"title": "Equipment delivery", "start": "2026-12-02T23:30:00-06:00",
         "end": "2026-12-03T00:20:00-06:00", "all_day": False, "end_unspecified": False},
    ],
}

if __name__ == "__main__":
    fixtures = {
        "review": {
            "gmail": gmail_summary_surface("fixture-gmail", MAIL, source=LABEL, summary=SYNTHESIS),
            "calendar": calendar_agenda_surface("fixture-calendar", CALENDAR, timezone_name=ZONE),
        },
        "empty": {
            "gmail": gmail_summary_surface("fixture-gmail-empty", {**MAIL, "messages": [], "status": "complete"}, source=LABEL, summary=""),
            "calendar": calendar_agenda_surface("fixture-calendar-empty", {**CALENDAR, "events": []}, timezone_name=ZONE),
        },
    }
    Path(__file__).with_name("data.json").write_text(json.dumps(fixtures, indent=2, ensure_ascii=False) + "\n")
