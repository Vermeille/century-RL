"""Arena events and the Trackio-facing telemetry adapter."""

from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable


@dataclass(frozen=True)
class ArenaEvent:
    key: str
    kind: str
    title: str
    text: str
    level: str
    step: int
    created_at: float


class TrackioTelemetry:
    """Trackio-specific media and alerts behind a small arena-facing interface."""

    def __init__(self, run, *, html_factory, alert_levels):
        self.run = run
        self.html_factory = html_factory
        self.alert_levels = alert_levels

    def log(self, metrics, *, step: int):
        self.run.log(metrics, step=step)

    def html(self, content: str):
        return self.html_factory(content)

    def alert(self, event: ArenaEvent):
        self.run.alert(
            title=event.title,
            text=event.text,
            level=getattr(self.alert_levels, event.level.upper()),
            step=event.step,
        )


def render_event_log(events: Iterable[ArenaEvent]) -> str:
    events = list(events)
    counts = {
        kind: sum(event.kind == kind for event in events)
        for kind in ("new_best", "regression", "cycle")
    }
    rendered = []
    for event in reversed(events):
        timestamp = datetime.fromtimestamp(
            event.created_at, tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S UTC")
        rendered.append(
            '<article class="event '
            f'{html.escape(event.level)}">'
            '<div class="event-head">'
            f'<span class="badge">{html.escape(event.kind.replace("_", " "))}</span>'
            f'<strong>{html.escape(event.title)}</strong>'
            f'<span class="step">step {event.step}</span>'
            "</div>"
            f'<p>{html.escape(event.text)}</p>'
            f'<time>{timestamp}</time>'
            "</article>"
        )
    body = "".join(rendered) or '<p class="empty">No arena events yet.</p>'
    return f"""
<style>
body {{ font-family: system-ui, sans-serif; margin: 0; color: #172033; }}
.summary {{ display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; }}
.summary span {{ border-radius: 14px; padding: 5px 9px; background: #edf1f8;
                 font-size: 12px; font-variant-numeric: tabular-nums; }}
.events {{ display: grid; gap: 10px; }}
.event {{ border-left: 5px solid #5b7cfa; border-radius: 6px;
          padding: 10px 12px; background: #f5f7ff; }}
.event.warn {{ border-color: #d97706; background: #fff8eb; }}
.event.error {{ border-color: #dc2626; background: #fff1f2; }}
.event-head {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
.badge {{ text-transform: uppercase; font-size: 11px; letter-spacing: .04em;
          padding: 2px 6px; border-radius: 10px; background: #dde5ff; }}
.warn .badge {{ background: #fde7bd; }}
.step {{ margin-left: auto; color: #596174; font-variant-numeric: tabular-nums; }}
p {{ margin: 7px 0 4px; }}
time {{ color: #737b8c; font-size: 12px; }}
.empty {{ color: #737b8c; }}
</style>
<div class="summary">
  <span>{counts['new_best']} new best</span>
  <span>{counts['regression']} regressions</span>
  <span>{counts['cycle']} cycles</span>
</div>
<section class="events">{body}</section>
"""
