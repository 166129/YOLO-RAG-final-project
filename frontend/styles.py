"""Presentation layer: page CSS and the citation card markup.

Kept apart from app.py so the app file stays readable as flow-of-control.
"""
from __future__ import annotations

import html

CSS = """
<style>
/* --- strip the Streamlit dev chrome: this is a finished app --- */
#MainMenu, header [data-testid="stToolbar"], footer {visibility: hidden;}
header {height: 0rem;}
.block-container {padding-top: 2.2rem; padding-bottom: 6rem; max-width: 52rem;}

/* --- brand header --- */
.rr-header {display: flex; align-items: center; gap: .75rem; margin-bottom: .15rem;}
.rr-title {font-size: 1.85rem; font-weight: 700; letter-spacing: -.02em; margin: 0;}
.rr-sub {color: #8B97A8; font-size: .93rem; margin: 0 0 1.1rem 0;}

/* --- citation cards --- */
.rr-cite {
  border: 1px solid #2A3240;
  border-left: 3px solid #F5B32E;
  border-radius: 8px;
  padding: .7rem .9rem;
  margin: .45rem 0;
  background: #131821;
}
.rr-cite-head {
  display: flex; align-items: center; gap: .5rem;
  flex-wrap: wrap; margin-bottom: .4rem;
}
.rr-badge {
  background: #F5B32E; color: #1A1200;
  font-size: .68rem; font-weight: 700; letter-spacing: .04em;
  text-transform: uppercase; padding: .12rem .45rem; border-radius: 4px;
}
.rr-page {color: #C6D0DE; font-size: .82rem; font-weight: 600;}
.rr-file {color: #6E7A8C; font-size: .75rem; font-family: ui-monospace, monospace;}
.rr-dist {margin-left: auto; color: #6E7A8C; font-size: .72rem;}
.rr-quote {
  color: #C8D3E0; font-size: .88rem; line-height: 1.55;
  border-left: 2px solid #2A3240; padding-left: .7rem; margin: 0;
}

/* --- detection chips --- */
.rr-chip {
  display: inline-block; background: #1E2530; border: 1px solid #2A3240;
  color: #E6EDF3; border-radius: 999px; padding: .15rem .6rem;
  font-size: .76rem; margin-right: .35rem;
}
.rr-chip b {color: #F5B32E;}

/* --- tighten chat bubbles --- */
[data-testid="stChatMessage"] {padding: .4rem .2rem;}
</style>
"""


def header(title: str, subtitle: str) -> str:
    return (
        f'<div class="rr-header"><span style="font-size:1.7rem">🚦</span>'
        f'<p class="rr-title">{html.escape(title)}</p></div>'
        f'<p class="rr-sub">{html.escape(subtitle)}</p>'
    )


def citation_card(c: dict) -> str:
    """One cited passage: which handbook, which page, and the text itself."""
    state = html.escape(str(c.get("state", "")))
    page = html.escape(str(c.get("page", "")))
    handbook = html.escape(str(c.get("handbook", "")))
    snippet = html.escape(str(c.get("snippet", "")))
    distance = c.get("distance")
    dist_html = f'<span class="rr-dist">distance {distance:.3f}</span>' if isinstance(
        distance, (int, float)
    ) else ""
    return (
        '<div class="rr-cite">'
        '<div class="rr-cite-head">'
        f'<span class="rr-badge">{state}</span>'
        f'<span class="rr-page">page {page}</span>'
        f'<span class="rr-file">{handbook}</span>'
        f"{dist_html}"
        "</div>"
        f'<p class="rr-quote">“{snippet}”</p>'
        "</div>"
    )


def detection_chips(detections: list[dict]) -> str:
    return " ".join(
        f'<span class="rr-chip"><b>{html.escape(str(d["label"]))}</b> '
        f'{d["confidence"]:.0%}</span>'
        for d in detections
    )
