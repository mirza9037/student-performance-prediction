"""Probability Atlas presentation helpers for the Streamlit application."""
from __future__ import annotations

from html import escape
import streamlit as st
import streamlit.components.v1 as components


COLORS = {
    "cobalt": "#2056C7",
    "teal": "#087F75",
    "amber": "#A85E00",
    "red": "#B42318",
    "ink": "#15181D",
    "muted": "#5A6572",
    "rule": "#DDE2E8",
    "paper": "#FFFFFF",
    "ground": "#F4F6F8",
}


def inject_theme() -> None:
    """Apply the shared Probability Atlas design system."""
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;600;700&family=Source+Sans+3:wght@400;500;600&display=swap');

:root {
  --atlas-ground: #F4F6F8;
  --atlas-paper: #FFFFFF;
  --atlas-ink: #15181D;
  --atlas-muted: #5A6572;
  --atlas-rule: #DDE2E8;
  --atlas-cobalt: #2056C7;
  --atlas-cobalt-soft: #EAF0FF;
  --atlas-teal: #087F75;
  --atlas-teal-soft: #E7F5F2;
  --atlas-amber: #A85E00;
  --atlas-amber-soft: #FFF3DB;
  --atlas-red: #B42318;
  --atlas-red-soft: #FDECEA;
  --atlas-radius: 12px;
}

html { scroll-behavior: auto; }
body, [class*="css"] {
  font-family: "Source Sans 3", sans-serif;
  color: var(--atlas-ink);
}
.stApp { background: var(--atlas-ground); color: var(--atlas-ink); color-scheme: light; }
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] li,
.stApp label,
.stApp label p { color: var(--atlas-ink) !important; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { opacity: .45; }
.block-container {
  max-width: 1240px;
  padding: 2.25rem 2.8rem 5rem;
}
h1, h2, h3, .atlas-brand__name {
  font-family: "Manrope", sans-serif;
  color: var(--atlas-ink) !important;
  letter-spacing: -.028em;
}
h1 { font-size: clamp(2rem, 3vw, 2.8rem); line-height: 1.08; font-weight: 650; }
h2 { font-size: clamp(1.45rem, 2vw, 2rem); line-height: 1.18; font-weight: 620; }
h3 { font-size: 1.18rem; line-height: 1.3; font-weight: 620; }
p, li { line-height: 1.6; }
a { color: var(--atlas-cobalt); text-decoration-thickness: 1px; text-underline-offset: 3px; }
a:hover { color: #17449F; text-decoration-thickness: 2px; }
::selection { background: #C8D8FF; color: var(--atlas-ink); }
*:focus-visible { outline: 2px solid var(--atlas-cobalt) !important; outline-offset: 2px !important; }

section[data-testid="stSidebar"] {
  width: 272px !important;
  background: var(--atlas-paper);
  border-right: 1px solid var(--atlas-rule);
}
section[data-testid="stSidebar"] > div { padding-top: 1.35rem; }
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-inline: .85rem; }
.atlas-brand { padding: .55rem .6rem 1.35rem; border-bottom: 1px solid var(--atlas-rule); margin-bottom: 1rem; }
.atlas-brand__name { display: block; font-size: 1.28rem; font-weight: 700; }
.atlas-brand__tag { display: block; margin-top: .2rem; color: var(--atlas-muted); font-size: .78rem; }
[data-testid="stSidebar"] [data-testid="stRadio"] > label,
[data-testid="stSidebar"] [data-testid="stRadio"] > div > label:first-child {
  color: var(--atlas-muted); font-size: .73rem; font-weight: 600; letter-spacing: .01em;
}
[data-testid="stSidebar"] [role="radiogroup"] { gap: .18rem; }
[data-testid="stSidebar"] [role="radiogroup"] label {
  min-height: 42px;
  padding: .55rem .7rem;
  border-radius: 8px;
  transition: background-color 150ms ease, color 150ms ease;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: #F1F3F6; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background: var(--atlas-cobalt-soft);
  color: #173E8F;
  font-weight: 600;
}
[data-testid="stSidebar"] [role="radiogroup"] label [data-testid="stMarkdownContainer"] p { font-size: .94rem; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) [data-testid="stMarkdownContainer"] p {
  color: #173E8F !important;
}
[data-testid="stSidebar"] hr { border-color: var(--atlas-rule); }
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] .stCaption p { color: var(--atlas-muted) !important; }
.atlas-model-card {
  margin: 1.15rem .55rem 0; padding: .95rem; border: 1px solid var(--atlas-rule);
  border-radius: 10px; background: #F8FAFC;
}
.atlas-model-card span { display: block; color: var(--atlas-muted); font-size: .72rem; font-weight: 600; letter-spacing: .04em; text-transform: uppercase; }
.atlas-model-card strong { display: block; margin-top: .35rem; color: var(--atlas-ink); font-size: .9rem; line-height: 1.35; }
.atlas-model-card small { display: block; margin-top: .5rem; color: var(--atlas-teal); font-size: .78rem; line-height: 1.4; }

.atlas-page-head { margin: .1rem 0 1.55rem; max-width: 760px; }
.atlas-page-head h1 { margin: 0; }
.atlas-page-head p { margin: .55rem 0 0; color: var(--atlas-muted) !important; font-size: 1rem; }
.atlas-context {
  display: flex; flex-wrap: wrap; gap: .55rem 1rem; align-items: center;
  padding: .78rem 1rem; margin: -.35rem 0 1.4rem;
  background: #EDF1F5; border: 1px solid var(--atlas-rule); border-radius: 9px;
  color: #46515E; font-size: .86rem;
}
.atlas-context strong { color: var(--atlas-ink); font-weight: 600; }
.atlas-workflow {
  display: grid; grid-template-columns: repeat(3,minmax(0,1fr));
  margin: 0 0 1.75rem; border-block: 1px solid var(--atlas-rule); background: var(--atlas-paper);
}
.atlas-workflow__step { padding: .9rem 1rem; border-inline-end: 1px solid var(--atlas-rule); }
.atlas-workflow__step:last-child { border-inline-end: 0; }
.atlas-workflow__step span { display: block; color: var(--atlas-cobalt); font-size: .75rem; font-weight: 700; }
.atlas-workflow__step strong { display: block; margin-top: .2rem; color: var(--atlas-ink); font-size: .9rem; }
.atlas-section-head { margin: 2.7rem 0 1.15rem; }
.atlas-section-head h2 { margin: 0; }
.atlas-section-head p { margin: .45rem 0 0; color: var(--atlas-muted) !important; max-width: 72ch; }

[data-testid="stVerticalBlockBorderWrapper"] {
  border-color: var(--atlas-rule) !important;
  border-radius: var(--atlas-radius) !important;
  background: var(--atlas-paper);
}
[data-testid="stAlert"] {
  border: 1px solid #C7D3E3;
  border-radius: 10px;
  box-shadow: none;
}
[data-testid="stAlert"] p { font-size: .92rem; line-height: 1.5; }
[data-testid="stExpander"] {
  border: 1px solid var(--atlas-rule);
  border-radius: 10px;
  background: var(--atlas-paper);
}

.atlas-result {
  background: var(--atlas-paper);
  border: 1px solid var(--atlas-rule);
  border-radius: var(--atlas-radius);
  padding: 1.55rem;
}
.atlas-result__top { display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; }
.atlas-result__label { color: var(--atlas-muted); font-size: .82rem; }
.atlas-result__number {
  margin-top: .28rem; color: var(--atlas-ink); font-family: "Manrope", sans-serif;
  font-size: clamp(3.25rem, 6vw, 5rem); font-weight: 600; letter-spacing: -.055em; line-height: 1;
  font-variant-numeric: tabular-nums;
}
.atlas-result__class { color: var(--atlas-cobalt); font-weight: 600; font-size: 1rem; }
.atlas-meter { height: 7px; margin: 1.25rem 0 .65rem; border-radius: 99px; background: #E5E9EE; overflow: hidden; }
.atlas-meter span { display: block; height: 100%; background: var(--atlas-cobalt); border-radius: inherit; }
.atlas-result__note { color: var(--atlas-muted); font-size: .87rem; line-height: 1.5; }
.atlas-ribbon {
  display: grid; grid-template-columns: repeat(4, minmax(0,1fr));
  margin-top: 1.35rem; border-block: 1px solid var(--atlas-rule);
}
.atlas-ribbon__item { padding: .9rem .8rem; border-right: 1px solid var(--atlas-rule); }
.atlas-ribbon__item:last-child { border-right: 0; }
.atlas-ribbon__item span { display: block; color: var(--atlas-muted); font-size: .75rem; }
.atlas-ribbon__item strong { display: block; margin-top: .22rem; font-size: .96rem; font-variant-numeric: tabular-nums; }
.atlas-ribbon__item--teal strong { color: var(--atlas-teal); }
.atlas-ribbon__item--amber strong { color: var(--atlas-amber); }

.atlas-metrics { display: grid; grid-template-columns: repeat(var(--metric-cols,4),minmax(0,1fr)); border-block: 1px solid var(--atlas-rule); }
.atlas-metric { padding: 1rem 1.1rem; border-right: 1px solid var(--atlas-rule); background: var(--atlas-paper); }
.atlas-metric:last-child { border-right: 0; }
.atlas-metric span { display: block; color: var(--atlas-muted); font-size: .78rem; }
.atlas-metric strong { display: block; margin-top: .35rem; font-family: "Manrope",sans-serif; font-size: 1.7rem; font-weight: 600; letter-spacing: -.03em; font-variant-numeric: tabular-nums; }
.atlas-metric small { display: block; margin-top: .3rem; color: var(--atlas-muted); font-size: .75rem; line-height: 1.35; }
.atlas-metric--cobalt strong { color: var(--atlas-cobalt); }
.atlas-metric--teal strong { color: var(--atlas-teal); }
.atlas-metric--amber strong { color: var(--atlas-amber); }
.atlas-metric--red strong { color: var(--atlas-red); }

.atlas-callout { padding: 1rem 1.05rem; border: 1px solid var(--atlas-rule); border-radius: 10px; background: var(--atlas-paper); }
.atlas-callout--teal { background: var(--atlas-teal-soft); border-color: #B7DDD7; }
.atlas-callout--amber { background: var(--atlas-amber-soft); border-color: #E8CB95; }
.atlas-callout--red { background: var(--atlas-red-soft); border-color: #E9B9B4; }
.atlas-callout strong { display: block; margin-bottom: .28rem; }
.atlas-callout strong { color: var(--atlas-ink); }
.atlas-callout p { margin: 0; color: #39434E !important; font-size: .9rem; line-height: 1.5; }

[data-testid="stButton"] button, [data-testid="stDownloadButton"] button {
  min-height: 44px; border-radius: 8px; border-color: #BEC6D0;
  font-family: "Source Sans 3", sans-serif; font-weight: 600;
  transition: transform 130ms ease, border-color 130ms ease, background 130ms ease;
}
[data-testid="stButton"] button:hover, [data-testid="stDownloadButton"] button:hover {
  transform: translateY(-1px); border-color: var(--atlas-cobalt); color: var(--atlas-cobalt);
}
[data-testid="stButton"] button:active, [data-testid="stDownloadButton"] button:active { transform: translateY(0); }
[data-testid="stButton"] button[kind="primary"] { background: var(--atlas-cobalt); border-color: var(--atlas-cobalt); color: white; }
[data-testid="stButton"] button[kind="primary"]:hover { background: #1749AE; color: white; }

[data-testid="stSlider"] [role="slider"] { color: var(--atlas-cobalt); }
[data-baseweb="slider"] div[role="progressbar"] { background-color: var(--atlas-cobalt) !important; }
[data-testid="stSelectbox"] > div > div, [data-testid="stNumberInput"] input,
[data-testid="stFileUploaderDropzone"] { border-radius: 8px; border-color: #C7CED7; background: var(--atlas-paper); }
[data-testid="stFileUploaderDropzone"] { border-style: solid; }

[data-testid="stDataFrame"] { border: 1px solid var(--atlas-rule); border-radius: 10px; overflow: auto; background: white; }
[data-testid="stDataFrame"] button { min-width: 36px; min-height: 36px; }
[data-testid="stPlotlyChart"] { background: var(--atlas-paper); border: 1px solid var(--atlas-rule); border-radius: 10px; padding: .4rem; }
[data-testid="stPlotlyChart"] .modebar { opacity: .12; transition: opacity 150ms ease; }
[data-testid="stPlotlyChart"]:hover .modebar { opacity: 1; }
code { border-radius: 7px; }

@media (max-width: 900px) {
  .block-container { padding: 1.6rem 1.25rem 4rem; }
  .atlas-ribbon { grid-template-columns: repeat(2,minmax(0,1fr)); }
  .atlas-ribbon__item:nth-child(2) { border-right: 0; }
  .atlas-ribbon__item:nth-child(-n+2) { border-bottom: 1px solid var(--atlas-rule); }
  .atlas-metrics { grid-template-columns: repeat(2,minmax(0,1fr)); }
  .atlas-metric:nth-child(2) { border-right: 0; }
  .atlas-metric:nth-child(-n+2) { border-bottom: 1px solid var(--atlas-rule); }
}
@media (max-width: 640px) {
  .block-container { padding: 1.15rem 1rem 3.5rem; }
  h1 { font-size: 2rem; }
  .atlas-page-head { margin-top: .2rem; margin-bottom: 1.2rem; }
  .atlas-context { font-size: .8rem; }
  .atlas-workflow { grid-template-columns: 1fr; }
  .atlas-workflow__step { border-inline-end: 0; border-block-end: 1px solid var(--atlas-rule); }
  .atlas-workflow__step:last-child { border-block-end: 0; }
  .atlas-result { padding: 1.15rem; }
  .atlas-result__number { font-size: 3.45rem; }
  .atlas-result__top { align-items: flex-end; }
  [data-testid="stPlotlyChart"] { padding: 0; }
  [data-testid="stDataFrame"] { max-width: calc(100vw - 2rem); }
}
@media (prefers-reduced-motion: reduce) {
  html { scroll-behavior: auto !important; }
  [data-testid="stButton"] button,
  [data-testid="stDownloadButton"] button,
  [data-testid="stSidebar"] [role="radiogroup"] label,
  [data-testid="stPlotlyChart"] .modebar { transition: none !important; }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_brand() -> None:
    st.sidebar.markdown(
        '<div class="atlas-brand"><span class="atlas-brand__name">StudPerf</span>'
        '<span class="atlas-brand__tag">Student probability atlas</span></div>',
        unsafe_allow_html=True,
    )


def sidebar_model_card(dataset_name: str, rows: int, accuracy: float) -> None:
    st.sidebar.markdown(
        f'<div class="atlas-model-card"><span>Research model</span>'
        f'<strong>{escape(dataset_name)}</strong>'
        f'<small>{rows:,} verified records · {accuracy:.1%} held-out accuracy</small></div>',
        unsafe_allow_html=True,
    )


def page_header(title: str, description: str) -> None:
    st.markdown(
        f'<div class="atlas-page-head"><h1>{escape(title)}</h1><p>{escape(description)}</p></div>',
        unsafe_allow_html=True,
    )


def context_bar(*items: tuple[str, str]) -> None:
    content = "".join(
        f'<span>{escape(label)} <strong>{escape(value)}</strong></span>' for label, value in items
    )
    st.markdown(f'<div class="atlas-context">{content}</div>', unsafe_allow_html=True)


def workflow_strip() -> None:
    st.markdown(
        '<div class="atlas-workflow">'
        '<div class="atlas-workflow__step"><span>1</span><strong>Set the student profile</strong></div>'
        '<div class="atlas-workflow__step"><span>2</span><strong>Read the pass probability</strong></div>'
        '<div class="atlas-workflow__step"><span>3</span><strong>Explore what changes it</strong></div>'
        '</div>',
        unsafe_allow_html=True,
    )


def section_header(title: str, description: str | None = None) -> None:
    detail = f"<p>{escape(description)}</p>" if description else ""
    st.markdown(
        f'<div class="atlas-section-head"><h2>{escape(title)}</h2>{detail}</div>',
        unsafe_allow_html=True,
    )


def metric_strip(items: list[tuple[str, str, str]], columns: int | None = None) -> None:
    cols = columns or len(items)
    cells = "".join(
        f'<div class="atlas-metric"><span>{escape(label)}</span>'
        f'<strong>{escape(value)}</strong><small>{escape(detail)}</small></div>'
        for label, value, detail in items
    )
    st.markdown(
        f'<div class="atlas-metrics" style="--metric-cols:{cols}">{cells}</div>',
        unsafe_allow_html=True,
    )


def probability_result(probability: float, threshold: float, model_name: str,
                       target_definition: str) -> None:
    label = "Likely pass" if probability >= threshold else "Likely fail"
    classification_tone = "var(--atlas-teal)" if probability >= threshold else "var(--atlas-red)"
    st.markdown(
        f"""
<div class="atlas-result">
  <div class="atlas-result__top">
    <div><div class="atlas-result__label">Estimated pass probability</div>
      <div class="atlas-result__number">{probability:.1%}</div></div>
    <div class="atlas-result__class" style="color:{classification_tone}">{escape(label)}</div>
  </div>
  <div class="atlas-meter" role="img" aria-label="Estimated pass probability {probability:.1%}">
    <span style="width:{probability:.2%}"></span>
  </div>
  <div class="atlas-result__note">This is a model estimate, not a guaranteed result or a measure of intelligence.</div>
  <div class="atlas-ribbon">
    <div class="atlas-ribbon__item"><span>Fail probability</span><strong>{1-probability:.1%}</strong></div>
    <div class="atlas-ribbon__item atlas-ribbon__item--amber"><span>Decision threshold</span><strong>{threshold:.0%}</strong></div>
    <div class="atlas-ribbon__item atlas-ribbon__item--teal"><span>Model</span><strong>{escape(model_name)}</strong></div>
    <div class="atlas-ribbon__item"><span>Target</span><strong>{escape(target_definition)}</strong></div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def callout(title: str, body: str, tone: str = "teal") -> None:
    st.markdown(
        f'<div class="atlas-callout atlas-callout--{escape(tone)}"><strong>{escape(title)}</strong>'
        f'<p>{escape(body)}</p></div>',
        unsafe_allow_html=True,
    )


def style_figure(fig, accent: str = "cobalt"):
    """Apply a consistent, accessible Plotly presentation."""
    color = COLORS.get(accent, COLORS["cobalt"])
    fig.update_layout(
        template="plotly_white",
        colorway=[color, COLORS["teal"], COLORS["amber"], COLORS["red"]],
        paper_bgcolor=COLORS["paper"],
        plot_bgcolor=COLORS["paper"],
        font={"family": "Source Sans 3, sans-serif", "color": COLORS["ink"], "size": 13},
        title={"font": {"family": "Manrope, sans-serif", "size": 18, "color": COLORS["ink"]}, "x": 0},
        margin={"l": 54, "r": 24, "t": 54, "b": 48},
        legend={"orientation": "h", "y": 1.08, "x": 0},
        hoverlabel={"bgcolor": COLORS["ink"], "font_color": "white", "bordercolor": COLORS["ink"]},
    )
    fig.update_xaxes(showgrid=False, linecolor=COLORS["rule"], zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor="#E8EBEF", linecolor=COLORS["rule"], zeroline=False, automargin=True)
    return fig


PLOT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}


def reset_scroll_when_page_changes(page: str) -> None:
    """Reset the parent Streamlit document after a navigation change."""
    if st.session_state.get("_last_visible_page") != page:
        st.session_state["_last_visible_page"] = page
        components.html(
            "<script>window.parent.scrollTo({top:0,left:0,behavior:'instant'});</script>",
            height=0,
        )
