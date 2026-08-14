import re
import time
import datetime
import streamlit as st
import base64
from pathlib import Path

from agent import (
    plan_trip, plan_trip_stream, refine_trip, chat_with_agent,
    get_weather, extract_place_names, get_place_locations,
    decide_travel_style, check_goal_achievement,
    ENERGY_OPTIONS, get_energy_options,
    SRI_LANKA_MONTHLY_GUIDE, SRI_LANKA_FIXED_EVENTS,
    annotate_sustainability, generate_packing_list, generate_pdf,
    get_exchange_rates, EMERGENCY_CONTACTS, COMMON_TOURIST_SCAMS,
    VISA_NOTE, ESSENTIAL_PHRASES, get_sample_itinerary,
    SUPPORTED_LANGUAGES, get_seasonal_highlights, get_weather_advisory,
    check_itinerary_weather,
)
from urllib.parse import quote as _url_quote
from memory import (
    save_trip, update_trip, get_recent_trips, get_total_trips,
    get_smart_memory_context, get_destination_frequency,
)

# PAGE CONFIG
st.set_page_config(
    page_title="WaLKer · AI Travel Agent",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# BRAND ASSET — Sri Lanka flag icon used next to the "WaLKer" wordmark
# (hero banner + sidebar logo). Loaded once and embedded as a base64 data
# URI so it renders inline via unsafe_allow_html without needing Streamlit's
# static file serving enabled.
@st.cache_data
def _load_flag_icon() -> str:
    flag_path = Path(__file__).parent / "log.png"
    encoded = base64.b64encode(flag_path.read_bytes()).decode()
    return f"data:image/png;base64,{encoded}"

FLAG_ICON = _load_flag_icon()

# BRAND ASSET — full WaLKer logo (wordmark + boot/leaf/Sri-Lanka icon),
# used in place of the plain text wordmark in the hero, sidebar, and chat
# header so the actual brand mark shows up everywhere instead of styled text.
@st.cache_data
def _load_logo() -> str:
    logo_path = Path(__file__).parent / "lo.png"
    encoded = base64.b64encode(logo_path.read_bytes()).decode()
    return f"data:image/png;base64,{encoded}"

LOGO_ICON = _load_logo()

# BRAND ASSET — full-color splash logo shown only on the boot screen below.
# Separate from LOGO_ICON (used in the hero/sidebar/chat header) so the
# splash can use a different image file: logo3.png, placed next to app.py.
@st.cache_data
def _load_splash_logo() -> str:
    splash_path = Path(__file__).parent / "logo3_bright_darkblue.png"
    encoded = base64.b64encode(splash_path.read_bytes()).decode()
    return f"data:image/png;base64,{encoded}"

SPLASH_LOGO = _load_splash_logo()

# BOOT / SPLASH SCREEN — shows the WaLKer logo full-screen for a moment
# before the actual app UI appears. Runs once per session: on the first
# script run we render the splash and stop; the rerun that follows sets
# app_booted=True so every run after that skips straight to the real app.
if "app_booted" not in st.session_state:
    st.session_state.app_booted = False

if not st.session_state.app_booted:
    st.markdown(f"""
    <style>
      #MainMenu, footer, header {{ visibility:hidden; }}
      .stApp {{ background:linear-gradient(160deg,#0a2732 0%,#0c2b28 45%,#15312f 100%) !important; }}
      .block-container {{ padding:0 !important; }}
      .walker-splash {{
        position:fixed; inset:0; display:flex; flex-direction:column;
        align-items:center; justify-content:center; gap:16px; z-index:99999;
      }}
      .walker-splash img {{
        width:440px; max-width:82vw;
        filter:drop-shadow(0 6px 24px rgba(0,0,0,0.45));
        animation:walkerPop 0.9s cubic-bezier(.2,.8,.25,1) both,
                  walkerFloat 2.4s ease-in-out 0.9s infinite;
      }}
      @keyframes walkerPop {{
        0%   {{ opacity:0; transform:scale(0.7) translateY(20px); }}
        60%  {{ opacity:1; transform:scale(1.06) translateY(-4px); }}
        100% {{ opacity:1; transform:scale(1) translateY(0); }}
      }}
      @keyframes walkerFloat {{
        0%,100% {{ transform:translateY(0); }}
        50%     {{ transform:translateY(-8px); }}
      }}
      .walker-splash .tagline {{
        font-family:Cambria,Georgia,serif; letter-spacing:3px; text-transform:uppercase;
        font-size:0.7rem; color:rgba(232,240,236,0.55);
        animation:walkerFade 0.9s ease 0.4s both;
      }}
      @keyframes walkerFade {{ from {{ opacity:0; }} to {{ opacity:1; }} }}
      .walker-splash .dots {{ display:flex; gap:6px; margin-top:4px; }}
      .walker-splash .dots span {{
        width:7px; height:7px; border-radius:50%; background:#3dba7e;
        animation:walkerDot 1.1s ease-in-out infinite;
      }}
      .walker-splash .dots span:nth-child(2) {{ animation-delay:0.15s; background:#2f8bb8; }}
      .walker-splash .dots span:nth-child(3) {{ animation-delay:0.3s; background:#e8b84b; }}
      @keyframes walkerDot {{
        0%,80%,100% {{ opacity:0.25; transform:translateY(0); }}
        40%         {{ opacity:1; transform:translateY(-5px); }}
      }}
    </style>
    <div class="walker-splash">
      <img src="{SPLASH_LOGO}" alt="WaLKer">
      <div class="tagline">Charting your Sri Lanka journey</div>
      <div class="dots"><span></span><span></span><span></span></div>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(2.0)
    st.session_state.app_booted = True
    st.rerun()


# STYLES
st.markdown("""
<style>
:root {
  --bg:#15312f;
  --bg2:#1a3d38;
  --panel:#204640;
  --panel2:#264f47;
  --border:rgba(255,255,255,0.10);
  --border2:rgba(255,255,255,0.16);
  --green:#3dba7e;
  --blue:#2f8bb8;
  --blue2:#1c5c7d;
  --gold:#e8b84b;
  --text:#e8f0ec;
  --text2:#b8ccc1;
  --muted:#759486;
} 
/* ============================================================
   FLUID SCALING PATCH (v2 — tightened ~10% to match target look)
   ============================================================ */

[data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
[data-testid="column"] { min-width: 54px !important; }

/* Page padding */
.block-container { padding: 0 clamp(0.7rem, 1.3vw, 1.2rem) clamp(1rem, 1.8vw, 1.4rem) !important; }

/* HERO */
.hero-wrap { height: clamp(120px, 8.4vw, 158px) !important; }
.hero-content { padding: 0 clamp(14px, 2.2vw, 34px) !important; gap: clamp(9px, 1.1vw, 14px) !important; }
.hero-badge { height: clamp(26px, 2.4vw, 38px) !important; }
.hero-text h1 { font-size: clamp(1.35rem, 2.15vw, 2.25rem) !important; }
.hero-logo { height: clamp(36px, 4.1vw, 63px) !important; }
.hero-text p { font-size: clamp(0.4rem, 0.5vw, 0.46rem) !important; margin: clamp(4px,0.55vw,7px) 0 0 !important; }
.hero-stat { padding: clamp(5px,0.7vw,7px) clamp(9px,1.25vw,12px) !important; }
.hero-stat-num { font-size: clamp(1rem, 1.15vw, 1.15rem) !important; }
.hero-stat-lbl { font-size: clamp(0.4rem, 0.45vw, 0.43rem) !important; }

/* SECTION HEADINGS */
.sec-title { font-size: clamp(1.05rem, 1.45vw, 1.25rem) !important; }
.sec-sub { font-size: clamp(0.46rem, 0.5vw, 0.5rem) !important; }
.panel-label { font-size: clamp(0.82rem, 1.08vw, 0.94rem) !important; }
.panel-sublabel { font-size: clamp(0.55rem, 0.63vw, 0.61rem) !important; }
.panel-caption { font-size: clamp(0.46rem, 0.5vw, 0.5rem) !important; }

/* BUDGET CARD BUTTONS */
.stButton[class*="st-key-budget_"] button {
  height: clamp(54px, 6.3vw, 66px) !important;
  font-size: clamp(0.6rem, 0.82vw, 0.7rem) !important;
}
.stButton[class*="st-key-budget_"] button p { font-size: clamp(0.6rem, 0.82vw, 0.7rem) !important; }

/* ARRIVAL CARD BUTTONS */
.stButton[class*="st-key-arr_"] button {
  height: clamp(50px, 5.9vw, 63px) !important;
  font-size: clamp(0.53rem, 0.72vw, 0.63rem) !important;
}
.stButton[class*="st-key-arr_"] button p { font-size: clamp(0.53rem, 0.72vw, 0.63rem) !important; }

/* INTEREST CARD IMAGES */
.st-key-form_right_panel img {
  height: clamp(100px, 11.7vw, 158px) !important;
  width: 100% !important;
  object-fit: cover !important;
}

/* GENERATE BUTTON */
.st-key-btn_generate[class] button {
  height: clamp(45px, 4.7vw, 58px) !important;
  font-size: clamp(0.9rem, 1.35vw, 1.17rem) !important;
}
.st-key-btn_generate[class] button p { font-size: clamp(0.9rem, 1.35vw, 1.17rem) !important; }

/* CHIPS */
.chip { font-size: clamp(0.47rem, 0.54vw, 0.54rem) !important; padding: clamp(2px,0.27vw,3px) clamp(7px,0.9vw,9px) !important; }

/* MONTH PICKER BUTTONS */
.stButton > button { font-size: clamp(0.5rem, 1.44vw, 0.81rem) !important; white-space: nowrap !important; }
/* Force every piece of real text in the app onto the same font — scoped to
   text-bearing tags rather than a blanket "*" selector, since Streamlit
   renders some of its own UI glyphs (expander chevrons, checkboxes, menu
   icons) as icon-font ligatures via <span class="material-symbols-...">;
   a universal override would turn those into literal words like
   "keyboard_arrow_down" instead of the icon shape. */
p, div, li, label, button, input, textarea, select, h1, h2, h3, h4, h5, h6, a,
span:not([class*="material"]):not([data-testid="stIconMaterial"]) {
  font-family: Cambria, Georgia, "Times New Roman", serif !important;
}
/* Belt-and-braces: force the correct icon font back on, in case the
   selector above still gets matched by a Streamlit DOM change. */
[data-testid="stIconMaterial"] {
  font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
}
.stApp{ background:#ffffff!important; }
#MainMenu,footer{ visibility:hidden; }
header{ visibility:visible!important; background:transparent!important; }
.block-container{ padding:0 1.6rem 2rem!important; max-width:100%!important; }

/* HERO */
.hero-wrap{ width:100%; height:220px; position:relative; overflow:hidden; border-radius:0 0 24px 24px; margin-bottom:20px; }
.hero-img{ width:100%; height:100%; object-fit:cover; display:block; }
.hero-bg-layer{ position:absolute; inset:0; background-size:cover; background-position:center; opacity:0;
  animation:heroKenBurns 24s ease-in-out infinite; will-change:opacity,transform; }
@keyframes heroKenBurns{
  0%   { opacity:0; transform:scale(1.06); }
  4%   { opacity:1; }
  20%  { opacity:1; transform:scale(1.16); }
  27%  { opacity:0; transform:scale(1.16); }
  100% { opacity:0; }
}
.hero-overlay{ position:absolute; inset:0;
  background:
    linear-gradient(105deg,rgba(5,18,11,0.94) 0%,rgba(5,18,11,0.68) 50%,rgba(5,18,11,0.28) 100%),
    linear-gradient(0deg,rgba(5,18,11,0.5) 0%,rgba(5,18,11,0) 42%); }
.hero-content{ position:absolute; inset:0; display:flex; align-items:center; padding:0 48px; gap:20px; }
.hero-badge{ height:52px; width:auto; filter:drop-shadow(0 2px 8px rgba(0,0,0,0.5)); }
.hero-text h1{ font-family:'Cambria',serif; font-size:3.2rem; font-weight:600; color:#fff; margin:0; text-shadow:0 2px 12px rgba(0,0,0,0.4); }
.hero-logo{ height:88px; width:auto; display:block; filter:drop-shadow(0 2px 10px rgba(0,0,0,0.45)); }
.hero-text p{ font-family:'Cambria',monospace; font-size:0.62rem; letter-spacing:2.5px; text-transform:uppercase; color:rgba(255,255,255,0.6); margin:10px 0 0; }
.hero-stats{ margin-left:auto; display:flex; gap:16px; }
.hero-stat{ text-align:center; background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.14); border-radius:9px; padding:10px 18px; }
.hero-stat-num{ font-family:'Cambria',serif; font-size:1.6rem; font-weight:600; color:#3dba7e; line-height:1; }
.hero-stat-lbl{ font-family:'Cambria',monospace; font-size:0.55rem; letter-spacing:1.2px; text-transform:uppercase; color:rgba(255,255,255,0.5); margin-top:4px; }

/* SIDEBAR */
[data-testid="stSidebar"]{ background:linear-gradient(180deg,#0a2732 0%,#0c2b28 45%,#0e2119 100%)!important; border-right:1px solid rgba(255,255,255,0.08)!important; }
[data-testid="stSidebar"] .block-container{ padding:1.2rem 1rem 2rem!important; }
[data-testid="collapsedControl"]{ display:flex!important; visibility:visible!important; opacity:1!important;
  background:#1a3528!important; border:1px solid rgba(61,186,126,0.3)!important; border-left:none!important;
  border-radius:0 8px 8px 0!important; position:fixed!important; top:50%!important; left:0!important;
  z-index:9999!important; width:24px!important; height:48px!important; cursor:pointer!important; }
[data-testid="collapsedControl"] svg {
    fill: #3dba7e !important;
    stroke: #3dba7e !important;
}
/* RESPONSIVE — phone + tablet. Streamlit already stacks st.columns
   vertically below its own ~640px breakpoint, so the budget / arrival /
   interest card grids reflow to one-per-row on their own; what's left is
   the fixed pixel sizing (hero banner, big serif headings, wide padding)
   that was tuned for a laptop screen and overflows or looks oversized on
   a phone. */
html, body, .stApp{ overflow-x:hidden!important; }
@media (max-width: 768px){
  .block-container{ padding:0 0.9rem 1.4rem!important; }

  .hero-wrap{ height:150px!important; border-radius:0 0 16px 16px!important; margin-bottom:14px!important; }
  .hero-content{ padding:0 18px!important; gap:10px!important; }
  .hero-badge{ height:32px!important; }
  .hero-text h1{ font-size:1.8rem!important; }
  .hero-logo{ height:38px!important; }
  .hero-text p{ font-size:0.5rem!important; letter-spacing:1.5px!important; margin:5px 0 0!important; }
  .hero-stats{ display:none!important; }

  /* Card buttons (budget / arrival / interest tiles) — Streamlit stacks the
     columns to one-per-row, so let each card breathe a bit less than the
     desktop 3-4-wide layout without shrinking tap targets below thumb size. */
  .stButton[class*="st-key-budget_"] button{ height:76px!important; font-size:0.78rem!important; }
  .stButton[class*="st-key-arr_"] button{ height:72px!important; font-size:0.7rem!important; }

  [data-testid="stSidebar"] .block-container{ padding:1rem 0.8rem 1.6rem!important; }
}

/* TEXT */
label,p,span,div,li,.stMarkdown p,.stMarkdown span,
div[data-testid="stWidgetLabel"] p, div[data-testid="stWidgetLabel"] span { color:var(--text)!important; }
div[data-testid="stSlider"] p{ color:var(--text2)!important; }
/* SLIDER — Streamlit's default theme accent is red (#FF4B4B) and nothing
   above overrides it for stSlider specifically, so the "days" slider was
   showing a red track/thumb that clashed with the green+gold palette used
   everywhere else. Re-themed to the same green with a gold thumb. */
div[data-testid="stSlider"] [role="slider"]{
  background-color: var(--gold) !important;
  border-color: var(--gold) !important;
  box-shadow: 0 0 0 3px rgba(232,184,75,0.25) !important;
}
div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div{
  background: var(--blue) !important; /* filled portion of the track */
}
div[data-testid="stSlider"] div[data-baseweb="slider"] > div{
  background: rgba(255,255,255,0.16) !important; /* unfilled portion */
}
div[data-testid="stSlider"] div[data-testid="stTickBar"]{
  color: var(--text2) !important;
}

/* INPUTS — same gradient as the Search Weather / Refresh Rate buttons,
   applied to every text input and textarea (city search, currency, day-1 /
   month / language selects live in their own rule blocks below, extra-info
   textarea, etc.) so both sidebar and main-content widgets share one look. */
.stTextInput input,.stTextArea textarea{
  background:linear-gradient(135deg,#0c3446,#19352a)!important; border:1.5px solid rgba(61,186,126,0.4)!important;
  color:#e8f0ec!important; border-radius:9px!important; font-size:0.87rem!important; }
.stTextInput input::placeholder,.stTextArea textarea::placeholder{ color:#5a7a6a!important; }
.stTextInput input:focus,.stTextArea textarea:focus{ border-color:#3dba7e!important; box-shadow:0 0 0 3px rgba(61,186,126,0.15)!important; }

/* RADIO */
.stRadio>div{ display:flex; flex-direction:column; gap:6px; }
.stRadio div[role="radiogroup"]>label{
  background:#1a3528!important; border:1.5px solid rgba(255,255,255,0.12)!important;
  border-radius:7px!important; padding:8px 14px!important; color:#a8bfb3!important;
  font-size:0.84rem!important; cursor:pointer!important; }
.stRadio div[role="radiogroup"]>label:has(input:checked){
  border-color:#e8b84b!important; background:rgba(232,184,75,0.1)!important; color:#e8b84b!important; font-weight:600!important; }

/* MAIN BUTTON — right-side buttons (Generate Itinerary, Apply Changes, send
   chat, clear chat, demo fallback, etc.) now use the same background colour
   as the left-side sidebar panel, per user request. Left-side sidebar
   buttons are styled separately below and are untouched. */
.stButton>button{
  background:linear-gradient(135deg,#0c3446,#19352a)!important; color:#e8f0ec!important;
  border:1.5px solid rgba(61,186,126,0.4)!important; border-radius:14px!important; font-weight:600!important;
  font-size:0.9rem!important; padding:12px 24px!important; width:100%;
  box-shadow:none!important; transition:all 0.2s!important; }
.stButton>button:hover{
  background:linear-gradient(135deg,#123a4d,#1f4432)!important;
  border-color:#3dba7e!important; transform:translateY(-1px)!important; }
.stButton>button:active,
.stButton>button:focus{
  background:linear-gradient(135deg,#0c3446,#19352a)!important;
  box-shadow:0 0 0 3px rgba(61,186,126,0.3)!important; outline:none!important; }

/* SIDEBAR BUTTON — background now matches the main-content page gradient
   (right side) instead of the green tint, per user request to swap the
   two panels' background colours onto each other's buttons. */
[data-testid="stSidebar"] .stButton>button{
  background:linear-gradient(135deg,#0c3446,#19352a)!important; color:#e8f0ec!important;
  border:1px solid rgba(61,186,126,0.25)!important; box-shadow:none!important;
  font-size:0.78rem!important; padding:6px 10px!important; }
[data-testid="stSidebar"] .stButton>button:hover{ background:linear-gradient(135deg,#123a4d,#1f4432)!important; transform:none!important; }
[data-testid="stSidebar"] .stButton>button:active,
[data-testid="stSidebar"] .stButton>button:focus{ background:linear-gradient(135deg,#0c3446,#19352a)!important; outline:none!important; box-shadow:none!important; }

/* DOWNLOAD BUTTON — same left-side background colour as the other
   right-side buttons/dropdowns. */
[data-testid="stDownloadButton"]>button{
  background:linear-gradient(135deg,#0c3446,#19352a)!important; color:#e8f0ec!important;
  border:1.5px solid rgba(61,186,126,0.4)!important; border-radius:10px!important;
  font-weight:600!important; font-size:0.86rem!important; padding:10px 20px!important; width:100%; box-shadow:none!important; }
[data-testid="stDownloadButton"]>button:hover{
  background:linear-gradient(135deg,#123a4d,#1f4432)!important; border-color:#3dba7e!important; transform:translateY(-1px)!important; }

/* EXPANDER (Trip Toolkit) — same gradient as the Search Weather / Refresh
   Rate buttons instead of the flat green it had before. */
.streamlit-expanderHeader{ background:linear-gradient(135deg,#0c3446,#19352a)!important; border:1px solid rgba(61,186,126,0.4)!important; border-radius:9px!important; color:#e8f0ec!important; }
details[open] .streamlit-expanderHeader{ border-radius:9px 9px 0 0!important; }
.streamlit-expanderContent{ background:linear-gradient(135deg,#0c3446,#19352a)!important; border:1px solid rgba(61,186,126,0.3)!important; border-top:none!important; border-radius:0 0 9px 9px!important; }
[data-testid="stExpander"]{ background:linear-gradient(135deg,#0c3446,#19352a)!important; border:1px solid rgba(61,186,126,0.25)!important; border-radius:9px!important; }
[data-testid="stExpander"] summary{ background:linear-gradient(135deg,#0c3446,#19352a)!important; color:#e8f0ec!important; }
[data-testid="stExpander"] details{ background:linear-gradient(135deg,#0c3446,#19352a)!important; }

/* TABS (Trip Toolkit — Packing List / Safety & Scams / Phrases) — this had
   no custom styling at all before, so it fell back to Streamlit's default
   white/red theme, which is the main thing that broke the dark background
   match inside the Trip Toolkit expander. Same gradient as every other
   right-side control, active tab picked out with the green accent. */
.stTabs [data-baseweb="tab-list"]{
  background:linear-gradient(135deg,#0c3446,#19352a)!important;
  border:1px solid rgba(61,186,126,0.3)!important;
  border-radius:9px!important;
  gap:2px!important;
  padding:4px!important;
}
.stTabs [data-baseweb="tab"]{
  background:transparent!important;
  color:#a8bfb3!important;
  border-radius:7px!important;
  font-family:'Cambria',serif!important;
}
.stTabs [data-baseweb="tab"]:hover{
  background:linear-gradient(135deg,#123a4d,#1f4432)!important;
  color:#e8f0ec!important;
}
.stTabs [aria-selected="true"]{
  background:linear-gradient(135deg,#123a4d,#1f4432)!important;
  color:#e8f0ec!important;
}
.stTabs [data-baseweb="tab-highlight"]{ background:#3dba7e!important; }
.stTabs [data-baseweb="tab-border"]{ background:rgba(61,186,126,0.2)!important; }
.stTabs [data-baseweb="tab-panel"]{
  background:linear-gradient(135deg,#0c3446,#19352a)!important;
  border:1px solid rgba(61,186,126,0.25)!important;
  border-top:none!important;
  border-radius:0 0 9px 9px!important;
  padding:14px 16px!important;
}

/* NUMBER INPUT — currency converter amount field. Streamlit doesn't theme
   this on its own, so left unstyled it renders with the browser/Streamlit
   default white background regardless of our dark-green palette. */
.stNumberInput input,
div[data-testid="stNumberInput"] input{
  background:linear-gradient(135deg,#0c3446,#19352a)!important; border:1.5px solid rgba(61,186,126,0.4)!important;
  color:#e8f0ec!important; border-radius:9px!important; font-size:0.87rem!important; }
.stNumberInput input:focus,
div[data-testid="stNumberInput"] input:focus{
  border-color:#3dba7e!important; box-shadow:0 0 0 3px rgba(61,186,126,0.15)!important; }
div[data-testid="stNumberInputStepUp"],
div[data-testid="stNumberInputStepDown"]{
  background:linear-gradient(135deg,#0c3446,#19352a)!important; border-color:rgba(61,186,126,0.4)!important; }
div[data-testid="stNumberInputStepUp"] svg,
div[data-testid="stNumberInputStepDown"] svg{ fill:#a8bfb3!important; }

/* CURRENCY REFRESH (↻) BUTTON — explicit override on top of the generic
   sidebar button rule, since a plain icon-only button is small enough that
   specificity clashes are easy to lose. Sized like a small chip pill (see
   .chip / .sug-chip) rather than stretching to the full width of the
   currency row above it. */
.st-key-btn_refresh_rates button{
  background:linear-gradient(135deg,#0c3446,#19352a)!important; color:#e8f0ec!important;
  border:1.5px solid rgba(61,186,126,0.4)!important; font-size:0.60rem!important;
  width:auto!important; display:inline-flex!important;
  padding:4px 12px!important; }
.st-key-btn_refresh_rates button:hover{
  background:linear-gradient(135deg,#123a4d,#1f4432)!important; border-color:#3dba7e!important; }

/* SEARCH WEATHER BUTTON — same small chip-pill treatment as the refresh
   rate button above, so it doesn't stretch to the full width of the city
   search box / currency value fields sitting next to it. */
.st-key-btn_weather button{
  width:auto!important; display:inline-flex!important;
  font-size:0.60rem!important; padding:4px 12px!important; }

/* STREAMLIT TOOLTIP POPUP — shows up as a stray white box on hover for any
   widget with a `help=` param (e.g. the refresh button) since it's unstyled
   by default and falls back to the browser/Streamlit light theme. */
[data-testid="stTooltipContent"]{
  background:#1a3528!important; border:1px solid rgba(61,186,126,0.3)!important;
  color:#e8f0ec!important; border-radius:8px!important; font-size:0.78rem!important;
}
div[data-baseweb="tooltip"]{ background:#1a3528!important; }


  
/* STREAMLIT'S OWN TOP TOOLBAR — the built-in rerun/menu/deploy icons in the
   header keep their light-theme default color otherwise, which shows up as
   a stray white icon floating over the dark background. */
[data-testid="stToolbar"] button,
[data-testid="stToolbarActions"] button,
[data-testid="stStatusWidget"] button,
[data-testid="stHeaderActionElements"] button{
  background:transparent!important; }
[data-testid="stToolbar"] svg,
[data-testid="stToolbarActions"] svg,
[data-testid="stStatusWidget"] svg,
[data-testid="stHeaderActionElements"] svg{
  fill:#3dba7e!important; color:#3dba7e!important; }
[data-testid="stStatusWidget"]{
  background:#1a3528!important; border:1px solid rgba(61,186,126,0.25)!important;
  border-radius:8px!important; color:#e8f0ec!important; }

/* SELECTBOX — same gradient as the Search Weather / Refresh Rate buttons,
   covers the currency-converter selectbox in the sidebar as well as the
   Day 1 / month / language selectboxes in the main content. Also covers
   the open/focused state and the dropdown option list, which previously
   fell back to Streamlit's default blue/green instead of this gradient. */
.stSelectbox>div>div{ background:linear-gradient(135deg,#0c3446,#19352a)!important; border:1.5px solid rgba(61,186,126,0.4)!important; color:#e8f0ec!important; border-radius:9px!important; }
.stSelectbox>div>div:focus-within{ border-color:#3dba7e!important; box-shadow:0 0 0 3px rgba(61,186,126,0.15)!important; }
[data-baseweb="popover"],[data-baseweb="menu"]{ background:linear-gradient(135deg,#0c3446,#19352a)!important; }
[data-baseweb="option"]{ color:#e8f0ec!important; background:linear-gradient(135deg,#0c3446,#19352a)!important; }
[data-baseweb="option"]:hover{ background:linear-gradient(135deg,#123a4d,#1f4432)!important; }
[data-baseweb="option"][aria-selected="true"]{ background:linear-gradient(135deg,#123a4d,#1f4432)!important; color:#e8f0ec!important; }

/* SELECTBOX — extra, stronger rules using stable data-testid selectors.
   Newer Streamlit versions changed the internal class names, which can make
   the .stSelectbox rule above silently stop matching (closed box AND the
   dropdown list both fall back to Streamlit's default white theme). These
   target the wrapper by testid and the dropdown popover directly so styling
   holds regardless of Streamlit version. */
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
div[data-testid="stSelectbox"] div[data-baseweb="select"]{
    background:linear-gradient(135deg,#0c3446,#19352a)!important;
    border:1.5px solid rgba(61,186,126,0.4)!important;
    color:#e8f0ec!important;
    border-radius:9px!important;
}
div[data-baseweb="popover"] li:hover,
div[data-baseweb="popover"] li[aria-selected="true"] {
    background:linear-gradient(135deg,#123a4d,#1f4432)!important;
    color:#ffffff!important;
}

ul[role="listbox"],
div[role="listbox"] {
    background: linear-gradient(135deg,#0c3446,#19352a) !important;
    border: 1px solid rgba(61,186,126,0.35) !important;
}
ul[role="listbox"] li,
li[role="option"] {
    background: linear-gradient(135deg,#0c3446,#19352a) !important;
    color: #e8f0ec !important;
}
ul[role="listbox"] li *,
li[role="option"] * {
    color: #e8f0ec !important;
}
li[role="option"]:hover,
li[aria-selected="true"][role="option"] {
    background: linear-gradient(135deg,#123a4d,#1f4432) !important;
    color: #ffffff !important;
}

/* PAST TRIPS LIST — real buttons in a bounded, normally-scrolling
div[data-testid="stSelectbox"] div[data-baseweb="select"] span,
div[data-testid="stSelectbox"] svg {
    color:#e8f0ec!important;
    fill:#e8f0ec!important;
}
div[data-baseweb="popover"] {
    background:linear-gradient(135deg,#0c3446,#19352a)!important;
}
div[data-baseweb="popover"] ul[role="listbox"] {
    background:linear-gradient(135deg,#0c3446,#19352a)!important;
}
div[data-baseweb="popover"] li,
div[data-baseweb="popover"] li[role="option"] {
    background:linear-gradient(135deg,#0c3446,#19352a)!important;
    color:#e8f0ec!important;
}
div[data-baseweb="popover"] li:hover,
div[data-baseweb="popover"] li[aria-selected="true"] {
    background:linear-gradient(135deg,#123a4d,#1f4432)!important;
    color:#ffffff!important;
}

/* PAST TRIPS LIST — real buttons in a bounded, normally-scrolling
   container (see the Python block that renders them), styled to look
   like list rows rather than the default green pill button. */
[class*="st-key-past_trips_list"] .stButton>button {
  background:linear-gradient(135deg,#0c3446,#19352a)!important;
  border:1px solid rgba(255,255,255,0.08)!important;
  border-radius:8px!important;
  color:#c3d4cb!important;
  font-family:'Cambria',monospace!important;
  font-size:0.7rem!important;
  font-weight:400!important;
  text-align:left!important;
  justify-content:flex-start!important;
  padding:8px 12px!important;
  box-shadow:none!important;
  margin-bottom:4px!important;
}
[class*="st-key-past_trips_list"] .stButton>button:hover {
  background:linear-gradient(135deg,#123a4d,#1f4432)!important;
  border-color:rgba(61,186,126,0.4)!important;
  color:#e8f0ec!important;
  transform:none!important;
  box-shadow:none!important;
}

/* ARRIVAL CARD BUTTONS — same key-based approach as the budget cards above.
   The previous ".arr-btn-wrap > div[data-testid=stButton] > button" selector
   never matched anything (Streamlit doesn't nest the wrapping <div> around
   the widget the way a plain st.markdown div appears to), so these buttons
   were rendering with zero custom styling — just the bare default button
   the later "hide default button styling" reset produces. */
.stButton[class*="st-key-arr_"] button {
  background: linear-gradient(135deg,#0c3446,#19352a) !important;
  border: 2px solid rgba(255,255,255,0.32) !important;
  border-radius: 12px !important;
  padding: 10px 4px 8px !important;
  width: 100% !important;
  height: 60px !important;
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 3px !important;
  color: #c3d4cb !important;
  font-family: 'Cambria', serif !important;
  font-size: 0.65rem !important;
  font-weight: 600 !important;
  box-shadow: none !important;
  transform: none !important;
  transition: border-color 0.15s, background 0.15s !important;
  white-space: pre-line !important;
  line-height: 1.4 !important;
  text-align: center !important;
}
.stButton[class*="st-key-arr_"] button p {
  font-family: 'Cambria', serif !important;
  font-size: 0.85rem !important;
  line-height: 1.4 !important;
}
.stButton[class*="st-key-arr_"] button:hover {
  border-color: rgba(47,139,184,0.7) !important;
  background: linear-gradient(135deg,#123a4d,#1f4432) !important;
}

/* INTEREST IMAGE CARD BUTTONS */
.img-btn-wrap > div[data-testid="stButton"] > button {
  background: transparent !important;
  border: 2px solid transparent !important;
  border-radius: 12px !important;

  width: 20px !important;   /* reduce width */
  height: 20px !important;  /* reduce height */

  padding: 0 !important;
  overflow: hidden !important;

  box-shadow: none !important;
  transition: border-color 0.18s, box-shadow 0.18s !important;

  display: block !important;
  cursor: pointer !important;
}
.img-btn-wrap > div[data-testid="stButton"] > button:hover {
  transform: none !important;
  box-shadow: 0 4px 20px rgba(0,0,0,0.5) !important;
  border-color: rgba(47,139,184,0.4) !important;
}
.img-btn-wrap.sel > div[data-testid="stButton"] > button {
  border-color: #4fa8d1 !important;
  box-shadow: 0 0 0 2px rgba(47,139,184,0.4) !important;
}

/* CHIPS */
.chip{ font-family:'Cambria',monospace; font-size:0.62rem; padding:3px 10px; border-radius:20px; display:inline-block; }
.chip-g{ background:rgba(61,186,126,0.12); color:#3dba7e; border:1px solid rgba(61,186,126,0.25); }
.chip-y{ background:rgba(232,184,75,0.12); color:#e8b84b; border:1px solid rgba(232,184,75,0.25); }
.chip-m{ background:rgba(255,255,255,0.05); color:#a8bfb3; border:1px solid rgba(255,255,255,0.1); }
/* TRENDING DESTINATIONS CHIPS — sidebar-scoped override so these pick up
   the main-content background gradient without touching .chip-g used
   elsewhere in the app (itinerary badges, interest chips, etc). */
[data-testid="stSidebar"] .chip-g{ background:linear-gradient(135deg,#0c3446,#19352a); color:#e8f0ec; border:1px solid rgba(61,186,126,0.25); }
.chip-row{ display:flex; gap:5px; flex-wrap:wrap; margin-top:10px; }
.goal-pass{ background:rgba(61,186,126,0.12); color:#3dba7e; border:1px solid rgba(61,186,126,0.25); font-family:'Cambria',monospace; font-size:0.62rem; padding:3px 10px; border-radius:20px; display:inline-block; }
.goal-fail{ background:rgba(220,80,60,0.12); color:#e87060; border:1px solid rgba(220,80,60,0.25); font-family:'Cambria',monospace; font-size:0.62rem; padding:3px 10px; border-radius:20px; display:inline-block; }

/* MISC */
.mono-label{ font-family:'Cambria',monospace; font-size:0.62rem; letter-spacing:1.5px; text-transform:uppercase; color:#3dba7e; margin-bottom:8px; display:block; }
.sec-title{ font-family:'Cambria',serif; font-size:1.75rem; font-weight:600; color:#e8f0ec; margin:0 0 2px; }
.sec-sub{ font-family:'Cambria',monospace; font-size:0.6rem; letter-spacing:1.5px; text-transform:uppercase; color:#5a7a6a; margin-bottom:16px; display:block; }
.divider{ border:none; border-top:1px solid rgba(12,52,70,0.15); margin:18px 0; }
.ai-box{ background:rgba(61,186,126,0.07); border:1px solid rgba(61,186,126,0.2); border-left:3px solid #3dba7e; border-radius:9px; padding:11px 15px; font-size:0.82rem; color:#a8bfb3; margin-top:10px; }
.ai-box strong{ color:#3dba7e; }
.hist-card{ background:#1f3d2f; border:1px solid rgba(255,255,255,0.07); border-radius:9px; padding:10px 14px; margin-bottom:8px; font-size:0.77rem; line-height:1.65; color:#a8bfb3; }
.hist-card b{ color:#3dba7e; }
.status-item{ background:#1a3528; border:1px solid rgba(255,255,255,0.08); border-radius:9px; padding:8px 12px; text-align:center; font-family:'Cambria',monospace; font-size:0.64rem; }
.status-on{ color:#3dba7e; } .status-off{ color:#5a7a6a; }

/* ARRIVAL ADVICE */
.arrival-advice{ border-radius:8px; padding:10px 13px; font-size:0.76rem; line-height:1.6; border-left:3px solid; margin-top:10px; }
.advice-morning{ background:rgba(232,184,75,0.08); border-color:#e8b84b; color:#c9a03e; }
.advice-afternoon{ background:rgba(61,186,126,0.08); border-color:#3dba7e; color:#3dba7e; }
.advice-evening{ background:rgba(100,160,220,0.08); border-color:#6490cc; color:#7aaae8; }
.advice-night{ background:rgba(160,100,220,0.08); border-color:#9b6fd4; color:#b08ae0; }

/* ITINERARY INNER */
#itin-inner{ font-family:Cambria,Georgia,"Times New Roman",serif; }
#itin-inner h1,#itin-inner h2{ font-family:Cambria,Georgia,"Times New Roman",serif; color:#3dba7e; border-bottom:1px solid rgba(61,186,126,0.2); padding-bottom:4px; margin:16px 0 6px; }
#itin-inner h1{ font-size:1.4rem; } #itin-inner h2{ font-size:1.2rem; }
#itin-inner h3{ font-family:Cambria,Georgia,"Times New Roman",serif; color:#e8b84b; font-size:1rem; margin:10px 0 3px; }
#itin-inner p,#itin-inner li{ font-family:Cambria,Georgia,"Times New Roman",serif; font-size:0.94rem; line-height:2.0; color:#e8f0ec; }
#itin-inner strong{ color:#e8b84b; }
#itin-inner hr{ border:none; border-top:1px solid rgba(255,255,255,0.08); margin:12px 0; }
#itin-inner ul{ padding-left:1.2rem; margin:4px 0; }

/* CHAT BUBBLES */
.chat-header-bar{ background:linear-gradient(135deg,#1a3528,#162d22); border:1px solid rgba(61,186,126,0.2); border-radius:14px 14px 0 0; padding:14px 20px; display:flex; align-items:center; gap:10px; }
.chat-hicon{ width:34px; height:34px; background:linear-gradient(135deg,#2f8bb8,#1c5c7d); border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:1rem; flex-shrink:0; }
.chat-htitle{ font-family:'Cambria',serif; font-size:1.05rem; font-weight:600; color:#e8f0ec; }
.chat-hsub{ font-family:'Cambria',monospace; font-size:0.54rem; letter-spacing:1px; text-transform:uppercase; color:#5a7a6a; }
.chat-online{ width:7px; height:7px; background:#3dba7e; border-radius:50%; box-shadow:0 0 6px #3dba7e; margin-left:auto; }

.bubble-user-row{ display:flex; justify-content:flex-end; margin-bottom:12px; }
.bubble-user{ background:linear-gradient(135deg,rgba(61,186,126,0.2),rgba(42,144,98,0.14)); border:1px solid rgba(61,186,126,0.3); border-radius:14px 14px 4px 14px; padding:10px 14px; max-width:72%; color:#0c3446 !important; font-size:0.86rem; line-height:1.7; }
.bubble-user-meta{ font-family:'Cambria',monospace; font-size:0.52rem; color:#0c3446 !important; text-align:right; margin-top:3px; }

.bubble-agent-row{ display:flex; justify-content:flex-start; align-items:flex-start; gap:8px; margin-bottom:12px; }
.chat-avatar{ width:28px; height:28px; background:linear-gradient(135deg,#2f8bb8,#1c5c7d); border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:0.8rem; flex-shrink:0; margin-top:2px; }
.bubble-agent{ background:#1f3d2f; border:1px solid rgba(255,255,255,0.09); border-radius:4px 14px 14px 14px; padding:10px 14px; max-width:80%; color:#e8f0ec; font-size:0.86rem; line-height:1.8; }
.bubble-agent-meta{ font-family:'Cambria',monospace; font-size:0.52rem; color:#0c3446 !important; margin-top:3px; }

.sug-row{ display:flex; gap:6px; flex-wrap:wrap; padding:10px 18px 8px; background:#ffffff; border:1px solid rgba(12,52,70,0.15); border-top:none; }
.sug-chip{ font-size:0.68rem; background:rgba(47,139,184,0.08); border:1px solid rgba(47,139,184,0.22); border-radius:20px; padding:4px 11px; color:#4fa8d1; white-space:nowrap; }
.sug-label{ font-family:'Cambria',monospace; font-size:0.54rem; letter-spacing:1px; text-transform:uppercase; color:#0c3446 !important; width:100%; }

.chat-empty{ background:#111f18; border:1px solid rgba(61,186,126,0.15); border-top:none; padding:28px; text-align:center; border-radius:0 0 14px 14px; }

::-webkit-scrollbar{ width:5px; }
::-webkit-scrollbar-track{ background:#0d1f17; }
::-webkit-scrollbar-thumb{ background:#1f3d2f; border-radius:4px; }
        
/* BUDGET CARD BUTTONS — one clickable card (icon + title + sub-label together).
   Streamlit gives every keyed widget's wrapper a real, working ".st-key-<key>"
   class, unlike the manual "<div class='...'>" wraps used elsewhere in this
   file (those never nest around the widget, so their CSS silently no-ops).
   NOTE: selector is compounded as ".stButton[class*=...]" (not a plain
   descendant selector) so its specificity beats the later blanket
   "div.stButton > button { border:none }" reset further down this file —
   with a plain descendant selector that reset was silently winning and the
   border/background never showed up at all. */
.stButton[class*="st-key-budget_"] button {
  background: linear-gradient(135deg,#0c3446,#19352a) !important;
  border: 2px solid rgba(255,255,255,0.32) !important;
  border-radius: 12px !important;
  height: 66px !important;
  width: 100% !important;
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 4px !important;
  white-space: pre-line !important;
  line-height: 1.45 !important;
  color: #c3d4cb !important;
  font-family: 'Cambria', serif !important;
  font-size: 0.95rem !important;
  font-weight: 600 !important;
  box-shadow: none !important;
  transform: none !important;
  transition: border-color 0.15s, background 0.15s !important;
}
.stButton[class*="st-key-budget_"] button:hover {
  border-color: rgba(47,139,184,0.7) !important;
  background: linear-gradient(135deg,#123a4d,#1f4432) !important;
}
.stButton[class*="st-key-budget_"] button p {
  font-family: 'Cambria', serif !important;
  font-size: 0.95rem !important;
  line-height: 1.45 !important;
}
/* GENERATE ITINERARY BUTTON — big, same left-side background gradient as
   the other right-side buttons/dropdowns. Uses the simple ".st-key-<key>
   button" descendant selector (confirmed working elsewhere in this file,
   e.g. .st-key-btn_send_chat) — NOT the compound ".stButton[class*=...]"
   form, since on this Streamlit version "st-key-<key>" lands on an
   ancestor wrapper div, not the same element as ".stButton", so the
   compound form never matches. */
.st-key-btn_generate[class] button {
  background: linear-gradient(135deg,#0c3446,#19352a) !important;
  border: 2px solid rgba(61,186,126,0.4) !important;
  border-radius: 14px !important;
  height: 68px !important;
  width: 100% !important;
  color: #e8f0ec !important;
  font-family: 'Cambria', serif !important;
  font-size: 1.35rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.5px !important;
  box-shadow: none !important;
  transform: none !important;
  transition: border-color 0.15s, background 0.15s, box-shadow 0.15s !important;
}
.st-key-btn_generate[class] button:hover {
  background: linear-gradient(135deg,#123a4d,#1f4432) !important;
  border-color: #3dba7e !important;
  box-shadow: 0 6px 18px rgba(61,186,126,0.2) !important;
}
.st-key-btn_generate[class] button p {
  font-family: 'Cambria', serif !important;
  font-size: 1.35rem !important;
  font-weight: 700 !important;
}
/* Make ALL buttons use Cambria */
.stButton > button {
    font-family: 'Cambria', serif !important;
}
/* LEFT COLUMN PANEL — "Plan Your Trip" section, given the same white card
   background as the interests panel per user request. Only the plain
   headings/labels and the near-white text colours (tuned for the old dark
   background, which would be invisible on white) are re-scoped to dark
   navy here — the slider, dropdowns, and the budget/arrival selection
   cards keep their own existing styling untouched. */
.st-key-form_left_panel{
  background:#ffffff !important;
  border-radius:0 !important;
  padding:0 22px !important;
}
.st-key-form_left_panel .sec-title{ color:#0c3446 !important; }
.st-key-form_left_panel .sec-sub{ color:#0c3446 !important; }
/* Dedicated, targeted class for the plain section labels ("How many
   days?", "Your Budget?", "When do you arrive?") so they don't depend
   on inline styles alone. Deliberately scoped to .panel-label only —
   does NOT touch the Budget/Arrival option cards, which keep their own
   white-on-navy styling untouched. */
.st-key-form_left_panel .panel-label{
  font-family:'Cambria',serif !important;
  font-size:1.30rem !important;
  font-weight:600 !important;
  color:#0c3446 !important;
}
.st-key-form_left_panel .panel-sublabel{
  font-size:0.71rem !important;
  color:#0c3446 !important;
}
.st-key-form_right_panel .panel-label{
  font-family:'Cambria',serif !important;
  font-size:1.30rem !important;
  font-weight:600 !important;
  color:#0c3446 !important;
}
/* Results placeholder ("Your itinerary will appear here") — lives
   outside both panel containers, on the plain white page background,
   so it gets its own class rather than relying on inline styles. */
.empty-state{ text-align:center !important; padding:70px 20px 0 !important; color:#0c3446 !important; }
.empty-state-icon{ font-size:4rem !important; margin-bottom:18px !important; }
.empty-state-title{ font-family:'Cambria',serif !important; font-size:1.7rem !important; color:#0c3446 !important; margin-bottom:10px !important; }
.empty-state-sub{ font-size:0.84rem !important; line-height:2 !important; color:#0c3446 !important; }
.st-key-form_left_panel .hint-text{ font-size:0.78rem !important; color:#0c3446 !important; padding:8px 0 !important; }
.st-key-form_right_panel .panel-caption{
  font-family:'Cambria',monospace !important; font-size:0.6rem !important; letter-spacing:1.5px !important;
  text-transform:uppercase !important; color:#2c4450 !important; display:block !important; margin-bottom:14px !important;
}
/* RIGHT COLUMN PANEL — "What are your interests?" section, given a white
   card background per user request. Text/box colours that were tuned for
   the dark page background are re-scoped here so they stay readable on
   white; nothing outside this panel is affected. */
.st-key-form_right_panel{
  background:#ffffff !important;
  border-radius:0 !important;
  padding:0 22px !important;
}
.st-key-form_right_panel .ai-box{
  color:#2c4450 !important;
}
.st-key-form_right_panel .ai-box strong{ color:#0c3446 !important; }
.st-key-form_right_panel .chip-g{
  background:rgba(12,52,70,0.08) !important;
  color:#0c3446 !important;
  border:1px solid rgba(12,52,70,0.3) !important;
}
/* SIDEBAR COLLAPSE ARROW */
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    position: fixed !important;
    left: 0 !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    z-index: 999999 !important;

    width: 32px !important;
    height: 52px !important;

    background: #1a3528 !important;
    border: 1px solid rgba(61,186,126,0.45) !important;
    border-left: none !important;
    border-radius: 0 9px 9px 0 !important;

    padding: 0 !important;
    margin: 0 !important;
}

/* Make the actual arrow visible */
[data-testid="collapsedControl"] svg {
    width: 22px !important;
    height: 22px !important;
    fill: #3dba7e !important;
    stroke: #3dba7e !important;
    color: #3dba7e !important;
    opacity: 1 !important;
    visibility: visible !important;
}

/* If Streamlit puts the icon inside a span */
[data-testid="collapsedControl"] span {
    color: #3dba7e !important;
    opacity: 1 !important;
    visibility: visible !important;
}

/* Hover */
[data-testid="collapsedControl"]:hover {
    background: #234b38 !important;
    border-color: #3dba7e !important;
}

/* DROPDOWN TEXT FIX — final, highest-priority pass. Placed last in the
   stylesheet so it wins the cascade over any earlier rule (including other
   !important rules of equal specificity) that failed to darken the open
   dropdown list. Targets every element inside the popover/listbox/option
   by attribute only, with no dependency on nesting under stSelectbox,
   since the popover renders in its own layer, not inside the select box. */
div[data-baseweb="popover"],
div[data-baseweb="menu"],
ul[role="listbox"] {
    background: #0c3446 !important;
    background-image: linear-gradient(135deg,#0c3446,#19352a) !important;
}
div[data-baseweb="popover"] *,
div[data-baseweb="menu"] *,
ul[role="listbox"] *,
li[role="option"] {
    color: #f4faf7 !important;
    background: transparent !important;
    opacity: 1 !important;
}
li[role="option"]:hover,
li[role="option"][aria-selected="true"] {
    background: #1f4432 !important;
    color: #ffffff !important;
}
/* Closed-box selected value + placeholder text */
div[data-testid="stSelectbox"] div[data-baseweb="select"] div,
div[data-testid="stSelectbox"] div[data-baseweb="select"] span {
    color: #f4faf7 !important;
}
div[data-testid="stSelectbox"],
div[data-testid="stSelectbox"] *,
div[data-testid="stSelectbox"] div[data-baseweb="select"],
div[data-testid="stSelectbox"] div[data-baseweb="select"] * {
    opacity: 1 !important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"] div,
div[data-testid="stSelectbox"] div[data-baseweb="select"] span {
    color: #f4faf7 !important;
}
div[data-testid="stSelectbox"] svg,
div[data-testid="stSelectbox"] svg {
    filter: brightness(0) invert(1) !important;
    opacity: 1 !important;
}

</style>
""", unsafe_allow_html=True)

# SESSION STATE
# Nothing is pre-picked on a fresh load — budget, arrival time, and energy
# preference all start as None so no card looks selected until the person
# actually clicks one. The generate button is gated on all three being set
# (see the "Please select..." checks further down).
for k, v in {
    "itinerary":"", "chat_messages":[], "chat_history":[],
    "weather_city":"Colombo", "weather_data":None,
    "goal_eval":None, "place_names":[], "generated":False, "show_demo_fallback":False,
    "interests_set":set(), "arrival_time":None, "energy_pref":None,
    "budget_choice":None, "shs_month":None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# INTEREST DATA
@st.cache_data
def img_to_b64(path):
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()

def img_to_b64_safe(path):
    """Same as img_to_b64 but returns None instead of crashing when the
    file isn't there yet — lets the 'What's Happening' month cards render
    with a friendly placeholder until real photos are dropped into
    images/months/, rather than taking the whole app down."""
    try:
        with open(path, "rb") as f:
            mime = "image/jpeg" if path.lower().endswith((".jpg", ".jpeg", ".jfif")) else \
                   "image/png" if path.lower().endswith(".png") else "image/jpeg"
            return f"data:{mime};base64," + base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return None

# Some months are known by their festival name rather than the month
# itself (e.g. December's photos might be filed as "dec1.jpg" OR
# "christmas1.jpg", May's as "may1.jpg" OR "vesak1.jpg", August's as
# "aug1.jpg" OR "esala1.jpg"). Checking every alias means photos can be
# dropped in with whatever name feels natural, no renaming required.
MONTH_FILENAME_ALIASES = {
    "JAN": ["jan", "duruthu"],
    "FEB": ["feb", "independence", "navam"],
    "MAR": ["mar"],
    "APR": ["apr", "avurudu", "newyear"],
    "MAY": ["may", "vesak"],
    "JUN": ["jun", "poson"],
    "JUL": ["jul", "kite"],
    "AUG": ["aug", "esala", "perahera"],
    "SEP": ["sep"],
    "OCT": ["oct", "deepavali"],
    "NOV": ["nov", "whale"],
    "DEC": ["dec", "christmas", "xmas"],
}

def load_month_images(month_key, max_images=6):
    """Finds every photo on disk for a given month, trying each known
    alias and file extension, e.g. images/months/esala1.jpg,
    images/months/esala2.jpg, images/months/vesak2.jfif ... Returns a
    list of base64 data-URIs, in whatever order they're found (1, 2, 3...).
    Empty list if nothing's been added for that month yet."""
    found = []
    aliases = MONTH_FILENAME_ALIASES.get(month_key, [month_key.lower()])
    exts = ["jpg", "jpeg", "png", "jfif", "webp"]
    for alias in aliases:
        # numbered: esala1.jpg, esala2.jpg, ...
        for n in range(1, max_images + 1):
            for ext in exts:
                b64 = img_to_b64_safe(f"images/months/{alias}{n}.{ext}")
                if b64:
                    found.append(b64)
                    break
        # unnumbered single file: dec.jpg
        for ext in exts:
            b64 = img_to_b64_safe(f"images/months/{alias}.{ext}")
            if b64:
                found.append(b64)
                break
    return found

INTEREST_PHOTOS = {
    "Beaches":            img_to_b64("images/beach.jpg"),
    "Hiking":             img_to_b64("images/hiking.jpg"),
    "Nature":             img_to_b64("images/nature.jpg"),
    "Photography":        img_to_b64("images/photography.jpg"),
    "History & Culture":  img_to_b64("images/history.jpg"),
    "Wildlife":           img_to_b64("images/wildlife.jpg"),
    "Food & Cuisine":     img_to_b64("images/food.jpg"),
    "Relaxation":         img_to_b64("images/relaxation.jpg"),
}

# WHAT'S HAPPENING IN SRI LANKA — month-by-month highlights.
# Drop a photo at images/months/<key-lowercase>.jpg (e.g. images/months/may.jpg)
# for any month and it'll show automatically; until then that month's card
# just shows the icon instead, so the app never breaks waiting on assets.
MONTH_HIGHLIGHTS = {
    "JAN": {"name": "Duruthu Perahera",        "where": "Kelaniya",        "icon": "🐘",
            "desc": "The year's first major perahera lights up Kelaniya Temple. Whale-watching season also opens in Mirissa."},
    "FEB": {"name": "Independence Day",         "where": "Colombo",        "icon": "🎉",
            "desc": "National celebrations on the 4th, plus the colourful Navam Perahera along Colombo's lakefront."},
    "MAR": {"name": "Dry Season Peaks",         "where": "South Coast",    "icon": "☀️",
            "desc": "Some of the clearest skies of the year on the south coast — prime time for beaches and surfing."},
    "APR": {"name": "Sinhala & Tamil New Year", "where": "Island-wide",    "icon": "🎊",
            "desc": "The island's biggest cultural holiday — games, sweets, and water-cutting rituals in every town."},
    "MAY": {"name": "Vesak Season",             "where": "Across Sri Lanka","icon": "🏮",
            "desc": "Experience illuminated streets and beautiful lantern displays for the Buddha's birth, enlightenment and passing."},
    "JUN": {"name": "Poson Poya",               "where": "Anuradhapura & Mihintale", "icon": "🕉️",
            "desc": "Marks the arrival of Buddhism in Sri Lanka — Mihintale draws huge crowds of white-clad pilgrims."},
    "JUL": {"name": "Kite Season Begins",       "where": "Galle Face, Colombo", "icon": "🪁",
            "desc": "Steady southwest winds bring out kite flyers along the coast, and Esala Perahera prep kicks off in Kandy."},
    "AUG": {"name": "Kandy Esala Perahera",     "where": "Kandy",          "icon": "🐘",
            "desc": "Sri Lanka's grandest festival — decorated elephants, dancers and drummers parade for ten straight nights."},
    "SEP": {"name": "Quiet Season",             "where": "West & South Coasts", "icon": "🌦️",
            "desc": "Monsoon shift makes the west/south coasts quieter — a good month for fewer crowds and lower prices."},
    "OCT": {"name": "Deepavali",                "where": "Island-wide",    "icon": "🪔",
            "desc": "The Festival of Lights is celebrated by Tamil communities with oil lamps, sweets and fireworks."},
    "NOV": {"name": "Whale Watching Opens",     "where": "Trincomalee",    "icon": "🐋",
            "desc": "East-coast whale watching season begins as the seas calm — blue whales and sperm whales are regular sightings."},
    "DEC": {"name": "Christmas & Peak Season",  "where": "Island-wide",    "icon": "🎄",
            "desc": "Festive lights in Colombo, Unduwap Poya, and the start of peak season as Yala safari season reopens."},
}

# ARRIVAL TIME CONFIG
ARRIVAL_OPTIONS = {
    "morning": {
        "label":  "Morning",
        "time":   "Before 12 pm",
        "advice": "Great timing! Day 1 can head straight to any destination — Mirissa, Sigiriya, Ella. No need to stop near the airport.",
        "cls":    "advice-morning",
    },
    "afternoon": {
        "label":  "Afternoon",
        "time":   "12 pm – 6 pm",
        "advice": "Not enough time to travel far. Day 1 will be a relaxed arrival in Negombo (15 min from airport). Journey begins Day 2.",
        "cls":    "advice-afternoon",
    },
    "evening": {
        "label":  "Evening",
        "time":   "6 pm – 10 pm",
        "advice": "Too late for long travel. Day 1 is a short transfer to Negombo — check in and rest. Exploring starts Day 2.",
        "cls":    "advice-evening",
    },
    "night": {
        "label":  "Night",
        "time":   "After 10 pm",
        "advice": "Straight to bed! Day 1 is a pure rest night in Negombo or Katunayake. The adventure begins properly on Day 2.",
        "cls":    "advice-night",
    },
}

# HERO
# Cinematic Ken Burns crossfade using the app's own curated destination shots
# (already bundled as base64 for INTEREST_PHOTOS) instead of a single static
# external image — no extra assets needed, no external dependency, and it
# visually reinforces "330+ destinations" with variety rather than one photo.
HERO_BG_IMAGES = [
    img_to_b64("images/111.jpg"),
    img_to_b64("images/22222.jpg"),
    img_to_b64("images/333.jpg"),
    img_to_b64("images/123.jpg"),
    img_to_b64("images/1234.jpg"),
    img_to_b64("images/555.jpg"),]
_hero_layers = "".join(
    f'<div class="hero-bg-layer" style="background-image:url({src});animation-delay:{-(i * 6)}s;"></div>'
    for i, src in enumerate(HERO_BG_IMAGES)
)

st.markdown(f"""
<div class="hero-wrap" role="img" aria-label="Scenes from Sri Lanka">
  {_hero_layers}
  <div class="hero-overlay"></div>
  <div class="hero-content">
    <img class="hero-badge" src="{FLAG_ICON}" alt="Sri Lanka Flag">
    <div class="hero-text"><img class="hero-logo" src="{LOGO_ICON}" alt="WaLKer"><p>Your AI Travel Companion for Sri Lanka</p></div>
    <div class="hero-stats">
      <div class="hero-stat"><div class="hero-stat-num">330+</div><div class="hero-stat-lbl">destinations</div></div>
      <div class="hero-stat"><div class="hero-stat-num">AI</div><div class="hero-stat-lbl">powered</div></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:8px 0 16px;">
      <div style="display:flex;align-items:center;justify-content:center;gap:8px;">
        <img src="{FLAG_ICON}" alt="Sri Lanka Flag" style="height:44px;width:auto;">
      </div>
      <div style="font-family:'Cambria',monospace;font-size:0.58rem;letter-spacing:1.5px;text-transform:uppercase;color:#5a7a6a;margin-top:4px;">{get_total_trips()} trips planned</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown('<span class="mono-label">🌤 Live Weather</span>', unsafe_allow_html=True)
    weather_input = st.text_input("city_search", value=st.session_state.weather_city,
        placeholder="Kandy, Galle, Ella...", label_visibility="collapsed")
    if st.button("Search Weather", key="btn_weather"):
        if weather_input.strip():
            st.session_state.weather_city = weather_input.strip()
            st.session_state.weather_data = get_weather(weather_input.strip())

    if st.session_state.weather_data is None:
        st.session_state.weather_data = get_weather(st.session_state.weather_city)

    wd = st.session_state.weather_data
    if wd and wd.get("success"):
        icon_map = {"Clear":"☀️","Clouds":"⛅","Rain":"🌧️","Drizzle":"🌦️","Thunderstorm":"⛈️","Snow":"❄️","Mist":"🌫️","Haze":"🌫️","Fog":"🌫️"}
        icon = icon_map.get(wd.get("icon",""), "🌡️")
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0c3446,#19352a);border:1px solid rgba(61,186,126,0.2);border-radius:14px;padding:16px 20px;margin-bottom:14px;">
          <p style="font-family:'Cambria',serif;font-size:1.1rem;font-weight:600;color:#e8f0ec;margin:0;">{icon} {wd['city']}, {wd['country']}</p>
          <p style="font-family:'Cambria',serif;font-size:2.4rem;font-weight:400;color:#3dba7e;line-height:1;margin:4px 0;">{wd['temp']}°C</p>
          <p style="font-family:'Cambria',monospace;font-size:0.7rem;color:#a8bfb3;margin:0;">{wd['description']} · feels {wd['feels_like']}°C</p>
          <div style="display:flex;gap:14px;margin-top:10px;padding-top:10px;border-top:1px solid rgba(255,255,255,0.08);">
            <div style="font-size:0.72rem;color:#5a7a6a;">Humidity<br><span style="color:#e8f0ec;font-weight:500;">{wd['humidity']}%</span></div>
            <div style="font-size:0.72rem;color:#5a7a6a;">Wind<br><span style="color:#e8f0ec;font-weight:500;">{wd['wind']} m/s</span></div>
          </div>
        </div>""", unsafe_allow_html=True)
    elif wd and not wd.get("success"):
        st.warning(f"⚠️ {wd.get('error','City not found')}")

    if wd and wd.get("success"):
        advisory = get_weather_advisory(wd)
        if advisory:
            st.markdown(f"""
            <div style="background:#2a2410;border:1px solid rgba(232,184,75,0.3);
                        border-radius:10px;padding:10px 14px;margin:-4px 0 14px;font-size:0.75rem;
                        line-height:1.5;color:#e8d9a8;">
              {advisory}
            </div>""", unsafe_allow_html=True)

    # ── Currency Converter ──────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<span class="mono-label">Currency Converter</span>', unsafe_allow_html=True)

    if "exchange_rates" not in st.session_state:
        st.session_state.exchange_rates = get_exchange_rates("USD")
    rates = st.session_state.exchange_rates

    conv_col1, conv_col2 = st.columns([2, 3])
    with conv_col1:
        conv_currency = st.selectbox("conv_cur", ["USD", "EUR", "GBP"], label_visibility="collapsed", key="conv_currency")
    with conv_col2:
        conv_amount = st.number_input("conv_amt", min_value=0.0, value=100.0, step=10.0, label_visibility="collapsed", key="conv_amount")

    if rates.get("success"):
        # rates fetched with base=USD, so convert conv_currency -> USD first if needed
        to_usd = conv_amount if conv_currency == "USD" else conv_amount / rates.get(conv_currency, 1)
        lkr_value = to_usd * rates["LKR"]
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0c3446,#19352a);border:1px solid rgba(61,186,126,0.2);border-radius:10px;
                    padding:10px 14px;font-size:0.85rem;">
        <span style="color:#a8bfb3;">≈</span> <span style="color:#e8f0ec;font-weight:600;font-size:1.1rem;">LKR {lkr_value:,.0f}</span>
        </div>""", unsafe_allow_html=True)
        as_of = rates.get("as_of")
        if as_of:
            st.markdown(f'<span style="font-size:0.66rem;color:#5a7a6a;">Rate as of {as_of}</span>', unsafe_allow_html=True)
    else:
        st.caption(f"⚠️ Live rate unavailable — showing approximate (1 USD ≈ LKR {rates['LKR']:.0f}). Verify before travel.")
        to_usd = conv_amount if conv_currency == "USD" else conv_amount / rates.get(conv_currency, 1)
        st.markdown(f'<span style="color:#a8bfb3;">≈ LKR {to_usd * rates["LKR"]:,.0f} (approx.)</span>', unsafe_allow_html=True)

    # ── Refresh button, now below everything ──
    if st.button("↻ Refresh rate", key="btn_refresh_rates", help="Fetch latest rates"):
        st.session_state.exchange_rates = get_exchange_rates("USD")
        st.rerun()
        
    # ── Trending Destinations (real data from your own trip history) ──
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<span class="mono-label">Trending Destinations</span>', unsafe_allow_html=True)
    trending = get_destination_frequency()
    if not trending:
        st.caption("Not enough trip data yet — trends appear as more trips are planned.")
    else:
        chips = " ".join(
            f'<span class="chip chip-g" style="margin:2px;">{name} · {count}</span>'
            for name, count in trending[:6]
        )
        st.markdown(f'<div class="chip-row">{chips}</div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<span class="mono-label">🕐 Past Trips</span>', unsafe_allow_html=True)
    recent = get_recent_trips(15)
    if not recent:
        st.caption("No trips yet — plan your first one!")
    else:
        # A native selectbox with 15 options triggers a confirmed Streamlit
        # rendering bug where the hover/keyboard-highlighted row in the
        # scrollable dropdown shows a blank white bar with no visible text
        # (streamlit/streamlit#10204) — no CSS override fixes it reliably,
        # since the highlight is drawn on an element outside what our
        # selectors can reach. A bounded, normal-scrolling list of real
        # buttons sidesteps the bug entirely and doubles as a one-click
        # load instead of "pick then press Load".
        def _trip_option_label(idx, row):
            days_r, interests_r, budget_r, timestamp_r, _ = row
            date_str = timestamp_r[:10] if timestamp_r else "unknown date"
            short_interests = interests_r if len(interests_r) <= 32 else interests_r[:32] + "…"
            return f"{date_str} · {days_r}d · {budget_r.split()[0]} · {short_interests}"

        with st.container(height=220, key="past_trips_list"):
            for i, row in enumerate(recent):
                label = _trip_option_label(i, row)
                if st.button(label, key=f"past_trip_{i}", use_container_width=True):
                    itinerary_r = row[4]
                    st.session_state.itinerary     = itinerary_r
                    st.session_state.generated     = True
                    st.session_state.place_names   = extract_place_names(itinerary_r)
                    st.session_state.goal_eval     = check_goal_achievement(itinerary_r)
                    st.session_state.chat_messages = []
                    st.session_state.chat_history  = []
                    st.rerun()

# FORM — Left + Right columns
form_left, form_right = st.columns([5, 7], gap="large")

# LEFT COLUMN
with form_left, st.container(key="form_left_panel"):
    st.markdown('<div class="sec-title">Plan Your Trip</div>', unsafe_allow_html=True)
    st.markdown('<span class="sec-sub">Tell the AI agent about your journey</span>', unsafe_allow_html=True)

    # ── Days slider ────────────────────────────
    st.markdown('<span class="panel-label">How many days?</span>', unsafe_allow_html=True)
    days = st.slider("days", 1, 21, 7, label_visibility="collapsed")

    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

    # ── Budget card ────────────────────────────
    st.markdown("""
    <div style="margin-bottom:8px;margin-top:6px;">
      <span class="panel-label">Your Budget?</span>
    </div>
    """, unsafe_allow_html=True)

    BUDGET_OPTIONS = {
        "Budget":   {"sub": "Under USD 50/day",  "key": "Budget (Under USD 50/day)"},
        "Mid-range":{"sub": "USD 50–150/day",    "key": "Mid-range (USD 50-150/day)"},
        "Luxury":   {"sub": "Over USD 150/day",      "key": "Luxury (Over USD 150/day)"},
    }

    budget_cols = st.columns(3, gap="small")
    for col, (label, opt) in zip(budget_cols, BUDGET_OPTIONS.items()):
        is_active = st.session_state.budget_choice == opt["key"]
        btn_key = f"budget_{label}"
        with col:
            # A single authoritative rule is injected for THIS card on every
            # render (not just when selected) so there's never a fight
            # between "the selected override" and "the base card style" in
            # two different <style> blocks — one rule always wins because
            # it's the only one describing this exact button's current
            # state. The class is chained 3x purely to inflate specificity
            # past anything else on the page without needing an ID.
            sel = f".st-key-{btn_key}" * 3
            if is_active:
                bg, border, color, weight, shadow = (
                    "linear-gradient(135deg,#264f47,#1a3d38)", "2px solid #1f6b47",
                    "#fff", "700",
                    "0 0 0 3px rgba(31,107,71,0.35), 0 0 26px rgba(31,107,71,0.6), 0 0 52px rgba(31,107,71,0.28), 0 6px 18px rgba(0,0,0,0.35)")
            else:
                bg, border, color, weight, shadow = (
                    "linear-gradient(135deg,#0c3446,#19352a)", "2px solid rgba(255,255,255,0.32)",
                    "#c3d4cb", "700",
                    "0 0 14px rgba(47,139,184,0.10), 0 4px 16px rgba(0,0,0,0.4)")
            st.markdown(f"""
            <style>
            {sel} button {{
                background: {bg} !important;
                border: {border} !important;
                border-radius: 12px !important;
                height: 92px !important;
                width: 100% !important;
                display: flex !important;
                flex-direction: column !important;
                align-items: center !important;
                justify-content: center !important;
                gap: 4px !important;
                white-space: pre-line !important;
                line-height: 1.45 !important;
                color: {color} !important;
                font-family: 'Cambria', serif !important;
                font-size: 0.85rem !important;
                font-weight: {weight} !important;
                box-shadow: {shadow} !important;
                transform: none !important;
                transition: border-color 0.15s, background 0.15s !important;
            }}
            {sel} button p {{
                font-family: 'Cambria', serif !important;
                font-size: 0.95rem !important;
                line-height: 1.45 !important;
            }}
            {"" if is_active else f'''{sel} button:hover {{
                box-shadow: 0 0 20px rgba(61,186,126,0.28), 0 4px 18px rgba(0,0,0,0.4) !important;
                border-color: rgba(61,186,126,0.7) !important;
            }}'''}
            </style>
            """, unsafe_allow_html=True)
            btn_label = f"{label}\n{opt['sub']}"
            if st.button(btn_label, key=btn_key, use_container_width=True):
                st.session_state.budget_choice = opt["key"]
                st.rerun()

    budget = st.session_state.budget_choice
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

# ── Arrival Time ──────────────────────
    st.markdown("""
    <div style="margin-bottom:10px;margin-top:6px;">
      <span class="panel-label">When do you arrive?</span><br>
      <span class="panel-sublabel">Your arrival time affects Day 1 planning</span>
    </div>
    """, unsafe_allow_html=True)
    selected_arrival = st.session_state.arrival_time
    arr_keys = list(ARRIVAL_OPTIONS.keys())

# Row 1: Morning, Afternoon · Row 2: Evening, Night
    for row_keys in (arr_keys[:2], arr_keys[2:]):
        row_cols = st.columns(2, gap="small")
        for col, key in zip(row_cols, row_keys):
            opt = ARRIVAL_OPTIONS[key]
            is_active = selected_arrival == key
            with col:
                # Same always-render, specificity-stacked technique as the
                # budget cards — one authoritative rule per card per render,
                # instead of a conditional "selected" rule trying to out-
                # specificity a separate base rule in a different <style>
                # block (that fight was silently losing, leaving selected
                # cards with zero styling at all).
                sel = f".st-key-arr_{key}" * 3
                if is_active:
                    bg, border, color, weight, shadow = (
                        "linear-gradient(135deg,#264f47,#1a3d38)", "2px solid #1f6b47",
                        "#fff", "700",
                        "0 0 0 3px rgba(31,107,71,0.35), 0 0 26px rgba(31,107,71,0.6), 0 0 52px rgba(31,107,71,0.28), 0 6px 18px rgba(0,0,0,0.35)")
                else:
                    bg, border, color, weight, shadow = (
                        "linear-gradient(135deg,#0c3446,#19352a)", "2px solid rgba(255,255,255,0.32)",
                        "#c3d4cb", "700",
                        "0 0 14px rgba(47,139,184,0.10), 0 4px 16px rgba(0,0,0,0.4)")
                st.markdown(f"""
                <style>
                {sel} button {{
                    background: {bg} !important;
                    border: {border} !important;
                    border-radius: 12px !important;
                    padding: 10px 4px 8px !important;
                    width: 100% !important;
                    height: 86px !important;
                    display: flex !important;
                    flex-direction: column !important;
                    align-items: center !important;
                    justify-content: center !important;
                    gap: 3px !important;
                    color: {color} !important;
                    font-family: 'Cambria', serif !important;
                    font-size: 0.85rem !important;
                    font-weight: {weight} !important;
                    box-shadow: {shadow} !important;
                    transform: none !important;
                    transition: border-color 0.15s, background 0.15s !important;
                    white-space: pre-line !important;
                    line-height: 1.4 !important;
                    text-align: center !important;
                }}
                {sel} button p {{
                    font-family: 'Cambria', serif !important;
                    font-size: 0.85rem !important;
                    line-height: 1.4 !important;
                }}
                {"" if is_active else f'''{sel} button:hover {{
                    box-shadow: 0 0 20px rgba(61,186,126,0.28), 0 4px 18px rgba(0,0,0,0.4) !important;
                    border-color: rgba(61,186,126,0.7) !important;
                }}'''}
                </style>
                """, unsafe_allow_html=True)
                tick = "✅" if is_active else ""
                btn_label = f"{tick}\n{opt['label']}\n{opt['time']}" if tick else f"{opt['label']}\n{opt['time']}"
                if st.button(btn_label, key=f"arr_{key}", use_container_width=True):
                    st.session_state.arrival_time = key
                    st.rerun()

    # Advice box — only once an arrival time is actually picked
    if selected_arrival:
        opt = ARRIVAL_OPTIONS[selected_arrival]
        st.markdown(f"""
          <div class="arrival-advice {opt['cls']}">
            {opt['advice']}
          </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="hint-text">Pick an arrival time above to see Day 1 advice.</div>',
            unsafe_allow_html=True,
        )

    # ── Energy preference (follow-up to arrival time) ──────────
    # Arrival TIME says what's physically possible on Day 1; this says what the
    # traveler actually WANTS — someone can land at 9am and still be jet-lagged,
    # or land at 6pm and still want to get moving. Kept as a separate choice
    # rather than folded into arrival time so both can vary independently.
    # Nothing here can be resolved until an arrival time is chosen, since the
    # available options depend on it.
    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
    st.markdown('<span class="panel-label">How do you want to start Day 1?</span>', unsafe_allow_html=True)

    # Options depend on arrival time — e.g. "Start right away" only makes
    # sense for a morning arrival — but the dropdown itself should always be
    # visible with a real placeholder, rather than swapping in plain text
    # before an arrival time is picked. Falls back to the full option set
    # until an arrival time narrows it down.
    current_energy_options = get_energy_options(selected_arrival) if selected_arrival else dict(ENERGY_OPTIONS)
    energy_keys    = list(current_energy_options.keys())
    energy_labels  = [f"{current_energy_options[k]['label']} — {current_energy_options[k]['desc']}" for k in energy_keys]
    PLACEHOLDER    = "Select how you'd like to start Day 1..."

    # If the previously picked option no longer applies after the arrival
    # time changed (e.g. switched to "night"), or nothing's been picked
    # yet, fall back to no selection instead of raising a ValueError.
    if st.session_state.energy_pref not in energy_keys:
        st.session_state.energy_pref = None

    if selected_arrival and len(energy_keys) == 1:
        # Night arrivals: there's no real choice — Day 1 is always a rest
        # night — so just state that instead of showing a pointless dropdown.
        only = current_energy_options[energy_keys[0]]
        st.session_state.energy_pref = energy_keys[0]
        st.markdown(
            f'<div style="font-size:0.78rem;color:#0c3446 !important;padding:8px 0;">'
            f'😴 {only["label"]} — {only["desc"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        # index=None + placeholder is Streamlit's native way to show grey
        # placeholder text with nothing selected, without the placeholder
        # ever being a real, pickable row in the options list.
        current_idx = (
            energy_keys.index(st.session_state.energy_pref)
            if st.session_state.energy_pref in energy_keys else None
        )
        picked_label = st.selectbox(
            "energy_pref", energy_labels, index=current_idx,
            placeholder=PLACEHOLDER, label_visibility="collapsed",
        )
        if picked_label is None:
            st.session_state.energy_pref = None
        else:
            st.session_state.energy_pref = energy_keys[energy_labels.index(picked_label)]

    # ── Travel month (drives seasonal/climate/festival guidance) ──────
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    st.markdown('<span class="panel-label">Which month are you travelling?</span>', unsafe_allow_html=True)
    month_options = ["Not sure yet"] + list(SRI_LANKA_MONTHLY_GUIDE.keys())
    travel_month_pick = st.selectbox("travel_month", month_options, label_visibility="collapsed")
    travel_month = None if travel_month_pick == "Not sure yet" else travel_month_pick

    if travel_month:
        g = SRI_LANKA_MONTHLY_GUIDE[travel_month]
        events_this_month = [e for e in SRI_LANKA_FIXED_EVENTS if travel_month[:3] in e["date"]]
        events_html = "".join(
            f'<div style="margin-top:6px;"><b style="color:#e8b84b;">{e["name"]}</b> '
            f'<span style="color:#5a7a6a;">({e["date"]})</span><br>'
            f'<span style="color:#a8bfb3;">{e["note"]}</span></div>'
            for e in events_this_month
        )
        st.markdown(f"""
        <div style="background:#1a3528;border:1px solid rgba(61,186,126,0.2);
                    border-radius:10px;padding:10px 14px;margin-top:6px;font-size:0.78rem;line-height:1.6;">
          <span style="color:#3dba7e;">✓ Good weather:</span> <span style="color:#e8f0ec;">{', '.join(g['good_for'])}</span><br>
          <span style="color:#e87060;">⚠ Avoid / rainy:</span> <span style="color:#e8f0ec;">{', '.join(g['avoid'])}</span><br>
          <span style="color:#5a7a6a;">{g['note']}</span>
          {events_html}
        </div>
        """, unsafe_allow_html=True)

    # ── Output language ────────────────────────────────────────────────
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    st.markdown('<span class="panel-label">Which language should the itinerary be written in?</span>', unsafe_allow_html=True)
    travel_language = st.selectbox("travel_language", SUPPORTED_LANGUAGES, label_visibility="collapsed")

    # ── Extra info ─────────────────────────────
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    st.markdown('<span class="panel-label">Anything extra?</span>', unsafe_allow_html=True)
    extra_info = st.text_area("extra",
        placeholder="e.g. travelling with family, honeymoon, starting from Colombo...",
        height=90, label_visibility="collapsed")


st.markdown("""
<style>
/* Hide default button styling */
div.stButton > button {
    background: transparent !important;
    border: none !important;
    color: #e8f0ec !important;
    font-family: 'Cambria', monospace !important;
    font-size: 0.75rem !important;
    text-align: center;
    padding: 6px 0px;
}

/* Remove hover green effect */
div.stButton > button:hover {
    background: transparent !important;
    color: #ffffff !important;
}

/* Remove focus outline */
div.stButton > button:focus {
    outline: none !important;
    box-shadow: none !important;
}
</style>
""", unsafe_allow_html=True)

# RIGHT COLUMN
with form_right, st.container(key="form_right_panel"):
    st.markdown("""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:2px;">
      <span class="panel-label">What are your interests?</span>
    </div>
    <span class="panel-caption">Click a card to select · click again to deselect</span>
    """, unsafe_allow_html=True)

    interest_labels = list(INTEREST_PHOTOS.keys())
    rows = [interest_labels[i:i+4] for i in range(0, len(interest_labels), 4)]

    for row in rows:
        cols = st.columns(4, gap="small")
        for col, label in zip(cols, row):
            with col:
                selected  = label in st.session_state.interests_set
                img_url   = INTEREST_PHOTOS[label]
                safe_key  = re.sub(r'[^a-zA-Z0-9]+', '', label)
                border_c  = "#1f6b47" if selected else "rgba(255,255,255,0.32)"
                bg_c      = "#264f47" if selected else "linear-gradient(135deg,#0c3446,#19352a)"
                tick      = "✓ " if selected else ""

                # Image + label rendered as one continuous card: the wrapping
                # container's own key ("intcard_<label>") gives a real CSS
                # hook to zero out the default gap Streamlit puts between the
                # markdown block and the button block below it, and both
                # pieces share the same border colour/radius on the seam so
                # they read as a single frame instead of two stacked boxes.
                int_sel = f".st-key-int_{safe_key}" * 3
                glow = (
                    "0 0 26px rgba(31,107,71,0.6), 0 0 50px rgba(31,107,71,0.3), 0 6px 16px rgba(0,0,0,0.35)"
                    if selected else
                    "0 0 14px rgba(61,186,126,0.10), 0 4px 14px rgba(0,0,0,0.4)"
                )
                with st.container(key=f"intcard_{safe_key}"):
                    st.markdown(f"""
                    <style>
                    .st-key-intcard_{safe_key} div[data-testid="stVerticalBlock"] {{
                        gap: 0 !important;
                    }}
                    {int_sel} button {{
                        background: {bg_c} !important;
                        border: 2px solid {border_c} !important;
                        border-top: none !important;
                        border-radius: 0 0 10px 10px !important;
                        color: {'#fff' if selected else '#c3d4cb'} !important;
                        font-weight: 700 !important;
                        font-family: 'Cambria', serif !important;
                        box-shadow: none !important;
                        width: 100% !important;
                        transition: border-color 0.15s, background 0.15s !important;
                    }}
                    {int_sel} button p {{
                        font-family: 'Cambria', serif !important;
                        font-size: 0.85rem !important;
                    }}
                    .st-key-intcard_{safe_key} {{
                        border-radius: 10px !important;
                        box-shadow: {glow} !important;
                        transition: box-shadow 0.15s !important;
                    }}
                    {"" if selected else f'''.st-key-intcard_{safe_key}:hover {{
                        box-shadow: 0 0 20px rgba(61,186,126,0.28), 0 4px 16px rgba(0,0,0,0.4) !important;
                    }}'''}
                    </style>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div style="border:2px solid {border_c};border-bottom:none;border-radius:10px 10px 0 0;
                                overflow:hidden;background:{bg_c};">
                      <img src="{img_url}"
                           style="width:100%;height:200px;object-fit:cover;display:block;
                                  opacity:{'1' if selected else '0.65'};" />
                    </div>
                    """, unsafe_allow_html=True)

                    btn_text = f"{tick}{label}"
                    if st.button(btn_text, key=f"int_{safe_key}", use_container_width=True):
                        if label in st.session_state.interests_set:
                            st.session_state.interests_set.discard(label)
                        else:
                            st.session_state.interests_set.add(label)
                        st.rerun()

    # Selected chips + AI style decision
    interests_selected = list(st.session_state.interests_set)
    if interests_selected:
        chips = " ".join(
            f'<span class="chip chip-g">{i}</span>'
            for i in interests_selected
        )
        st.markdown(f'<div class="chip-row">{chips}</div>', unsafe_allow_html=True)
        if budget:
            style_dec = decide_travel_style(interests_selected, budget)
            st.markdown(f"""
            <div class="ai-box">
              <strong>🤖 AI Decision:</strong> {style_dec['stay']} · {style_dec['pace']} pace
              · {style_dec['cost_level']} cost · {style_dec['focus']}
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="ai-box">🤖 Pick a budget above and I\'ll tell you the travel style.</div>',
                unsafe_allow_html=True,
            )

    # ── What's Happening in Sri Lanka? ──────────────────────────
    # Standalone month browser — has nothing to do with the "Which month
    # are you travelling?" field above; it's just a browsable calendar of
    # festivals/events. Defaults to the CURRENT real-world month on first
    # load so there's always something on screen without the user having
    # to click anything, and every month button stays clickable to browse.
    if st.session_state.shs_month is None:
        st.session_state.shs_month = datetime.date.today().strftime("%b").upper()

    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:2px;">
      <span class="panel-label">🇱🇰 What's Happening in Sri Lanka?</span>
    </div>
    <span class="panel-caption">Click a month to see festivals &amp; events</span>
    """, unsafe_allow_html=True)

    month_keys = list(MONTH_HIGHLIGHTS.keys())
    month_rows = [month_keys[i:i+6] for i in range(0, len(month_keys), 6)]

    for row in month_rows:
        cols = st.columns(6, gap="small")
        for col, mkey in zip(cols, row):
            with col:
                is_active = st.session_state.shs_month == mkey
                # Same tripled-selector specificity trick as the arrival-time
                # cards (proven to work) — a single plain class selector was
                # silently losing to Streamlit's own button defaults.
                sel = f".st-key-shsm_{mkey}" * 3
                st.markdown(f"""
                <style>
                {sel} button {{
                    background: {'linear-gradient(135deg,#123a4d,#1f4432)' if is_active else 'linear-gradient(135deg,#0c3446,#19352a)'} !important;
                    border: 2px solid {'#3dba7e' if is_active else 'rgba(255,255,255,0.32)'} !important;
                    color: #ffffff !important;
                    font-weight: 700 !important;
                    font-family: 'Cambria', monospace !important;
                    border-radius: 8px !important;
                }}
                {sel} button p {{
                    color: #ffffff !important;
                    font-family: 'Cambria', monospace !important;
                }}
                </style>
                """, unsafe_allow_html=True)
                if st.button(mkey, key=f"shsm_{mkey}", use_container_width=True):
                    st.session_state.shs_month = mkey
                    st.rerun()

    picked = MONTH_HIGHLIGHTS[st.session_state.shs_month]
    month_imgs = load_month_images(st.session_state.shs_month)

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    if month_imgs:
        # Hero = whichever photo is currently selected; extras render as a
        # row of real clickable thumbnails right below (the thumbnails
        # themselves are buttons — no separate "Photo 1/2/3" label needed).
        hero_idx_key = f"shs_hero_{st.session_state.shs_month}"
        if hero_idx_key not in st.session_state:
            st.session_state[hero_idx_key] = 0
        hero_idx = min(st.session_state[hero_idx_key], len(month_imgs) - 1)

        st.markdown(f"""
        <div style="border:2px solid rgba(61,186,126,0.4);border-radius:12px;overflow:hidden;
                    background:linear-gradient(135deg,#0c3446,#19352a);">
          <img src="{month_imgs[hero_idx]}"
               style="width:100%;max-height:340px;object-fit:contain;display:block;
                      background:#08202b;" />
          <div style="padding:14px 18px;">
            <div style="font-family:'Cambria',serif;font-size:1.15rem;font-weight:700;color:#ffffff;">
              {picked['icon']} {picked['name']}
            </div>
            <div style="font-family:'Cambria',monospace;font-size:0.68rem;letter-spacing:1px;
                        text-transform:uppercase;color:#3dba7e;margin:4px 0 10px;">📍 {picked['where']}</div>
            <div style="font-size:0.85rem;line-height:1.6;color:#c3d4cb;">{picked['desc']}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Thumbnail row — each one IS the clickable element (background-image
        # on a real st.button, text hidden), not a picture sitting next to a
        # separate button.
        if len(month_imgs) > 1:
            st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
            thumb_cols = st.columns(len(month_imgs), gap="small")
            for i, (col, im) in enumerate(zip(thumb_cols, month_imgs)):
                with col:
                    is_hero = i == hero_idx
                    tsel = f".st-key-shsthumb_{st.session_state.shs_month}_{i}" * 3
                    st.markdown(f"""
                    <style>
                    {tsel} button {{
                        background-image: url('{im}') !important;
                        background-size: cover !important;
                        background-position: center !important;
                        height: 58px !important;
                        min-height: 58px !important;
                        border-radius: 8px !important;
                        border: 2px solid {'#3dba7e' if is_hero else 'rgba(255,255,255,0.3)'} !important;
                        opacity: {'1' if is_hero else '0.65'} !important;
                        padding: 0 !important;
                    }}
                    {tsel} button p {{ opacity: 0 !important; }}
                    </style>
                    """, unsafe_allow_html=True)
                    if st.button(" ", key=f"shsthumb_{st.session_state.shs_month}_{i}", use_container_width=True):
                        st.session_state[hero_idx_key] = i
                        st.rerun()
    else:
        # No photo dropped in for this month yet — still show the info,
        # just without a hero image, so the widget never looks broken.
        st.markdown(f"""
        <div style="border:2px dashed rgba(61,186,126,0.4);border-radius:12px;
                    background:linear-gradient(135deg,#0c3446,#19352a);padding:22px 18px;">
          <div style="font-family:'Cambria',serif;font-size:1.15rem;font-weight:700;color:#ffffff;">
            {picked['icon']} {picked['name']}
          </div>
          <div style="font-family:'Cambria',monospace;font-size:0.68rem;letter-spacing:1px;
                      text-transform:uppercase;color:#3dba7e;margin:4px 0 10px;">📍 {picked['where']}</div>
          <div style="font-size:0.85rem;line-height:1.6;color:#c3d4cb;">{picked['desc']}</div>
          <div style="font-size:0.68rem;color:#5a7a6a;margin-top:12px;">
            💡 Add a photo at images/months/{st.session_state.shs_month.lower()}1.jpg to show it here.
          </div>
        </div>
        """, unsafe_allow_html=True)


# GENERATE BUTTON
interests_selected = list(st.session_state.interests_set)

st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
if st.button("Generate My Itinerary", use_container_width=True, key="btn_generate"):
    missing = []
    if not st.session_state.budget_choice:
        missing.append("a budget")
    if not st.session_state.arrival_time:
        missing.append("an arrival time")
    if st.session_state.arrival_time and not st.session_state.energy_pref:
        missing.append("how you'd like to start Day 1")
    if not interests_selected:
        missing.append("at least one interest")

    if missing:
        st.warning("Please select " + ", ".join(missing) + ".")
    else:
        memory_ctx    = get_smart_memory_context(3)
        interests_str = ", ".join(interests_selected)

        # Streamed generation: tokens appear live in this placeholder as the
        # AI writes the itinerary, instead of a blank spinner for several
        # seconds followed by a wall of text all at once. st.write_stream
        # both renders the chunks and returns the concatenated full text.
        st.markdown("🌴 **Planning your perfect Sri Lanka trip...**")
        stream_box = st.empty()
        with stream_box.container():
            itinerary = st.write_stream(
                plan_trip_stream(
                    days, interests_selected, budget,
                    arrival_time=st.session_state.arrival_time,
                    energy=st.session_state.energy_pref,
                    extra_info=extra_info,
                    memory_context=memory_ctx,
                    travel_month=travel_month,
                    language=travel_language,
                )
            )
        # write_stream can hand back a list of chunks in some Streamlit
        # versions rather than a single joined string — normalise it so the
        # rest of this block always deals with plain text.
        if not isinstance(itinerary, str):
            itinerary = "".join(str(part) for part in itinerary)

        is_error = itinerary.strip().startswith("⚠️")
        goal_eval = {"status": "error", "error": itinerary} if is_error else check_goal_achievement(itinerary)

        if is_error:
            # AI call failed (rate limit / network / auth) — clear the partial
            # stream output and show a friendly message instead of leaving a
            # half-written itinerary on screen.
            stream_box.empty()
            st.error(itinerary)
            st.session_state.show_demo_fallback = True
        else:
            st.session_state.show_demo_fallback = False
            st.session_state.itinerary     = itinerary
            st.session_state.goal_eval     = goal_eval
            st.session_state.generated     = True
            st.session_state.place_names   = extract_place_names(itinerary)
            st.session_state.chat_messages = []
            st.session_state.chat_history  = []
            # Track the trip's DB id so a later refine updates this row
            # instead of inserting a new one each time.
            st.session_state.trip_id = save_trip(days, interests_str, budget, itinerary)
            st.rerun()

# Demo-mode fallback — only surfaces right after a real AI call fails, so a
# live demo/judging session never dead-ends on a spinner or error screen.
# Loads a pre-written sample itinerary with zero network calls.
if st.session_state.get("show_demo_fallback"):
    st.caption("AI service unavailable right now — you can load a sample itinerary to keep the demo moving.")
    if st.button("🎬 Show Demo Example Instead", key="btn_demo_fallback", use_container_width=True):
        itinerary, goal_eval = get_sample_itinerary()
        st.session_state.itinerary          = itinerary
        st.session_state.goal_eval          = goal_eval
        st.session_state.generated          = True
        st.session_state.place_names        = extract_place_names(itinerary)
        st.session_state.chat_messages      = []
        st.session_state.chat_history       = []
        st.session_state.show_demo_fallback = False
        st.rerun()

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# TRIP TOOLKIT — packing list, safety/scams, phrases. Not gated behind
# "generated": every piece of this is local/static data (EMERGENCY_CONTACTS,
# VISA_NOTE, COMMON_TOURIST_SCAMS, ESSENTIAL_PHRASES are constants;
# generate_packing_list() is deterministic from interests+month), none of it
# needs Groq. Collapsed by default so it doesn't compete visually with the
# itinerary — it's one click away, not the first thing on the page.
with st.expander("Trip Toolkit — Packing List, Safety & Scams, Phrases", expanded=False):
    tab_pack, tab_safety, tab_phrases = st.tabs(["Packing List", "Safety & Scams", "Phrases"])

    with tab_pack:
        packing = generate_packing_list(interests_selected, travel_month)
        pack_col1, pack_col2 = st.columns(2)
        with pack_col1:
            st.markdown("**Essentials**")
            for item in packing["essentials"]:
                st.markdown(f"- {item}")
            if packing["seasonal"]:
                st.markdown("**For this season**")
                for item in packing["seasonal"]:
                    st.markdown(f"- {item}")
        with pack_col2:
            if packing["for_your_trip"]:
                st.markdown("**For your selected activities**")
                for item in packing["for_your_trip"]:
                    st.markdown(f"- {item}")
            else:
                st.caption("Select interests above to see activity-specific items.")

    with tab_safety:
        st.markdown("**Emergency contacts**")
        for c in EMERGENCY_CONTACTS:
            st.markdown(f"- **{c['label']}:** {c['value']}")
        st.markdown("**Entry requirements**")
        st.caption(VISA_NOTE)
        st.markdown("**Common tourist scams to know about**")
        for s in COMMON_TOURIST_SCAMS:
            st.markdown(f"- {s}")

    with tab_phrases:
        st.caption("A few basics — Sinhala is spoken island-wide, Tamil mainly in the north/east.")
        for p in ESSENTIAL_PHRASES:
            st.markdown(f"**{p['english']}** — Sinhala: *{p['sinhala']}* · Tamil: *{p['tamil']}*")

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# RESULTS
if not st.session_state.generated:
    st.markdown("""
    <div class="empty-state">
      <div class="empty-state-icon">🗺️</div>
      <div class="empty-state-title">Your itinerary will appear here</div>
      <div class="empty-state-sub">Select interests above and hit Generate to begin.</div>
    </div>""", unsafe_allow_html=True)

else:
    itinerary = st.session_state.itinerary
    display_itinerary, sustainability_matches = annotate_sustainability(itinerary)

    # Goal eval banner
    goal = st.session_state.goal_eval or {}
    if goal:
        sc = {"complete":"#3dba7e","partial":"#e8b84b","incomplete":"#e87060"}.get(goal.get("status",""),"#5a7a6a")
        chips_html = " ".join(
            f'<span class="{"goal-pass" if ok else "goal-fail"}">{"✓" if ok else "✗"} {c}</span>'
            for c, ok in goal.get("checks",{}).items())
        st.markdown(f"""
        <div style="background:#1a3528;border:1px solid rgba(255,255,255,0.1);border-left:4px solid {sc};
                    border-radius:9px;padding:12px 16px;margin-bottom:14px;">
          <div style="font-family:'Cambria',monospace;font-size:0.58rem;letter-spacing:1.5px;text-transform:uppercase;color:{sc};margin-bottom:6px;">Agent Self-Evaluation</div>
          <div style="font-size:0.8rem;color:#a8bfb3;margin-bottom:8px;">{goal.get('label','')}</div>
          <div style="display:flex;gap:5px;flex-wrap:wrap;">{chips_html}</div>
        </div>""", unsafe_allow_html=True)

    if sustainability_matches:
        badge_chips = " ".join(
            f'<span class="chip chip-g" style="margin:2px;">{name}</span>'
            for name, _ in sustainability_matches
        )
        st.markdown(f"""
        <div style="background:rgba(61,186,126,0.08);border:1px solid rgba(61,186,126,0.25);
                    border-radius:9px;padding:10px 14px;margin-bottom:12px;font-size:0.78rem;">
          <span style="color:#3dba7e;font-weight:600;">🌱 This trip includes sustainability-conscious stays:</span>
          <div class="chip-row" style="margin-top:6px;">{badge_chips}</div>
        </div>""", unsafe_allow_html=True)

    # ── Seasonal attractions / local events for places actually in this trip ──
    seasonal_hits = get_seasonal_highlights(st.session_state.get("place_names", []), travel_month)
    if seasonal_hits:
        hits_html = "".join(
            f'<div style="margin-top:6px;"><b style="color:#e8b84b;">🎉 {h["name"]}</b> '
            f'<span style="color:#5a7a6a;">— {h["place"]}</span><br>'
            f'<span style="color:#a8bfb3;">{h["note"]}</span></div>'
            for h in seasonal_hits
        )
        st.markdown(f"""
        <div style="background:#2a2410;border:1px solid rgba(232,184,75,0.3);border-radius:9px;
                    padding:10px 14px;margin-bottom:12px;font-size:0.78rem;line-height:1.5;">
          <span style="color:#e8b84b;font-weight:600;">📅 Timed right — seasonal highlights on this trip:</span>
          {hits_html}
        </div>""", unsafe_allow_html=True)

    # Itinerary + Map
    res_left, res_right = st.columns([3, 2], gap="large")

    with res_left:
        st.markdown('<span class="mono-label">📋 Your Itinerary</span>', unsafe_allow_html=True)

        def md_to_html(text):
            text = re.sub(r'^### (.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
            text = re.sub(r'^## (.+)$',  r'<h2>\1</h2>', text, flags=re.MULTILINE)
            text = re.sub(r'^# (.+)$',   r'<h1>\1</h1>', text, flags=re.MULTILINE)
            text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
            text = re.sub(r'^---+$', r'<hr>', text, flags=re.MULTILINE)
            text = re.sub(r'^[-*] (.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)
            text = re.sub(r'(<li>.*?</li>\n?)+', lambda m: '<ul>' + m.group(0) + '</ul>', text)
            lines = text.split('\n')
            out = []
            for line in lines:
                s = line.strip()
                if not s: out.append('<br>')
                elif s.startswith('<'): out.append(s)
                else: out.append(s + '<br>')
            return '\n'.join(out)

        st.markdown(
            f'<div style="background:#1a3528;border:1px solid rgba(255,255,255,0.1);border-left:4px solid #3dba7e;'
            f'border-radius:14px;padding:24px 26px;max-height:600px;overflow-y:auto;">'
            f'<div id="itin-inner">{md_to_html(display_itinerary)}</div></div>',
            unsafe_allow_html=True)

        arr_opt = ARRIVAL_OPTIONS.get(st.session_state.arrival_time, ARRIVAL_OPTIONS["morning"])
        st.markdown(f"""<div class="chip-row" style="margin-top:10px;">
          <span class="chip chip-g">{len(itinerary.split())} words</span>
          <span class="chip chip-y">llama-3.3-70b</span>
          <span class="chip chip-m">memory-aware</span>
          <span class="chip chip-m">{arr_opt['label']} arrival</span>
        </div>""", unsafe_allow_html=True)

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button("⬇️  Download (.txt)", data=itinerary,
                file_name="srilanka_itinerary.txt", mime="text/plain", use_container_width=True)
        with dl_col2:
            try:
                pdf_bytes = generate_pdf(display_itinerary, "My Sri Lanka Itinerary — WaLKer")
                st.download_button("📄  Download (.pdf)", data=pdf_bytes,
                    file_name="srilanka_itinerary.pdf", mime="application/pdf", use_container_width=True)
            except Exception:
                st.caption("PDF export needs the `fpdf2` package (pip install fpdf2).")

        whatsapp_summary = (
            f"My Sri Lanka trip plan via WaLKer 🌴\n"
            f"{days} days · {budget.split()[0]} budget · {', '.join(interests_selected)}\n\n"
            f"{itinerary[:500]}..."
        )
        whatsapp_url = f"https://wa.me/?text={_url_quote(whatsapp_summary)}"
        st.markdown(
            f'<a href="{whatsapp_url}" target="_blank" style="display:block;text-align:center;'
            f'background:rgba(61,186,126,0.12);color:#3dba7e;border:1.5px solid rgba(61,186,126,0.35);'
            f'border-radius:10px;padding:10px 20px;font-weight:600;font-size:0.86rem;'
            f'text-decoration:none;margin-top:8px;">📲 Share via WhatsApp</a>',
            unsafe_allow_html=True,
        )

        st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
        with st.expander("🔄 Refine this itinerary"):
            # Check current live weather at the places already in this
            # itinerary, so a bad-weather warning shows before the user even
            # types a refinement request — and gets folded into the AI's
            # instructions if they do refine.
            weather_flags = check_itinerary_weather(st.session_state.get("place_names", []))
            if weather_flags:
                flags_html = "".join(
                    f'<div style="margin-top:4px;"><b style="color:#e87060;">{f["place"]}</b> — '
                    f'{f["description"]}, {f["temp"]}°C</div>'
                    for f in weather_flags
                )
                st.markdown(f"""
                <div style="background:#3a1e1a;border:1px solid rgba(232,112,96,0.35);border-radius:9px;
                            padding:10px 14px;margin-bottom:10px;font-size:0.78rem;line-height:1.5;color:#f0b3a8;">
                  <b>⚠️ Weather alert:</b> current conditions look poor at these planned stops —
                  {flags_html}
                  <div style="margin-top:6px;color:#e8c9c2;">Refining below will automatically ask the AI to suggest an alternative for these.</div>
                </div>""", unsafe_allow_html=True)
                if st.button("🔁 Auto-fix for weather", key="btn_weather_autofix", use_container_width=True):
                    with st.spinner("Rerouting around bad weather..."):
                        new_it, ge = refine_trip(
                            st.session_state.itinerary,
                            "Keep the rest of the trip as-is, just address the weather alert.",
                            weather_flags=weather_flags,
                        )
                    if ge.get("status") == "error":
                        st.error(new_it)
                    else:
                        st.session_state.itinerary   = new_it
                        st.session_state.goal_eval   = ge
                        st.session_state.place_names = extract_place_names(new_it)
                        st.session_state.chat_messages = []
                        st.session_state.chat_history  = []
                        trip_id = st.session_state.get("trip_id")
                        if trip_id:
                            update_trip(trip_id, new_it)
                        else:
                            st.session_state.trip_id = save_trip(
                                days, ", ".join(interests_selected), budget, new_it
                            )
                        st.rerun()

            refine_input = st.text_area("refine",
                placeholder="e.g. Add more beach time on day 3, swap Kandy for Ella...",
                height=60, label_visibility="collapsed")
            if st.button("Apply Changes", key="btn_refine"):
                if refine_input.strip():
                    with st.spinner("Refining..."):
                        new_it, ge = refine_trip(
                            st.session_state.itinerary, refine_input,
                            weather_flags=weather_flags,
                        )

                    if ge.get("status") == "error":
                        st.error(new_it)
                    else:
                        st.session_state.itinerary   = new_it
                        st.session_state.goal_eval   = ge
                        st.session_state.place_names = extract_place_names(new_it)
                        st.session_state.chat_messages = []
                        st.session_state.chat_history  = []
                        trip_id = st.session_state.get("trip_id")
                        if trip_id:
                            # Same trip, refined — update the existing row rather
                            # than inserting a new one each time "Apply Changes" is clicked.
                            update_trip(trip_id, new_it)
                        else:
                            st.session_state.trip_id = save_trip(
                                days, ", ".join(interests_selected), budget, new_it
                            )
                        st.rerun()
                else:
                    st.warning("Describe what to change first.")

    with res_right:
        locations = get_place_locations(st.session_state.place_names)
        st.markdown(f'<span class="mono-label">🗺️ Map · {len(locations)} places</span>', unsafe_allow_html=True)
        if not locations:
            st.info("Locations will appear after generation.")
        else:
            try:
                import folium
                from streamlit_folium import st_folium
                m = folium.Map(location=[7.8731, 80.7718], zoom_start=7, tiles="CartoDB dark_matter")
                for loc in locations:
                    folium.CircleMarker(
                        location=[loc["latitude"], loc["longitude"]],
                        radius=9, color="#3dba7e", fill=True, fill_color="#3dba7e", fill_opacity=0.85,
                        tooltip=folium.Tooltip(loc["name"],
                            style="font-family:Cambria;font-size:12px;background:#1a3528;color:#e8f0ec;border:none;padding:4px 8px;border-radius:6px;"),
                        popup=folium.Popup(loc["name"], max_width=120),
                    ).add_to(m)
                    folium.Marker(
                        location=[loc["latitude"], loc["longitude"]],
                        icon=folium.DivIcon(
                            html=f'<div style="font-family:Cambria,sans-serif;font-size:10px;font-weight:600;color:#e8f0ec;white-space:nowrap;margin-top:12px;text-shadow:0 1px 4px rgba(0,0,0,0.9);">{loc["name"]}</div>',
                            icon_size=(100,20), icon_anchor=(0,0)),
                    ).add_to(m)
                st_folium(m, width=None, height=580, returned_objects=[])
            except ImportError:
                for loc in locations:
                    st.markdown(f'<div class="hist-card">📍 <b style="color:#3dba7e;">{loc["name"]}</b></div>', unsafe_allow_html=True)
                st.caption("pip install folium streamlit-folium")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    msg_count = len([m for m in st.session_state.chat_history if m["role"] == "user"])

    st.markdown(f"""
    <div class="chat-header-bar">
      <div class="chat-hicon">🤖</div>
      <div style="flex:1;">
        <div class="chat-htitle" style="display:flex;align-items:center;gap:6px;"><img src="{LOGO_ICON}" alt="WaLKer" style="height:20px;width:auto;"> AI Agent</div>
        <div class="chat-hsub">Ask anything about your itinerary · {msg_count} messages</div>
      </div>
      <div class="chat-online"></div>
    </div>""", unsafe_allow_html=True)

    # Chat messages area — white "inside the agent" per user request; the
    # header bar above and the input/send row below keep their existing
    # dark styling untouched.
    chat_container_style = """
    background:#ffffff;
    border:1px solid rgba(12,52,70,0.15);
    border-top:none;
    padding:16px 18px;
    min-height:180px;
    max-height:400px;
    overflow-y:auto;
    """

    if not st.session_state.chat_history:
        st.markdown(f"""
        <div style="{chat_container_style}text-align:center;display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:180px;">
          <div style="font-size:2rem;margin-bottom:8px;">💬</div>
          <div style="font-size:0.82rem;color:#0c3446 !important;line-height:1.9;">
            Ask me anything about your trip!<br>
            <span style="color:#3dba7e;">Add places · Get tips · Plan transport · Budget breakdown</span>
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        def format_agent_text(text: str) -> str:
            text = re.sub(r'\*\*(.+?)\*\*', r'<strong style="color:#e8b84b;">\1</strong>', text)
            lines = text.split('\n')
            out, in_list = [], False
            for line in lines:
                s = line.strip()
                if s.startswith('- ') or s.startswith('* '):
                    if not in_list:
                        out.append('<ul style="padding-left:1.1rem;margin:6px 0;">')
                        in_list = True
                    out.append(f'<li style="margin-bottom:5px;color:#e8f0ec;">{s[2:]}</li>')
                else:
                    if in_list:
                        out.append('</ul>')
                        in_list = False
                    if s:
                        out.append(f'<p style="margin:4px 0;color:#e8f0ec;">{s}</p>')
            if in_list:
                out.append('</ul>')
            return '\n'.join(out)

        t = datetime.datetime.now().strftime("%H:%M")
        msgs_html = ""
        for msg in st.session_state.chat_history:
            is_user = msg["role"] == "user"
            if is_user:
                safe = msg["content"].replace("<","&lt;").replace(">","&gt;")
                msgs_html += f"""
                <div class="bubble-user-row">
                  <div>
                    <div class="bubble-user">{safe}</div>
                    <div class="bubble-user-meta">You · {t}</div>
                  </div>
                </div>"""
            else:
                formatted = format_agent_text(msg["content"])
                msgs_html += f"""
                <div class="bubble-agent-row">
                  <div class="chat-avatar">🤖</div>
                  <div>
                    <div class="bubble-agent">{formatted}</div>
                    <div class="bubble-agent-meta">Agent · {t}</div>
                  </div>
                </div>"""

        st.markdown(f'<div style="{chat_container_style}">{msgs_html}</div>', unsafe_allow_html=True)

    # Quick suggestions row
    suggestions = ["🚂 Best transport?","🏨 Hotel tips?","🍛 Must-try foods?","📅 Add Kandy","💰 Budget breakdown?","🌧️ Best time to visit?"]
    sug_html = " ".join(f'<span class="sug-chip" style="cursor:pointer;">{s}</span>' for s in suggestions)
    st.markdown(f"""
    <div class="sug-row">
      <span class="sug-label">Quick questions</span>
      {sug_html}
    </div>""", unsafe_allow_html=True)

    # ── Input box + Send button INSIDE the chat frame ──
    st.markdown("""
    <div style="background:#0f1e15;border:1px solid rgba(61,186,126,0.15);
                border-top:1px solid rgba(61,186,126,0.1);
                border-radius:0 0 14px 14px;padding:10px 14px;">
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <style>
    /* Vertically center the input and send button on the same row instead of
       relying on a magic-number margin — flex-start (Streamlit's default) was
       top-aligning them, and the button is a different height than the input. */
    .st-key-chat_input_row div[data-testid="stHorizontalBlock"] {
        align-items: center !important;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.container(key="chat_input_row"):
        input_col, btn_col = st.columns([9, 1], gap="small")
        with input_col:
            st.markdown("""
            <style>
            div[data-testid="stTextInput"] > div > div > input {
                background: linear-gradient(135deg,#0c3446,#19352a) !important;
                border: 1.5px solid rgba(61,186,126,0.4) !important;
                border-radius: 10px !important;
                color: #e8f0ec !important;
                font-size: 0.86rem !important;
                padding: 10px 14px !important;
            }
            div[data-testid="stTextInput"] > div > div > input:focus {
                border-color: #3dba7e !important;
                box-shadow: 0 0 0 3px rgba(61,186,126,0.15) !important;
            }
            </style>
            """, unsafe_allow_html=True)
            if st.session_state.get("clear_chat_input"):
                st.session_state["clear_chat_input"] = False
                st.session_state["chat_input"] = ""
            user_q = st.text_input(
                "chat_input_field",
                placeholder="Ask anything... e.g. Best restaurants in Kandy · Train routes · Budget tips",
                label_visibility="collapsed",
                key="chat_input"
            )
        with btn_col:
            st.markdown("""
            <style>
            .st-key-btn_send_chat button {
                background: linear-gradient(135deg,#0c3446,#19352a) !important;
                border: 1.5px solid rgba(61,186,126,0.4) !important;
                border-radius: 10px !important;
                color: #e8f0ec !important;
                font-size: 1.1rem !important;
                padding: 9px 10px !important;
                width: 100% !important;
                box-shadow: none !important;
            }
            </style>
            """, unsafe_allow_html=True)
            send_clicked = st.button("➤", key="btn_send_chat", use_container_width=True)

# Handle send
    if send_clicked and user_q and user_q.strip():
        with st.spinner("🤖 Agent thinking..."):
            reply, updated = chat_with_agent(
                st.session_state.chat_messages, user_q.strip(), itinerary)
        st.session_state.chat_messages = updated
        st.session_state.chat_history.append({"role": "user",      "content": user_q.strip()})
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.session_state["clear_chat_input"] = True
        st.rerun()
    # Clear button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat history", key="btn_clr", use_container_width=False):
            st.session_state.chat_messages = []
            st.session_state.chat_history  = []
            st.rerun()
