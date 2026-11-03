"""Custom CSS injected into the Streamlit page."""
from .config import ACCENT, AMBER, CYAN, DOWN, MUTED, TEXT, UP

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'Inter', sans-serif; }}
.stApp {{
  background:
    radial-gradient(1100px 600px at 0% -10%, rgba(124,92,255,.16), transparent 60%),
    radial-gradient(900px 500px at 100% 0%, rgba(34,211,238,.09), transparent 60%),
    #0b0f19;
}}
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }}
section[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, #111827 0%, #0d1220 100%);
  border-right: 1px solid rgba(255,255,255,.06);
}}
.brand {{ font-size: 1.55rem; font-weight: 800; letter-spacing: -.02em;
  background: linear-gradient(90deg, {ACCENT}, {CYAN}); -webkit-background-clip: text;
  -webkit-text-fill-color: transparent; }}
.brand-sub {{ color: {MUTED}; font-size: .85rem; margin-top: -4px; margin-bottom: .6rem; }}
.hero {{ display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap;
  gap: 1rem; margin-bottom: 1.2rem; }}
.hero h1 {{ font-size: 2.1rem; font-weight: 800; letter-spacing: -.03em; margin: 0; padding: 0; color: {TEXT}; }}
.hero .ticker {{ color: {MUTED}; font-weight: 600; }}
.badges {{ display: flex; gap: .5rem; flex-wrap: wrap; }}
.badge {{ font-size: .75rem; font-weight: 600; padding: .3rem .7rem; border-radius: 999px;
  border: 1px solid rgba(255,255,255,.1); background: rgba(255,255,255,.04); color: {TEXT}; }}
.badge.live {{ color: {UP}; border-color: rgba(0,212,160,.35); background: rgba(0,212,160,.08); }}
.badge.sample {{ color: {AMBER}; border-color: rgba(251,191,36,.35); background: rgba(251,191,36,.08); }}
.metric-card {{
  background: linear-gradient(145deg, rgba(255,255,255,.055), rgba(255,255,255,.015));
  border: 1px solid rgba(255,255,255,.08); border-radius: 16px; padding: 1rem 1.2rem;
  box-shadow: 0 8px 30px rgba(0,0,0,.25); height: 100%;
}}
.metric-label {{ font-size: .72rem; text-transform: uppercase; letter-spacing: .09em; color: {MUTED}; font-weight: 600; }}
.metric-value {{ font-size: 1.65rem; font-weight: 750; color: {TEXT}; margin-top: .25rem; line-height: 1.2; }}
.metric-sub {{ font-size: .82rem; color: {MUTED}; margin-top: .2rem; }}
.up {{ color: {UP} !important; }}
.down {{ color: {DOWN} !important; }}
.trend-card {{ border-radius: 18px; padding: 1.4rem 1.5rem; border: 1px solid rgba(255,255,255,.08); }}
.trend-card.up {{ background: linear-gradient(145deg, rgba(0,212,160,.16), rgba(0,212,160,.03)); border-color: rgba(0,212,160,.3); }}
.trend-card.down {{ background: linear-gradient(145deg, rgba(255,77,109,.16), rgba(255,77,109,.03)); border-color: rgba(255,77,109,.3); }}
.trend-dir {{ font-size: 2.8rem; font-weight: 800; letter-spacing: -.02em; line-height: 1.1; margin: .4rem 0 .2rem; }}
.trend-prob {{ font-size: 1.05rem; color: {TEXT}; font-weight: 600; }}
.trend-line {{ color: {TEXT}; font-size: .92rem; margin-top: .55rem; }}
.disclaimer {{ font-size: .75rem; color: {MUTED}; margin-top: .9rem; padding-top: .7rem;
  border-top: 1px dashed rgba(255,255,255,.12); }}
.section-title {{ font-size: 1.05rem; font-weight: 700; color: {TEXT}; margin: 1.2rem 0 .5rem; }}
.stTabs [data-baseweb="tab-list"] {{ gap: .4rem; }}
.stTabs [data-baseweb="tab"] {{ background: rgba(255,255,255,.03); border-radius: 10px 10px 0 0; padding: .5rem 1rem; }}
.footer-note {{ text-align: center; color: {MUTED}; font-size: .78rem; margin-top: 2rem; }}
</style>
"""
