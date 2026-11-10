"""Look and feel: CSS plus small HTML components. No extra dependencies."""
import html
import numpy as np
import streamlit as st

TIER_COLORS = ["#1B9E77", "#4FB06D", "#84C260", "#B5D057", "#E1D653",
               "#F2C94C", "#F2A33C", "#EE7B34", "#E4552F", "#C93A2E"]

NICE = {"VehPower": "Vehicle power", "VehAge": "Vehicle age", "DrivAge": "Driver age",
        "BonusMalus": "Bonus-malus", "Density": "Density", "VehBrand": "Vehicle brand",
        "VehGas": "Fuel", "Region": "Region", "bm_off_ladder": "Off the claim-free ladder"}

CSS = """
<style>
:root{--ink:#1B2A3A;--muted:#6B7C93;--line:#E3E9F2;}
.stApp{background:linear-gradient(180deg,#F4F8FC 0%,#EAF1F8 100%);}
.block-container{padding-top:1.6rem;padding-bottom:3rem;max-width:1180px;}
header[data-testid="stHeader"]{background:transparent;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0F2A43 0%,#12476B 60%,#0B6E99 100%);}
[data-testid="stSidebar"] *{color:#EAF2FA !important;}
.brand{padding:6px 0 14px;border-bottom:1px solid rgba(255,255,255,.18);margin-bottom:12px;}
.brand .name{font-size:1.15rem;font-weight:700;letter-spacing:-.01em;}
.brand .tag{font-size:.8rem;opacity:.8;}
.hero{background:linear-gradient(120deg,#0F2A43 0%,#0B6E99 55%,#14A3A3 100%);border-radius:18px;padding:26px 30px;margin-bottom:22px;box-shadow:0 10px 30px rgba(15,42,67,.18);}
.hero h1{margin:0;font-size:1.9rem;font-weight:700;color:#fff;letter-spacing:-.01em;}
.hero p{margin:6px 0 0;color:rgba(255,255,255,.88);font-size:1rem;}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin:6px 0 18px;}
.kpi{background:#fff;border:1px solid var(--line);border-top:4px solid var(--accent,#0B6E99);border-radius:16px;padding:16px 18px;box-shadow:0 4px 14px rgba(15,42,67,.06);transition:transform .15s ease,box-shadow .15s ease;}
.kpi:hover{transform:translateY(-2px);box-shadow:0 10px 24px rgba(15,42,67,.12);}
.kpi .lab{font-size:.75rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);font-weight:600;}
.kpi .val{font-size:1.9rem;font-weight:700;color:var(--ink);margin-top:4px;line-height:1.1;}
.kpi .sub{font-size:.82rem;color:var(--muted);margin-top:4px;}
.card{background:#fff;border:1px solid var(--line);border-radius:16px;padding:18px 20px;margin:8px 0 18px;box-shadow:0 4px 14px rgba(15,42,67,.06);}
.ctitle{font-size:1.05rem;font-weight:700;color:var(--ink);margin-bottom:6px;}
.note{font-size:.8rem;color:var(--muted);margin-top:10px;}
.scalewrap{margin:0 0 18px;}
.tierscale{display:flex;gap:4px;margin:10px 0 6px;}
.tierscale div{flex:1;height:12px;border-radius:7px;opacity:.35;}
.tierscale div.on{opacity:1;transform:scaleY(1.7);box-shadow:0 0 0 2px rgba(27,42,58,.3);}
.scaleends{display:flex;justify-content:space-between;font-size:.75rem;color:var(--muted);}
.fsum{font-size:.9rem;color:var(--muted);margin-bottom:8px;}
.frow{display:grid;grid-template-columns:240px 1fr 70px;align-items:center;gap:14px;padding:8px 0;border-bottom:1px solid var(--line);}
.flabel{font-size:.9rem;color:var(--ink);}
.fval{font-size:.9rem;font-weight:700;color:var(--ink);text-align:right;}
.fbar{position:relative;height:12px;background:#EEF3F9;border-radius:6px;}
.fmid{position:absolute;left:50%;top:-4px;bottom:-4px;width:2px;background:#9FB0C4;}
.ffill{position:absolute;top:0;bottom:0;border-radius:6px;}
.ffill.up{background:linear-gradient(90deg,#F2A33C,#E4552F);}
.ffill.down{background:linear-gradient(270deg,#4FB06D,#1B9E77);}
.trow{display:grid;grid-template-columns:70px 1fr 120px;align-items:center;gap:12px;padding:5px 0;}
.tname{font-size:.85rem;color:var(--ink);font-weight:600;}
.tname.wide{width:150px;}
.ttrack{height:16px;background:#EEF3F9;border-radius:8px;overflow:hidden;}
.tfill{height:100%;border-radius:8px;transition:width .4s ease;}
.tnum{font-size:.82rem;color:var(--muted);text-align:right;}
[data-testid="stFormSubmitButton"] button,.stDownloadButton button{background:linear-gradient(120deg,#0B6E99,#14A3A3);color:#fff;border:0;border-radius:12px;padding:.55rem 1.4rem;font-weight:600;box-shadow:0 6px 16px rgba(11,110,153,.3);transition:transform .15s ease,box-shadow .15s ease;}
[data-testid="stFormSubmitButton"] button:hover,.stDownloadButton button:hover{transform:translateY(-1px);color:#fff;box-shadow:0 10px 22px rgba(11,110,153,.35);}
.userbox{padding:10px 12px;border-radius:12px;background:rgba(255,255,255,.10);margin:8px 0 10px;}
.userbox .uname{font-weight:700;font-size:.95rem;}
.userbox .urole{font-size:.78rem;opacity:.8;}
[data-testid="stSidebar"] .stButton button{background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.35);border-radius:10px;}
[data-testid="stSidebar"] .stButton button:hover{background:rgba(255,255,255,.22);}
</style>
"""


def apply():
    st.markdown(CSS, unsafe_allow_html=True)


def show(markup, sidebar=False):
    (st.sidebar if sidebar else st).markdown(markup, unsafe_allow_html=True)


def tier_label(t):
    return "Lower risk" if t <= 3 else ("Medium risk" if t <= 7 else "Higher risk")


def hero(title, subtitle):
    return f'<div class="hero"><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>'


def kpi(label, value, sub="", accent="#0B6E99"):
    return (f'<div class="kpi" style="--accent:{accent}"><div class="lab">{html.escape(label)}</div>'
            f'<div class="val">{html.escape(value)}</div><div class="sub">{html.escape(sub)}</div></div>')


def kpi_row(items):
    return '<div class="kpis">' + "".join(items) + '</div>'


def tier_scale(tier):
    cells = "".join(f'<div class="{"on" if i + 1 == tier else ""}" style="background:{c}"></div>'
                    for i, c in enumerate(TIER_COLORS))
    return ('<div class="scalewrap"><div class="tierscale">' + cells + '</div>'
            '<div class="scaleends"><span>Tier 1: lowest risk</span><span>Tier 10: highest risk</span></div></div>')


def tier_bars(counts):
    total = sum(counts.values()) or 1
    mx = max(counts.values()) or 1
    rows = "".join(
        f'<div class="trow"><div class="tname">Tier {t}</div>'
        f'<div class="ttrack"><div class="tfill" style="width:{100 * counts.get(t, 0) / mx:.1f}%;background:{TIER_COLORS[t - 1]}"></div></div>'
        f'<div class="tnum">{counts.get(t, 0):,} · {100 * counts.get(t, 0) / total:.1f}%</div></div>'
        for t in range(1, 11))
    return '<div class="card"><div class="ctitle">Risk tiers in this file</div>' + rows + '</div>'


def compare_bars(items, title, note=""):
    mx = max(v for _, v, _ in items)
    rows = "".join(
        f'<div class="trow" style="grid-template-columns:170px 1fr 80px"><div class="tname">{html.escape(l)}</div>'
        f'<div class="ttrack"><div class="tfill" style="width:{100 * v / mx:.1f}%;background:{c}"></div></div>'
        f'<div class="tnum">{v:.4f}</div></div>' for l, v, c in items)
    return (f'<div class="card"><div class="ctitle">{html.escape(title)}</div>{rows}'
            f'<div class="note">{html.escape(note)}</div></div>')


def _fmt_value(name, val):
    if name == "bm_off_ladder":
        return "yes" if int(val) == 1 else "no"
    return str(val)


def factors_card(ex):
    items = [(f"{NICE.get(n, n)}: {_fmt_value(n, v)}", f) for n, v, f in ex["drivers"]]
    items.append(("All other characteristics", ex["other_factor"]))
    mx = max(abs(np.log(f)) for _, f in items) or 1.0
    rows = []
    for label, f in items:
        w = 50 * abs(np.log(f)) / mx
        side = "up" if f >= 1 else "down"
        pos = f"left:50%;width:{w:.1f}%" if f >= 1 else f"right:50%;width:{w:.1f}%"
        rows.append(f'<div class="frow"><div class="flabel">{html.escape(label)}</div>'
                    f'<div class="fbar"><div class="fmid"></div><div class="ffill {side}" style="{pos}"></div></div>'
                    f'<div class="fval">× {f:.2f}</div></div>')
    head = (f'<div class="fsum">Model base level <b>{ex["base_level"]:.4f}</b> × the factors below = '
            f'<b>{ex["annual_rate"]:.4f}</b> claims per year</div>')
    note = ('<div class="note">Bars to the right raise the predicted rate, bars to the left lower it. These show how '
            'this prediction is built; they are associations in the data, not the effect of changing a characteristic.</div>')
    return '<div class="card"><div class="ctitle">How this prediction is composed</div>' + head + "".join(rows) + note + '</div>'

