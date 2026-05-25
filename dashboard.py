# ============================================================
# ADVANCED DATA ANALYTICS DASHBOARD
# Run:  streamlit run dashboard_enhanced.py
# ============================================================

import sys

from dash1 import cat_cols
try:
    import streamlit as st
    _ = st.session_state
except Exception:
    raise SystemExit("\n❌  Run with:  streamlit run dashboard_enhanced.py\n")

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Optional ML libs ──
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.ensemble import IsolationForest
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_OK = True
except ImportError:
    STATSMODELS_OK = False

try:
    from prophet import Prophet
    PROPHET_OK = True
except ImportError:
    PROPHET_OK = False

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Data Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ============================================================
# HIDE STREAMLIT HEADER / DEPLOY PANEL
# ============================================================

st.markdown("""
<style>

/* Hide top header بالكامل */
[data-testid="stHeader"] {
    display: none;
}

/* Hide deploy button specifically */
.stDeployButton {
    display: none;
}

/* Remove top padding created by header */
.block-container {
    padding-top: 1rem;
}

</style>
""", unsafe_allow_html=True)
# ============================================================
# DESIGN TOKENS
# ============================================================

C_BG      = "#0d1117"
C_SIDEBAR = "#0a0e18"
C_SURFACE = "#111827"
C_PANEL   = "#0f1623"
C_BORDER  = "#1e2d42"
C_CYAN    = "#00dcc8"
C_BLUE    = "#1a6fd4"
C_ORANGE  = "#d4740a"
C_PURPLE  = "#6c3fc7"
C_GREEN   = "#10b981"
C_RED     = "#ef4444"
C_TEXT    = "#c5cede"
C_MUTED   = "#5a6a82"
C_DIM     = "#3a4a5e"
C_BRIGHT  = "#e8eef8"

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,14,24,0.9)",
    font=dict(family="Exo 2, system-ui, sans-serif", color=C_TEXT, size=10),
    title_font=dict(family="Rajdhani, sans-serif", size=13, color=C_BRIGHT),
    xaxis=dict(
        gridcolor="rgba(30,45,66,0.9)", zerolinecolor=C_BORDER,
        tickfont=dict(size=9, color=C_MUTED), linecolor=C_BORDER, ticklen=3,
        title_font=dict(size=9, color=C_MUTED),
    ),
    yaxis=dict(
        gridcolor="rgba(30,45,66,0.9)", zerolinecolor=C_BORDER,
        tickfont=dict(size=9, color=C_MUTED), linecolor=C_BORDER, ticklen=3,
        title_font=dict(size=9, color=C_MUTED),
    ),
    legend=dict(
        bgcolor="rgba(10,14,24,0.9)", bordercolor=C_BORDER, borderwidth=1,
        font=dict(size=9, color=C_TEXT), orientation="h",
        yanchor="bottom", y=1.01, xanchor="left", x=0,
    ),
    margin=dict(l=8, r=8, t=36, b=8),
)

PALETTES = {
    "Portal":      [C_BLUE, C_ORANGE, C_PURPLE, C_CYAN, "#2a9fd4","#c7603f","#4c6fd4","#a040c7"],
    "Blue+Orange": [C_BLUE, C_ORANGE,"#2a7fe4","#e8851a","#3a8ff4","#f8952a"],
    "Cyan+Purple": [C_CYAN, C_PURPLE,"#10ece8","#7c4fd7","#20fcf8","#8c5fe7"],
    "Vivid":       px.colors.qualitative.Vivid,
    "Plasma":      "plasma",
    "Viridis":     "viridis",
    "RdBu":        "RdBu",
    "Blues":       "Blues",
}

GOOGLE_FONTS = [
    "Exo 2", "Rajdhani", "Inter", "Roboto", "Poppins",
    "Montserrat", "Source Code Pro", "Nunito",
]

# ============================================================
# SESSION STATE DEFAULTS
# ============================================================

if "bg_color"    not in st.session_state: st.session_state.bg_color    = C_BG
if "font_family" not in st.session_state: st.session_state.font_family = "Exo 2"
if "font_color"  not in st.session_state: st.session_state.font_color  = C_TEXT
if "raw_edits"   not in st.session_state: st.session_state.raw_edits   = {}
if "edit_history"not in st.session_state: st.session_state.edit_history= []
if "chat_open"   not in st.session_state: st.session_state.chat_open   = False
if "chat_msgs"   not in st.session_state: st.session_state.chat_msgs   = []
if "df_edited"   not in st.session_state: st.session_state.df_edited   = None

# ============================================================
# HELPERS
# ============================================================

def style_fig(fig, height=280):
    fig.update_layout(height=height, **PLOTLY_BASE)
    fig.update_traces(marker_line_width=0)
    return fig

def num_cols(df):
    return df.select_dtypes(include=np.number).columns.tolist()

def date_cols(df):
    """Return datetime columns — also auto-converts string columns that look like dates."""
    found = []
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            found.append(c)
        elif df[c].dtype == object:
            sample = df[c].dropna().head(50)
            try:
                converted = pd.to_datetime(sample, infer_datetime_format=True, errors="raise")
                if converted.notna().sum() >= len(sample) * 0.6:
                    found.append(c)
            except Exception:
                pass
    return found

def cat_cols(df):
    return df.select_dtypes(include=["object", "category"]).columns.tolist()

def is_numeric(df, col):
    return col in num_cols(df)

def compatible_y_for_x(df, x_col):
    nc = num_cols(df)
    cc = cat_cols(df)
    if x_col in nc:
        return nc
    if x_col in cc:
        return nc
    return nc

def compatible_corr_cols(df, selected_cols):
    nc = num_cols(df)
    already = set(selected_cols)
    return [c for c in nc if c not in already]

def get_applicable_agg(df, cols):
    nc = num_cols(df)
    aggs = ["count"]
    if any(c in nc for c in cols):
        aggs = ["mean", "sum", "median", "min", "max", "count", "std"]
    return aggs

def get_compatible_charts_for_col(df, col):
    nc = num_cols(df)
    cc = cat_cols(df)
    dc = date_cols(df)
    if col in nc:
        return ["Histogram", "Box", "Violin", "Line", "Area", "Bar"]
    if col in cc:
        return ["Bar", "Horizontal Bar", "Pie", "Donut", "Treemap"]
    if col in dc:
        return ["Line", "Area"]
    return ["Bar"]

def get_compatible_charts_for_axes(df, x_col, y_col, z_col=None):
    nc = num_cols(df)
    cc = cat_cols(df)
    charts = []
    if x_col and y_col:
        if x_col in cc and y_col in nc:
            charts = ["Bar", "Horizontal Bar", "Box", "Violin", "Funnel", "Pie", "Treemap"]
        elif x_col in nc and y_col in nc:
            charts = ["Scatter", "Line", "Area", "Bar", "Histogram", "Heatmap"]
        elif x_col in cc and y_col in cc:
            charts = ["Heatmap", "Bar", "Pie", "Treemap"]
        else:
            charts = ["Bar", "Scatter"]
    if z_col:
        charts = [c for c in charts if c in ["Scatter", "Bar", "Heatmap"]] + ["Bubble"]
    return charts or ["Bar", "Scatter", "Line"]

def corr_method_applicable(df, cols):
    return ["pearson", "spearman", "kendall"]

def top_bottom_filter(df, col, n, direction="both"):
    if col not in num_cols(df):
        return df
    sorted_df = df.sort_values(col, ascending=False)
    if direction == "top":
        return sorted_df.head(n)
    elif direction == "bottom":
        return sorted_df.tail(n)
    else:
        top = sorted_df.head(n)
        bot = sorted_df.tail(n)
        return pd.concat([top, bot]).drop_duplicates()

# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_csv(file_bytes, file_name):
    import io
    if file_name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes))
    elif file_name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(file_bytes))
    elif file_name.endswith(".json"):
        return pd.read_json(io.BytesIO(file_bytes))
    return None

def generate_sample_data():
    np.random.seed(42)
    n = 600
    regions    = ["AMER", "EMEA", "APAC"]
    proj_types = ["Development", "Marketing", "Research", "Operations"]
    ship_modes = ["Standard", "Express", "Overnight", "Economy"]
    statuses   = ["Active", "Completed", "On Hold", "At Risk"]
    priorities = ["High", "Medium", "Low"]
    df = pd.DataFrame({
        "project_id":    [f"PRJ-{i:04d}" for i in range(n)],
        "project_name":  [f"Project {np.random.choice(['Alpha','Beta','Gamma','Delta','Sigma'])} {i}" for i in range(n)],
        "project_type":  np.random.choice(proj_types, n, p=[.35,.30,.20,.15]),
        "order_region":  np.random.choice(regions, n, p=[.40,.35,.25]),
        "shipping_mode": np.random.choice(ship_modes, n, p=[.45,.30,.15,.10]),
        "status":        np.random.choice(statuses, n),
        "priority":      np.random.choice(priorities, n),
        "manager":       np.random.choice(["Alice","Bob","Carol","David","Eve"], n),
        "sales":         np.random.lognormal(9, 1.2, n).astype(int),
        "revenue":       np.random.lognormal(12, 1.2, n).astype(int),
        "budget":        np.random.lognormal(11.8, 1.1, n).astype(int),
        "profit":        np.random.normal(5000, 3000, n).astype(int),
        "efficiency":    np.round(np.random.beta(8, 2, n), 3),
        "discount_rate": np.round(np.random.uniform(0, 0.5, n), 3),
        "label":         np.random.binomial(1, 0.43, n),
        "order_date":    pd.date_range("2022-01-01", periods=n, freq="D"),
        "risk_score":    np.round(np.random.uniform(0, 1, n), 2),
        "customer_segment": np.random.choice(["Consumer","Corporate","Home Office"], n),
        "category_name": np.random.choice(["Electronics","Furniture","Office Supplies","Technology"], n),
    })
    return df

# ============================================================
# CSS
# ============================================================

def inject_css(bg, font, font_color):
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&family=Exo+2:wght@300;400;500;600&family=Inter:wght@300;400;500&family=Poppins:wght@300;400;500&family=Montserrat:wght@400;600&family=Roboto:wght@300;400;500&family=Nunito:wght@400;600&family=Source+Code+Pro&display=swap');

:root {{
    --bg:{bg}; --sidebar-bg:{C_SIDEBAR};
    --surface:{C_SURFACE}; --panel-bg:{C_PANEL};
    --border:{C_BORDER}; --cyan:{C_CYAN}; --blue:{C_BLUE};
    --orange:{C_ORANGE}; --purple:{C_PURPLE};
    --text:{font_color}; --bright:{C_BRIGHT}; --muted:{C_MUTED}; --dim:{C_DIM};
}}

html, body, [class*="css"] {{
    font-family:'{font}',system-ui,sans-serif;
    color:{font_color}; background:{bg};
}}
.stApp {{
    background:{bg};
    background-image:
        radial-gradient(ellipse 120% 60% at 60% 0%,rgba(26,109,212,.06) 0%,transparent 60%),
        radial-gradient(ellipse 80% 40% at 0% 100%,rgba(108,63,199,.05) 0%,transparent 50%);
}}
.block-container {{ padding:0 1.4rem 2rem 1.4rem; max-width:1800px; }}

[data-testid="stSidebar"] {{
    background:{C_SIDEBAR} !important;
    border-right:1px solid {C_BORDER} !important;
}}
[data-testid="stSidebar"]::after {{
    content:'';position:absolute;top:15%;right:0;
    width:3px;height:55%;
    background:linear-gradient(180deg,transparent 0%,{C_PURPLE} 30%,{C_CYAN} 70%,transparent 100%);
    border-radius:3px 0 0 3px;
}}
[data-testid="stSidebar"] .block-container {{ padding:0 0.85rem 1.5rem 0.85rem; }}

::-webkit-scrollbar{{width:3px;height:3px;}}
::-webkit-scrollbar-track{{background:{bg};}}
::-webkit-scrollbar-thumb{{background:{C_CYAN};border-radius:2px;}}

[data-testid="stSelectbox"] > div > div {{
    background:{C_SURFACE} !important; border:1px solid {C_BORDER} !important;
    border-radius:5px !important; color:{font_color} !important; font-size:0.82rem !important;
}}
[data-testid="stSelectbox"] > div > div:focus-within {{
    border-color:{C_CYAN} !important; box-shadow:0 0 0 2px rgba(0,220,200,.15) !important;
}}
[data-testid="stMultiSelect"] > div > div {{
    background:{C_SURFACE} !important; border:1px solid {C_BORDER} !important;
    border-radius:5px !important; font-size:0.82rem !important;
}}
[data-testid="stMultiSelect"] span[data-baseweb="tag"] {{
    background:rgba(26,109,212,.25) !important; border:1px solid rgba(26,109,212,.4) !important;
    color:#7ab8f5 !important; font-size:0.7rem !important; border-radius:3px !important;
}}
[data-testid="stFileUploader"] {{
    border:1px dashed rgba(0,220,200,.3) !important; border-radius:7px !important;
    background:rgba(0,220,200,.03) !important;
}}
.stDownloadButton button, .stButton button {{
    background:{C_BLUE} !important; color:#fff !important;
    border:none !important; border-radius:5px !important;
    font-family:'Rajdhani',sans-serif !important; font-weight:700 !important;
    font-size:0.8rem !important; letter-spacing:.08em !important;
    text-transform:uppercase !important; box-shadow:0 0 14px rgba(26,109,212,.35) !important;
    transition:all .2s ease !important;
}}
.stDownloadButton button:hover, .stButton button:hover {{
    background:#2a7fe4 !important; transform:translateY(-1px) !important;
    box-shadow:0 0 22px rgba(26,109,212,.6) !important;
}}
[data-testid="stDataFrame"] {{ border-radius:6px; overflow:hidden; border:1px solid {C_BORDER} !important; }}
[data-testid="stMetric"] {{
    background:{C_SURFACE} !important; border:1px solid {C_BORDER} !important;
    border-left:2px solid {C_CYAN} !important; border-radius:6px !important; padding:.8rem 1rem !important;
}}
[data-testid="stMetricLabel"] {{
    color:{C_MUTED} !important; font-size:0.65rem !important;
    letter-spacing:.12em !important; text-transform:uppercase !important; font-weight:600 !important;
}}
[data-testid="stMetricValue"] {{
    font-family:'Source Code Pro',monospace !important; color:{C_CYAN} !important; font-size:1.2rem !important;
}}
.stTabs [data-baseweb="tab-list"] {{
    background:{C_SURFACE} !important; border-radius:6px !important;
    padding:3px !important; border:1px solid {C_BORDER} !important; gap:2px !important;
}}
.stTabs [data-baseweb="tab"] {{
    background:transparent !important; border-radius:4px !important;
    color:{C_MUTED} !important; font-family:'Exo 2',sans-serif !important;
    font-size:0.76rem !important; font-weight:500 !important;
    letter-spacing:.05em !important; padding:5px 13px !important; text-transform:uppercase !important;
}}
.stTabs [aria-selected="true"] {{
    background:{C_BLUE} !important; color:#fff !important;
    font-weight:600 !important; box-shadow:0 0 12px rgba(26,109,212,.5) !important;
}}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display:none !important; }}
[data-testid="stAlert"] {{
    background:rgba(0,220,200,.05) !important; border:1px solid rgba(0,220,200,.2) !important;
    border-radius:6px !important; color:{font_color} !important; font-size:0.8rem !important;
}}
hr {{ border-color:{C_BORDER} !important; }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HTML HELPERS
# ============================================================

def kpi_card(label, value, sub="", color=C_CYAN):
    return f"""
    <div style="background:linear-gradient(135deg,{C_SURFACE} 0%,{C_PANEL} 100%);
        border:1px solid {C_BORDER}; border-top:2px solid {color};
        border-radius:7px; padding:10px 12px; position:relative; overflow:hidden;
        box-shadow:0 0 16px {color}20;">
        <div style="font-family:'Source Code Pro',monospace;font-size:1.2rem;color:{color};line-height:1.15;">{value}</div>
        <div style="font-size:0.62rem;letter-spacing:.13em;text-transform:uppercase;
                    color:{C_MUTED};margin-top:5px;font-weight:600;">{label}</div>
        {"<div style='font-size:0.68rem;color:"+C_DIM+";margin-top:2px;'>"+sub+"</div>" if sub else ""}
    </div>"""

def sidebar_label(text):
    st.markdown(f"""
    <div style="font-family:'Rajdhani',sans-serif;font-size:0.6rem;
                letter-spacing:.18em;text-transform:uppercase;color:{C_CYAN};
                margin:1rem 0 0.45rem 0;font-weight:700;
                border-bottom:1px solid {C_BORDER};padding-bottom:4px;">{text}</div>
    """, unsafe_allow_html=True)

def panel_header(title):
    st.markdown(f"""
    <div style="background:{C_PANEL};border:1px solid {C_BORDER};
                border-radius:7px 7px 0 0;padding:6px 13px;
                font-family:'Rajdhani',sans-serif;font-size:0.6rem;
                letter-spacing:.13em;text-transform:uppercase;
                color:{C_MUTED};font-weight:700;margin-top:8px;">{title}</div>
    """, unsafe_allow_html=True)

def section_label(text, color=C_CYAN):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:7px;margin:1.1rem 0 0.7rem 0;">
        <div style="width:2px;height:12px;background:{color};border-radius:1px;box-shadow:0 0 5px {color}80;"></div>
        <span style="font-family:'Rajdhani',sans-serif;font-size:0.65rem;letter-spacing:.18em;
                     text-transform:uppercase;color:{color};font-weight:700;">{text}</span>
    </div>""", unsafe_allow_html=True)

def tab_title(text):
    st.markdown(f"""
    <div style="font-family:'Rajdhani',sans-serif;font-size:0.9rem;font-weight:700;
                color:{C_BRIGHT};letter-spacing:.1em;text-transform:uppercase;
                padding:.7rem 0 .7rem 0;border-bottom:1px solid {C_BORDER};margin-bottom:1rem;">{text}</div>
    """, unsafe_allow_html=True)

def info_badge(text, color=C_CYAN):
    return f"""<span style="display:inline-block;background:rgba(0,220,200,0.1);
        border:1px solid rgba(0,220,200,0.3);border-radius:4px;padding:1px 7px;
        font-size:0.7rem;color:{color};font-family:'Source Code Pro',monospace;">{text}</span>"""


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(f"""
    <div style="padding:11px 4px 12px 4px;border-bottom:1px solid {C_BORDER};margin-bottom:2px;">
        <div style="font-family:'Rajdhani',sans-serif;font-size:1rem;letter-spacing:.1em;
                    text-transform:uppercase;color:{C_CYAN};font-weight:700;margin-bottom:2px;">
            📊 Analytics Dashboard
        </div>
        <div style="font-size:0.65rem;color:{C_DIM};">Upload your data to begin</div>
    </div>""", unsafe_allow_html=True)

    sidebar_label("📁 Data Source")
    uploaded = st.file_uploader(
        "Upload file", type=["csv","xlsx","xls","json"],
        label_visibility="collapsed", key="file_upload"
    )

    if uploaded:
        file_bytes = uploaded.read()
        file_size  = len(file_bytes) / 1024
        df_raw = load_csv(file_bytes, uploaded.name)
        st.markdown(f"""
        <div style="background:rgba(0,220,200,.05);border:1px solid rgba(0,220,200,.2);
                    border-radius:5px;padding:6px 10px;margin:5px 0;font-size:0.72rem;">
            <span style="color:{C_CYAN};">📄 {uploaded.name}</span><br>
            <span style="color:{C_DIM};">{df_raw.shape[0]:,} rows · {df_raw.shape[1]} cols · {file_size:.1f} KB</span>
        </div>""", unsafe_allow_html=True)
        if st.button("🗑 Remove File", key="remove_file"):
            st.session_state.file_upload = None
            st.rerun()
    else:
        df_raw = generate_sample_data()
        st.caption("⚡ Using sample dataset — upload your own above")

    for col in df_raw.columns:
        if df_raw[col].dtype == object:
            sample = df_raw[col].dropna().head(100)
            try:
                parsed = pd.to_datetime(sample, infer_datetime_format=True, errors="raise")
                if parsed.notna().sum() >= len(sample) * 0.6:
                    df_raw[col] = pd.to_datetime(df_raw[col], infer_datetime_format=True, errors="coerce")
            except Exception:
                pass
        elif pd.api.types.is_numeric_dtype(df_raw[col]):
            # Unix timestamp detection (10-digit = seconds, 13-digit = ms)
            sample_vals = df_raw[col].dropna().head(20)
            if sample_vals.between(1e9, 2e10).all():
                try:
                    df_raw[col] = pd.to_datetime(df_raw[col], unit="s", errors="coerce")
                except Exception:
                    pass
            elif sample_vals.between(1e12, 2e13).all():
                try:
                    df_raw[col] = pd.to_datetime(df_raw[col], unit="ms", errors="coerce")
                except Exception:
                    pass

    if st.session_state.df_edited is not None:
        df_raw = st.session_state.df_edited.copy()

    sidebar_label("🔍 Global Filters")
    cc_all = cat_cols(df_raw)
    filter_cols = st.multiselect(
        "Filter by columns", cc_all, default=[],
        max_selections=4, placeholder="Pick columns…",
        label_visibility="collapsed"
    )
    active_filters = {}
    for fc in filter_cols:
        vals = sorted(df_raw[fc].dropna().unique().tolist())
        chosen = st.multiselect(fc, vals, default=vals)
        active_filters[fc] = chosen

    df = df_raw.copy()
    for col, vals in active_filters.items():
        df = df[df[col].isin(vals)]
    st.caption(f"🔎 {df.shape[0]:,} / {df_raw.shape[0]:,} rows shown")

    sidebar_label("🎨 Appearance")
    col_a, col_b = st.columns(2)
    with col_a:
        bg_color   = st.color_picker("Background", st.session_state.bg_color, key="bg_pick")
        st.session_state.bg_color = bg_color
    with col_b:
        font_color = st.color_picker("Font Color", st.session_state.font_color, key="fc_pick")
        st.session_state.font_color = font_color

    font_family = st.selectbox(
        "Font", GOOGLE_FONTS,
        index=GOOGLE_FONTS.index(st.session_state.font_family),
        label_visibility="collapsed"
    )
    st.session_state.font_family = font_family

    sidebar_label("🎭 Chart Palette")
    pal_name = st.selectbox("Palette", list(PALETTES.keys()), label_visibility="collapsed")
    palette  = PALETTES[pal_name]
    pal_seq  = palette if isinstance(palette, list) else None

inject_css(
    st.session_state.bg_color,
    st.session_state.font_family,
    st.session_state.font_color,
)

# ============================================================
# TOP HEADER
# ============================================================

st.markdown(f"""
<div style="background:linear-gradient(135deg,{C_SURFACE} 0%,#0f1e30 60%,{C_SURFACE} 100%);
    border:1px solid {C_BORDER};border-radius:9px;padding:12px 18px;margin:0.5rem 0 1rem 0;
    display:flex;align-items:center;gap:14px;
    box-shadow:0 2px 18px rgba(0,0,0,.4),inset 0 1px 0 rgba(255,255,255,.03);">
    <div style="width:42px;height:42px;border-radius:8px;flex-shrink:0;
        background:linear-gradient(135deg,{C_BLUE},{C_CYAN});
        display:flex;align-items:center;justify-content:center;
        font-family:'Rajdhani',sans-serif;font-weight:800;font-size:15px;color:#fff;
        box-shadow:0 0 14px rgba(0,220,200,.3);">DA</div>
    <div>
        <div style="font-family:'Rajdhani',sans-serif;font-size:1.25rem;font-weight:700;
                    color:{C_BRIGHT};letter-spacing:.07em;text-transform:uppercase;line-height:1.1;">
            Advanced Data Analytics Dashboard
        </div>
        <div style="font-size:0.7rem;color:{C_MUTED};margin-top:2px;">
            {df.shape[0]:,} rows · {df.shape[1]} columns · {len(num_cols(df))} numeric · {len(cat_cols(df))} categorical · {len(date_cols(df))} datetime
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================

nc = num_cols(df)
cc = cat_cols(df)
dc = date_cols(df)

tab_ov, tab_ex, tab_cmp, tab_corr, tab_trend, tab_raw = st.tabs([
    "📋 Overview", "🔍 Explore", "🔀 Compare", "🔗 Correlation", "📈 Trends", "🗂 Raw Data"
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════

with tab_ov:
    tab_title("Dataset Overview & Intelligence Summary")

    total_rows  = df.shape[0]
    total_cols  = df.shape[1]
    miss_cells  = df.isnull().sum().sum()
    miss_pct    = f"{miss_cells / df.size * 100:.1f}%"
    dup_rows    = df.duplicated().sum()
    mem_kb      = df.memory_usage(deep=True).sum() / 1024

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    for col, lbl, val, sub, clr in [
        (k1, "Total Rows",      f"{total_rows:,}",    "",             C_CYAN),
        (k2, "Total Columns",   str(total_cols),       "",             C_BLUE),
        (k3, "Numeric Cols",    str(len(nc)),           "",             C_GREEN),
        (k4, "Categorical Cols",str(len(cc)),           "",             C_ORANGE),
        (k5, "Missing Cells",   miss_pct,               f"{miss_cells:,} cells", C_PURPLE),
        (k6, "Duplicate Rows",  str(dup_rows),          f"{mem_kb:.1f} KB",      C_RED),
    ]:
        col.markdown(kpi_card(lbl, val, sub, clr), unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    r1a, r1b = st.columns([1.6, 1])
    with r1a:
        section_label("Column Information", C_CYAN)
        col_info = pd.DataFrame({
            "Column":     df.columns.tolist(),
            "Data Type":  [str(df[c].dtype) for c in df.columns],
            "Non-Null":   [df[c].notna().sum() for c in df.columns],
            "Null Count": [df[c].isna().sum() for c in df.columns],
            "Unique":     [df[c].nunique() for c in df.columns],
            "Sample":     [str(df[c].dropna().iloc[0]) if df[c].notna().any() else "—" for c in df.columns],
        })
        st.dataframe(col_info, use_container_width=True, height=300)

    with r1b:
        section_label("Data Type Distribution", C_ORANGE)
        dtype_map = {
            "Numeric":     len(nc),
            "Categorical": len(cc),
            "DateTime":    len(dc),
            "Other":       total_cols - len(nc) - len(cc) - len(dc),
        }
        dtype_df = pd.DataFrame({"Type": list(dtype_map.keys()), "Count": list(dtype_map.values())})
        dtype_df = dtype_df[dtype_df["Count"] > 0]
        fig_dtype = px.pie(
            dtype_df, names="Type", values="Count", hole=0.52,
            color_discrete_sequence=[C_GREEN, C_ORANGE, C_BLUE, C_MUTED]
        )
        st.plotly_chart(style_fig(fig_dtype, 300), use_container_width=True)

    r2a, r2b = st.columns(2)
    with r2a:
        section_label("Numeric Columns — Statistical Summary", C_BLUE)
        if nc:
            desc = df[nc].describe().T.round(3)
            desc.insert(0, "Column", desc.index)
            desc = desc.reset_index(drop=True)
            st.dataframe(desc, use_container_width=True, height=260)
        else:
            st.info("No numeric columns found.")

    with r2b:
        section_label("Missing Values Per Column", C_PURPLE)
        miss_df = pd.DataFrame({
            "Column":    df.columns.tolist(),
            "Missing":   [df[c].isna().sum() for c in df.columns],
            "Missing %": [round(df[c].isna().sum() / len(df) * 100, 1) for c in df.columns],
        }).sort_values("Missing", ascending=False)
        miss_has = miss_df[miss_df["Missing"] > 0]
        if not miss_has.empty:
            fig_miss = px.bar(
                miss_has, x="Column", y="Missing %",
                color="Missing %", color_continuous_scale="OrRd",
                text="Missing %"
            )
            fig_miss.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(style_fig(fig_miss, 260), use_container_width=True)
        else:
            st.success("✅ No missing values found in the dataset!")

    section_label("🔮 Future Prediction Possibilities", C_CYAN)
    pred_cols = []
    if dc:
        pred_cols.append(f"**Time Series Forecasting** — {len(nc)} numeric columns can be forecast using datetime: {', '.join(dc[:3])}")
    if len(nc) >= 2:
        pred_cols.append(f"**Regression / Trend Prediction** — {len(nc)} numeric columns available for regression analysis")
    if "label" in df.columns or any("delay" in c.lower() or "risk" in c.lower() for c in df.columns):
        pred_cols.append("**Classification / Delay Prediction** — binary label column detected for supervised learning")
    if len(nc) >= 2:
        pred_cols.append(f"**Anomaly Detection** — {len(nc)} numeric columns can be scanned for outliers")
    if len(nc) >= 2:
        pred_cols.append(f"**Customer / Data Clustering** — K-Means segmentation possible with {len(nc)} numeric features")

    if pred_cols:
        for p in pred_cols:
            st.markdown(f"<div style='padding:6px 12px;margin:4px 0;background:rgba(0,220,200,0.05);border-left:2px solid {C_CYAN};border-radius:0 5px 5px 0;font-size:0.82rem;'>✦ {p}</div>", unsafe_allow_html=True)
    else:
        st.info("Upload a richer dataset to unlock prediction possibilities.")


# ══════════════════════════════════════════════════════════════
# TAB 2 — EXPLORE
# ══════════════════════════════════════════════════════════════

with tab_ex:
    tab_title("Single Column Explorer")

    ex_c1, ex_c2, ex_c3, ex_c4 = st.columns([2, 2, 2, 2])
    with ex_c1:
        ex_col = st.selectbox("Select Column", df.columns.tolist(), key="ex_col_sel")
    smart_charts = get_compatible_charts_for_col(df, ex_col)
    with ex_c2:
        ex_chart = st.selectbox("Chart Type", smart_charts, key="ex_chart_sel")
    with ex_c3:
        ex_color = st.color_picker("Graph Color", C_BLUE, key="ex_color_pick")
    with ex_c4:
        ex_color_by = st.selectbox("Color By", ["None"] + cc, key="ex_colorby")

    tb_c1, tb_c2, tb_c3 = st.columns([1, 2, 1])
    with tb_c1:
        show_topbot = st.toggle("Show Top & Lowest", value=False, key="ex_tb")
    with tb_c2:
        if show_topbot and ex_col in nc:
            tb_n_opt = st.selectbox(
                "Select Range",
                ["Top 5 & Lowest 5", "Top 10 & Lowest 10", "Top 20 & Lowest 20", "Custom"],
                key="ex_tb_n"
            )
            if tb_n_opt == "Custom":
                tb_n = st.number_input("Enter N", min_value=1, max_value=500, value=10, key="ex_tb_custom")
            else:
                tb_n = int(tb_n_opt.split()[1])
        else:
            tb_n = 10
            tb_n_opt = "Top 10 & Lowest 10"
    with tb_c3:
        pass

    color_arg = None if ex_color_by == "None" else ex_color_by
    is_num    = ex_col in nc
    fig       = None

    try:
        plot_df = df.copy()
        if show_topbot and is_num:
            n_val = tb_n if "Custom" in tb_n_opt else int(tb_n_opt.split()[1])
            sorted_asc = plot_df.sort_values(ex_col)
            bot = sorted_asc.head(n_val)
            top = sorted_asc.tail(n_val)
            plot_df = pd.concat([bot, top]).drop_duplicates()
            plot_df["_rank"] = ["Lowest"] * len(bot) + ["Top"] * len(top)
            color_arg_tb = "_rank"
        else:
            color_arg_tb = color_arg

        if ex_chart == "Histogram" and is_num:
            bins = st.slider("Bins", 10, 120, 40, key="ex_bins_sl")
            fig = px.histogram(plot_df, x=ex_col, color=color_arg_tb, nbins=bins, marginal="box",
                               color_discrete_sequence=([ex_color] if not color_arg_tb else pal_seq))
        elif ex_chart == "Bar":
            vc = plot_df[ex_col].value_counts().head(30).reset_index()
            vc.columns = [ex_col, "Count"]
            fig = px.bar(vc, x=ex_col, y="Count", color_discrete_sequence=[ex_color])
        elif ex_chart == "Horizontal Bar":
            vc = plot_df[ex_col].value_counts().head(30).reset_index()
            vc.columns = [ex_col, "Count"]
            fig = px.bar(vc, x="Count", y=ex_col, orientation="h", color_discrete_sequence=[ex_color])
        elif ex_chart == "Box" and is_num:
            fig = px.box(plot_df, y=ex_col, color=color_arg_tb, color_discrete_sequence=pal_seq or [ex_color])
        elif ex_chart == "Violin" and is_num:
            fig = px.violin(plot_df, y=ex_col, color=color_arg_tb, box=True, color_discrete_sequence=pal_seq or [ex_color])
        elif ex_chart == "Line" and is_num:
            fig = px.line(plot_df.reset_index(), x="index", y=ex_col, color_discrete_sequence=[ex_color])
        elif ex_chart == "Area" and is_num:
            fig = px.area(plot_df.reset_index(), x="index", y=ex_col, color_discrete_sequence=[ex_color])
        elif ex_chart == "Pie":
            vc = plot_df[ex_col].value_counts().head(15).reset_index()
            vc.columns = [ex_col, "Count"]
            fig = px.pie(vc, names=ex_col, values="Count", hole=0, color_discrete_sequence=pal_seq or [ex_color])
        elif ex_chart == "Donut":
            vc = plot_df[ex_col].value_counts().head(15).reset_index()
            vc.columns = [ex_col, "Count"]
            fig = px.pie(vc, names=ex_col, values="Count", hole=0.5, color_discrete_sequence=pal_seq or [ex_color])
        elif ex_chart == "Treemap":
            vc = plot_df[ex_col].value_counts().head(30).reset_index()
            vc.columns = [ex_col, "Count"]
            fig = px.treemap(vc, path=[ex_col], values="Count", color_discrete_sequence=pal_seq or [ex_color])
        else:
            st.info(f"'{ex_chart}' is not compatible with **{ex_col}**.")
    except Exception as e:
        st.error(f"Chart error: {e}")

    if fig:
        panel_header(f"{ex_col}  ·  {ex_chart}")
        st.plotly_chart(style_fig(fig, 460), use_container_width=True)

        if show_topbot and is_num:
            n_val = tb_n if "Custom" in tb_n_opt else int(tb_n_opt.split()[1])
            t_a, t_b = st.columns(2)
            with t_a:
                section_label(f"Top {n_val} Values", C_GREEN)
                top_df = df.nlargest(n_val, ex_col)[[ex_col]].reset_index(drop=True)
                top_df.index += 1
                st.dataframe(top_df, use_container_width=True, height=200)
            with t_b:
                section_label(f"Lowest {n_val} Values", C_RED)
                bot_df = df.nsmallest(n_val, ex_col)[[ex_col]].reset_index(drop=True)
                bot_df.index += 1
                st.dataframe(bot_df, use_container_width=True, height=200)

    if is_num:
        section_label("Statistical Summary", C_CYAN)
        s = df[ex_col].describe()
        sc = st.columns(6)
        for col_w, (k, v) in zip(sc, [
            ("Mean",    f"{s['mean']:.2f}"),
            ("Median",  f"{df[ex_col].median():.2f}"),
            ("Std Dev", f"{s['std']:.2f}"),
            ("Min",     f"{s['min']:.2f}"),
            ("Max",     f"{s['max']:.2f}"),
            ("Nulls",   str(df[ex_col].isna().sum())),
        ]):
            col_w.metric(k, v)


# ══════════════════════════════════════════════════════════════
# TAB 3 — COMPARE
# ══════════════════════════════════════════════════════════════

with tab_cmp:
    tab_title("Multi-Column Comparison")

    cmp_r1a, cmp_r1b, cmp_r1c = st.columns(3)
    with cmp_r1a:
        x_col = st.selectbox("X Axis", df.columns.tolist(), key="cmp_x")
    compat_y = compatible_y_for_x(df, x_col)
    with cmp_r1b:
        y_col = st.selectbox("Y Axis", ["— Select —"] + compat_y, key="cmp_y")
    y_selected = y_col != "— Select —"
    if y_selected:
        remaining = [c for c in df.columns if c not in [x_col, y_col]]
        z_options = ["None"] + remaining
    else:
        z_options = ["None"]
    with cmp_r1c:
        z_col = st.selectbox("Z Axis (optional)", z_options, key="cmp_z", disabled=not y_selected)

    applicable_cols = [c for c in [x_col, y_col if y_selected else None] if c]
    agg_opts = get_applicable_agg(df, applicable_cols)
    cmp_r2a, cmp_r2b, cmp_r2c = st.columns(3)
    with cmp_r2a:
        cmp_agg = st.selectbox("Aggregation", agg_opts, key="cmp_agg")
    y_arg = y_col if y_selected else None
    z_arg = z_col if z_col != "None" else None
    compat_charts = get_compatible_charts_for_axes(df, x_col, y_arg, z_arg)
    with cmp_r2b:
        cmp_chart = st.selectbox("Chart Type", compat_charts, key="cmp_chart")
    with cmp_r2c:
        cmp_color = st.color_picker("Graph Color", C_ORANGE, key="cmp_color")

    tb2_c1, tb2_c2 = st.columns([1, 3])
    with tb2_c1:
        show_tb2 = st.toggle("Show Top & Lowest", value=False, key="cmp_tb")
    with tb2_c2:
        if show_tb2 and y_arg and y_arg in nc:
            tb2_opt = st.selectbox(
                "Range", ["Top 5 & Lowest 5", "Top 10 & Lowest 10", "Top 20 & Lowest 20", "Custom"],
                key="cmp_tb_n"
            )
            if tb2_opt == "Custom":
                tb2_n = st.number_input("N", min_value=1, max_value=500, value=10, key="cmp_tb_custom")
            else:
                tb2_n = int(tb2_opt.split()[1])
        else:
            tb2_n = 10
            tb2_opt = "Top 10 & Lowest 10"

    fig = None
    if y_selected and y_arg:
        try:
            plot_df = df.copy()
            if show_tb2 and y_arg in nc:
                n_val = tb2_n
                top_rows = plot_df.nlargest(n_val, y_arg)
                bot_rows = plot_df.nsmallest(n_val, y_arg)
                plot_df  = pd.concat([top_rows, bot_rows]).drop_duplicates()

            if cmp_chart == "Scatter":
                fig = px.scatter(plot_df, x=x_col, y=y_arg, color=z_arg,
                                 size=z_arg if z_arg and z_arg in nc else None,
                                 opacity=0.65, color_discrete_sequence=pal_seq or [cmp_color],
                                 trendline="ols" if x_col in nc else None)
            elif cmp_chart == "Bubble" and z_arg and z_arg in nc:
                fig = px.scatter(plot_df, x=x_col, y=y_arg, size=z_arg,
                                 color=z_arg, color_continuous_scale="Viridis", opacity=0.7)
            elif cmp_chart in ("Bar", "Horizontal Bar"):
                g = [x_col]
                agg_df = plot_df.groupby(g)[y_arg].agg(cmp_agg).reset_index()
                oh = "h" if cmp_chart == "Horizontal Bar" else "v"
                fig = px.bar(agg_df,
                             x=y_arg if oh=="h" else x_col,
                             y=x_col if oh=="h" else y_arg,
                             orientation=oh, color_discrete_sequence=[cmp_color])
            elif cmp_chart == "Line":
                agg_df = plot_df.groupby(x_col)[y_arg].agg(cmp_agg).reset_index()
                fig = px.line(agg_df, x=x_col, y=y_arg, color_discrete_sequence=[cmp_color])
                fig.update_traces(line_width=2)
            elif cmp_chart == "Area":
                agg_df = plot_df.groupby(x_col)[y_arg].agg(cmp_agg).reset_index()
                fig = px.area(agg_df, x=x_col, y=y_arg, color_discrete_sequence=[cmp_color])
            elif cmp_chart == "Box":
                fig = px.box(plot_df, x=x_col, y=y_arg, color_discrete_sequence=pal_seq or [cmp_color])
            elif cmp_chart == "Violin":
                fig = px.violin(plot_df, x=x_col, y=y_arg, box=True, color_discrete_sequence=pal_seq or [cmp_color])
            elif cmp_chart == "Heatmap":
                if z_arg:
                    pivot = pd.pivot_table(plot_df, values=z_arg, index=x_col, columns=y_arg, aggfunc=cmp_agg)
                else:
                    pivot = pd.pivot_table(plot_df, values=y_arg, index=x_col, aggfunc=cmp_agg)
                fig = px.imshow(pivot, text_auto=".2f",
                                color_continuous_scale=palette if isinstance(palette, str) else "Blues")
            elif cmp_chart == "Funnel":
                agg_df = plot_df.groupby(x_col)[y_arg].agg(cmp_agg).reset_index()
                fig = px.funnel(agg_df, x=y_arg, y=x_col)
            elif cmp_chart == "Treemap":
                path = [x_col] + ([z_arg] if z_arg else [])
                agg_df = plot_df.groupby(path).size().reset_index(name="Count")
                fig = px.treemap(agg_df, path=path, values="Count", color_discrete_sequence=pal_seq or [cmp_color])
            elif cmp_chart == "Pie":
                agg_df = plot_df.groupby(x_col)[y_arg].agg(cmp_agg).reset_index()
                fig = px.pie(agg_df, names=x_col, values=y_arg, color_discrete_sequence=pal_seq or [cmp_color])
            elif cmp_chart == "Histogram":
                fig = px.histogram(plot_df, x=x_col, y=y_arg, color_discrete_sequence=[cmp_color])
        except Exception as e:
            st.error(f"Chart error: {e}")

    if fig:
        lbl = f"{x_col} vs {y_arg}" + (f" vs {z_arg}" if z_arg else "") + f" · {cmp_chart} · {cmp_agg}"
        panel_header(lbl)
        st.plotly_chart(style_fig(fig, 500), use_container_width=True)
        if show_tb2 and y_arg and y_arg in nc:
            n_val = tb2_n
            t_a, t_b = st.columns(2)
            with t_a:
                section_label(f"Top {n_val} — {y_arg}", C_GREEN)
                top_tbl = df.nlargest(n_val, y_arg)[[x_col, y_arg]].reset_index(drop=True)
                top_tbl.index += 1
                st.dataframe(top_tbl, use_container_width=True, height=200)
            with t_b:
                section_label(f"Lowest {n_val} — {y_arg}", C_RED)
                bot_tbl = df.nsmallest(n_val, y_arg)[[x_col, y_arg]].reset_index(drop=True)
                bot_tbl.index += 1
                st.dataframe(bot_tbl, use_container_width=True, height=200)
    elif not y_selected:
        st.info("👆 Select a **Y Axis** column to begin comparison.")


# ══════════════════════════════════════════════════════════════
# TAB 4 — CORRELATION
# ══════════════════════════════════════════════════════════════

with tab_corr:
    tab_title("Correlation Analysis (up to 6 columns)")

    corr_cols_selected = []
    ca, cb = st.columns([2, 4])
    with ca:
        c1_opts = nc
        col1 = st.selectbox("Column 1", ["— Select —"] + c1_opts, key="corr_c1")

    if col1 != "— Select —":
        corr_cols_selected.append(col1)
        c2_opts = compatible_corr_cols(df, corr_cols_selected)
        col2 = st.selectbox("Column 2", ["— Select —"] + c2_opts, key="corr_c2")

        if col2 != "— Select —":
            corr_cols_selected.append(col2)
            method_opts = corr_method_applicable(df, corr_cols_selected)
            corr_method = st.radio("Correlation Method", method_opts, horizontal=True, key="corr_method")

            for i in range(3, 7):
                c_opts = compatible_corr_cols(df, corr_cols_selected)
                if not c_opts:
                    break
                c_sel = st.selectbox(f"Column {i} (optional)", ["— Stop here —"] + c_opts, key=f"corr_c{i}")
                if c_sel == "— Stop here —":
                    break
                corr_cols_selected.append(c_sel)

            if len(corr_cols_selected) >= 2:
                corr_data = df[corr_cols_selected].dropna()
                corr_matrix = corr_data.corr(method=corr_method)

                cr1, cr2 = st.columns([3, 1])
                with cr1:
                    panel_header(f"{corr_method.capitalize()} Correlation — {len(corr_cols_selected)} columns")
                    fig_corr = px.imshow(
                        corr_matrix, zmin=-1, zmax=1,
                        color_continuous_scale=palette if isinstance(palette, str) else "RdBu",
                        text_auto=".3f",
                    )
                    st.plotly_chart(style_fig(fig_corr, 480), use_container_width=True)

                with cr2:
                    section_label("Strongest Pairs", C_ORANGE)
                    pairs = (
                        corr_matrix.abs().unstack()
                        .sort_values(ascending=False)
                        .drop_duplicates()
                        .reset_index()
                    )
                    pairs.columns = ["Col A", "Col B", "|r|"]
                    pairs = pairs[pairs["|r|"] < 1].head(12)
                    pairs["|r|"] = pairs["|r|"].round(3)
                    st.dataframe(pairs, use_container_width=True, height=380)

                    section_label("Interpretation", C_CYAN)
                    for _, row in pairs.head(5).iterrows():
                        r = corr_matrix.loc[row["Col A"], row["Col B"]]
                        strength = "Strong" if abs(r) > 0.7 else "Moderate" if abs(r) > 0.4 else "Weak"
                        direction = "positive" if r > 0 else "negative"
                        st.markdown(
                            f"<div style='font-size:0.72rem;padding:3px 0;color:{C_TEXT};'>"
                            f"<span style='color:{C_CYAN};'>{row['Col A']}</span> ↔ "
                            f"<span style='color:{C_ORANGE};'>{row['Col B']}</span>: "
                            f"{strength} {direction} ({r:.3f})</div>",
                            unsafe_allow_html=True
                        )
        else:
            st.info("👆 Select **Column 2** to continue.")
    else:
        st.info("👆 Select **Column 1** to begin correlation analysis.")

# ══════════════════════════════════════════════════════════════
# TAB 5 — TRENDS  (SMART DATETIME + BYPASS MODE)
# ══════════════════════════════════════════════════════════════

with tab_trend:
    tab_title("Time Series Trends & Future Predictions")

    freq_map = {"D": "Daily", "W": "Weekly", "ME": "Monthly", "QE": "Quarterly", "YE": "Yearly"}

    # ── STEP 1: Auto-detect & coerce datetime columns ──
    dc_live = []
    df_trend_src = df.copy()

    for col in df_trend_src.columns:
        if pd.api.types.is_datetime64_any_dtype(df_trend_src[col]):
            dc_live.append(col)
        elif df_trend_src[col].dtype == object:
            sample = df_trend_src[col].dropna().head(100)
            try:
                parsed = pd.to_datetime(sample, infer_datetime_format=True, errors="raise")
                if parsed.notna().sum() >= len(sample) * 0.6:
                    df_trend_src[col] = pd.to_datetime(
                        df_trend_src[col], infer_datetime_format=True, errors="coerce"
                    )
                    dc_live.append(col)
            except Exception:
                pass
        elif pd.api.types.is_numeric_dtype(df_trend_src[col]):
            sv = df_trend_src[col].dropna().head(20)
            if len(sv) and sv.between(1e9, 2e10).all():
                try:
                    df_trend_src[col] = pd.to_datetime(df_trend_src[col], unit="s", errors="coerce")
                    dc_live.append(col)
                except Exception:
                    pass
            elif len(sv) and sv.between(1e12, 2e13).all():
                try:
                    df_trend_src[col] = pd.to_datetime(df_trend_src[col], unit="ms", errors="coerce")
                    dc_live.append(col)
                except Exception:
                    pass

    # ── STEP 2: If still no datetime, create synthetic index-based date ──
    SYNTHETIC_DATE_COL = "__synthetic_date__"
    using_synthetic = False

    if not dc_live:
        st.info(
            "⚠️ No datetime column found. A synthetic sequential date index has been created "
            "so you can still explore trends and run forecasts."
        )
        df_trend_src[SYNTHETIC_DATE_COL] = pd.date_range(
            start="2020-01-01", periods=len(df_trend_src), freq="D"
        )
        dc_live = [SYNTHETIC_DATE_COL]
        using_synthetic = True
    else:
        st.success(f"✅ Datetime column(s) detected: **{', '.join(dc_live)}**")

    nc_t = num_cols(df_trend_src)
    cc_t = cat_cols(df_trend_src)

    # ── STEP 3: Detect available frequencies based on date range ──
    def get_valid_freqs(df_src, date_col):
        """Only return frequencies that make sense for the date span."""
        try:
            dmin = df_src[date_col].dropna().min()
            dmax = df_src[date_col].dropna().max()
            span_days = (dmax - dmin).days
            valid = {}
            if span_days >= 1:   valid["D"]  = "Daily"
            if span_days >= 7:   valid["W"]  = "Weekly"
            if span_days >= 30:  valid["ME"] = "Monthly"
            if span_days >= 90:  valid["QE"] = "Quarterly"
            if span_days >= 365: valid["YE"] = "Yearly"
            return valid if valid else {"D": "Daily"}
        except Exception:
            return {"D": "Daily"}

    # ════════════════════════════════════════
    # SECTION A — TIME-BASED ANALYSIS OPTIONS
    # ════════════════════════════════════════
    section_label("⚙️ Time-Based Analysis Options", C_CYAN)

    tr1, tr2, tr3, tr4 = st.columns(4)
    with tr1:
        date_col_t = st.selectbox("Date Column", dc_live, key="tr_date")
        if using_synthetic:
            st.caption("📅 Synthetic date index (row order)")

    valid_freqs = get_valid_freqs(df_trend_src, date_col_t)

    with tr2:
        multi_metrics = st.multiselect(
            "Metric(s) — Y Axis", nc_t,
            default=nc_t[:1] if nc_t else [],
            key="tr_metrics", max_selections=5
        )
        metric_col = multi_metrics[0] if multi_metrics else (nc_t[0] if nc_t else None)
    with tr3:
        freq = st.selectbox(
            "Aggregation Frequency",
            list(valid_freqs.keys()),
            format_func=lambda x: valid_freqs[x],
            key="tr_freq"
        )
    with tr4:
        tagg = st.selectbox(
            "Aggregation Function",
            ["sum", "mean", "count", "max", "min", "std"],
            key="tr_agg"
        )

    tr_b1, tr_b2 = st.columns(2)
    with tr_b1:
        split_by = st.selectbox("Split By (optional)", ["None"] + cc_t, key="tr_split")

    # Date Range
    section_label("📅 Date Range Filter", C_BLUE)
    try:
        tmp_dates = df_trend_src[date_col_t].dropna()
        min_d = tmp_dates.min().date()
        max_d = tmp_dates.max().date()
        dr_c1, dr_c2 = st.columns(2)
        with dr_c1:
            start_d = st.date_input("Start Date", value=min_d, min_value=min_d, max_value=max_d, key="tr_start")
        with dr_c2:
            end_d = st.date_input("End Date", value=max_d, min_value=min_d, max_value=max_d, key="tr_end")
        tmp = df_trend_src[
            (df_trend_src[date_col_t] >= pd.Timestamp(start_d)) &
            (df_trend_src[date_col_t] <= pd.Timestamp(end_d))
        ].dropna(subset=[date_col_t]).copy()
    except Exception:
        tmp = df_trend_src.dropna(subset=[date_col_t]).copy()
        start_d, end_d = None, None

    # ════════════════════════════════════════
    # SECTION B — TREND OPTIONS
    # ════════════════════════════════════════
    section_label("📊 Trend Options", C_ORANGE)

    topt_c1, topt_c2, topt_c3 = st.columns(3)
    with topt_c1:
        show_ma = st.toggle("Moving Average", value=False, key="tr_ma")
        if show_ma:
            ma_window_sel = st.selectbox("MA Window", [7, 14, 30, 60, 90, "Custom"], key="tr_ma_win")
            ma_window = st.number_input("Custom window", min_value=2, max_value=365, value=30, key="tr_ma_custom") \
                if ma_window_sel == "Custom" else int(ma_window_sel)
        else:
            ma_window = 7
    with topt_c2:
        show_rolling_std = st.toggle("Rolling Std Dev Band", value=False, key="tr_rollstd")
        show_growth      = st.toggle("Growth Rate %",        value=False, key="tr_growth")
    with topt_c3:
        show_seasonality  = st.toggle("Seasonality Detection", value=False, key="tr_season")
        show_multi_model  = st.toggle("Multi-Model Forecast",  value=False, key="tr_multimodel")

    # ════════════════════════════════════════
    # SECTION C — TOP & LOWEST SELECTOR
    # ════════════════════════════════════════
    section_label("🏆 Top & Lowest Selector", C_GREEN)
    tb3_c1, tb3_c2 = st.columns([1, 3])
    with tb3_c1:
        show_tb3 = st.toggle("Show Top & Lowest Periods", value=False, key="trend_tb")
    with tb3_c2:
        if show_tb3:
            tb3_opt = st.selectbox(
                "Range", ["Top 5 & Lowest 5", "Top 10 & Lowest 10", "Top 20 & Lowest 20", "Custom"],
                key="trend_tb_n"
            )
            tb3_n = st.number_input("N", min_value=1, max_value=500, value=10, key="trend_tb_custom") \
                if tb3_opt == "Custom" else int(tb3_opt.split()[1])
        else:
            tb3_n = 5

    # ════════════════════════════════════════
    # SECTION D — PERIOD COMPARISON OPTIONS
    # ════════════════════════════════════════
    section_label("📅 Period Comparison Options", C_PURPLE)

    # Only offer comparisons that fit the date span
    avail_comparisons = []
    try:
        span = (df_trend_src[date_col_t].max() - df_trend_src[date_col_t].min()).days
        if span >= 60:  avail_comparisons.append("Month-over-Month")
        if span >= 180: avail_comparisons.append("Quarter-over-Quarter")
        if span >= 730: avail_comparisons.append("Year-over-Year")
    except Exception:
        pass

    if avail_comparisons:
        period_cmp = st.selectbox("Compare By", avail_comparisons, key="tr_yoy")
    else:
        period_cmp = None
        st.caption("📅 Date range too short for period comparison (need ≥ 60 days).")

    # ════════════════════════════════════════
    # SECTION E — FUTURE PREDICTION OPTIONS
    # ════════════════════════════════════════
    section_label("🔮 Future Prediction Options", C_CYAN)
    fp1, fp2, fp3 = st.columns(3)
    with fp1:
        model_opts = ["Linear Regression"]
        if STATSMODELS_OK: model_opts += ["Exponential Smoothing", "ARIMA"]
        if PROPHET_OK:     model_opts += ["Prophet"]
        model_opts += ["Simple Moving Average"]

        selected_models = st.multiselect(
            "Forecast Models", model_opts,
            default=model_opts[:2], key="pred_models_multi"
        ) if show_multi_model else [st.selectbox("Forecast Model", model_opts, key="pred_model")]

    with fp2:
        horizon_opts = {"7 Days": 7, "30 Days": 30, "90 Days": 90,
                        "6 Months": 180, "1 Year": 365, "Custom": 0}
        horizon_sel  = st.selectbox("Forecast Horizon", list(horizon_opts.keys()), key="pred_horizon")
        horizon_days = st.number_input("Days", min_value=1, max_value=1095, value=30, key="pred_days") \
            if horizon_sel == "Custom" else horizon_opts[horizon_sel]
    with fp3:
        show_ci = st.toggle("Confidence Interval (95%)", value=True, key="pred_ci")

    run_pred = st.button("▶ Run Forecast", key="run_forecast")

    st.markdown("---")

    # ════════════════════════════════════════════════════════
    # DISPLAY SECTION 1 — MAIN TREND CHART
    # ════════════════════════════════════════════════════════
    agg_df = pd.DataFrame()

    if metric_col:
        section_label("📈 Trend Chart", C_CYAN)
        try:
            if split_by != "None":
                agg_df = (
                    tmp.groupby([pd.Grouper(key=date_col_t, freq=freq), split_by])[metric_col]
                    .agg(tagg).reset_index()
                )
                fig_trend = px.line(
                    agg_df, x=date_col_t, y=metric_col, color=split_by,
                    color_discrete_sequence=pal_seq or [C_BLUE, C_ORANGE, C_PURPLE, C_CYAN],
                    title=f"{tagg.capitalize()} of {metric_col}"
                )
                fig_trend.update_traces(line_width=2)
                trend_dir, trend_clr = "— N/A (split view)", C_MUTED
            else:
                agg_df = tmp.resample(freq, on=date_col_t)[metric_col].agg(tagg).reset_index()
                if show_growth:
                    agg_df["growth"] = agg_df[metric_col].pct_change() * 100

                fig_trend = go.Figure()
                all_metrics   = multi_metrics if multi_metrics else [metric_col]
                colors_cycle  = [C_BLUE, C_ORANGE, C_PURPLE, C_CYAN, C_GREEN]

                for idx_m, m_col in enumerate(all_metrics):
                    m_agg = tmp.resample(freq, on=date_col_t)[m_col].agg(tagg).reset_index()
                    clr   = colors_cycle[idx_m % len(colors_cycle)]
                    fig_trend.add_trace(go.Scatter(
                        x=m_agg[date_col_t], y=m_agg[m_col],
                        mode="lines+markers", name=m_col,
                        line=dict(color=clr, width=2),
                        fill="tozeroy" if idx_m == 0 else "none",
                        fillcolor="rgba(26,109,212,0.06)" if idx_m == 0 else None,
                        marker=dict(size=4),
                    ))

                if show_ma and len(agg_df) > ma_window:
                    agg_df["MA"] = agg_df[metric_col].rolling(int(ma_window)).mean()
                    fig_trend.add_trace(go.Scatter(
                        x=agg_df[date_col_t], y=agg_df["MA"],
                        mode="lines", name=f"{ma_window}-Period MA",
                        line=dict(color=C_ORANGE, width=2, dash="dash"),
                    ))

                if show_rolling_std and len(agg_df) > 7:
                    rw = max(7, ma_window if show_ma else 7)
                    agg_df["roll_std"]  = agg_df[metric_col].rolling(rw).std()
                    agg_df["roll_mean"] = agg_df[metric_col].rolling(rw).mean()
                    agg_df["upper_std"] = agg_df["roll_mean"] + agg_df["roll_std"]
                    agg_df["lower_std"] = agg_df["roll_mean"] - agg_df["roll_std"]
                    fig_trend.add_trace(go.Scatter(
                        x=list(agg_df[date_col_t]) + list(agg_df[date_col_t][::-1]),
                        y=list(agg_df["upper_std"]) + list(agg_df["lower_std"][::-1]),
                        fill="toself", fillcolor="rgba(108,63,199,0.12)",
                        line=dict(color="rgba(0,0,0,0)"),
                        name="Rolling ±1 Std Dev",
                    ))

                if show_growth and "growth" in agg_df.columns:
                    fig_trend.add_trace(go.Bar(
                        x=agg_df[date_col_t], y=agg_df["growth"],
                        name="Growth Rate %",
                        marker_color=[C_GREEN if v >= 0 else C_RED for v in agg_df["growth"].fillna(0)],
                        opacity=0.5, yaxis="y2",
                    ))
                    fig_trend.update_layout(yaxis2=dict(
                        title="Growth %", overlaying="y", side="right",
                        showgrid=False,
                        tickfont=dict(size=9, color=C_MUTED),
                        titlefont=dict(size=9, color=C_MUTED),
                    ))

                if len(agg_df) >= 4:
                    recent = agg_df[metric_col].dropna()
                    q = max(1, len(recent) // 4)
                    avg_recent = recent.tail(q).mean()
                    avg_early  = recent.head(q).mean()
                    if avg_recent > avg_early * 1.02:
                        trend_dir, trend_clr = "📈 RISING",  C_GREEN
                    elif avg_recent < avg_early * 0.98:
                        trend_dir, trend_clr = "📉 FALLING", C_RED
                    else:
                        trend_dir, trend_clr = "➡️ STABLE",  C_ORANGE
                else:
                    trend_dir, trend_clr = "— N/A", C_MUTED

                if len(agg_df) > 1:
                    peak_idx  = agg_df[metric_col].idxmax()
                    dip_idx   = agg_df[metric_col].idxmin()
                    fig_trend.add_trace(go.Scatter(
                        x=[agg_df.loc[peak_idx, date_col_t]], y=[agg_df.loc[peak_idx, metric_col]],
                        mode="markers+text", name="Peak",
                        marker=dict(size=14, color=C_GREEN, symbol="triangle-up",
                                    line=dict(color="white", width=1)),
                        text=[f"Peak: {agg_df.loc[peak_idx, metric_col]:,.0f}"],
                        textposition="top center", textfont=dict(size=9, color=C_GREEN),
                    ))
                    fig_trend.add_trace(go.Scatter(
                        x=[agg_df.loc[dip_idx, date_col_t]], y=[agg_df.loc[dip_idx, metric_col]],
                        mode="markers+text", name="Dip",
                        marker=dict(size=14, color=C_RED, symbol="triangle-down",
                                    line=dict(color="white", width=1)),
                        text=[f"Dip: {agg_df.loc[dip_idx, metric_col]:,.0f}"],
                        textposition="bottom center", textfont=dict(size=9, color=C_RED),
                    ))

            if show_seasonality and split_by == "None" and not agg_df.empty and len(agg_df) >= 14:
                try:
                    vals = agg_df[metric_col].fillna(method="ffill").values
                    if len(vals) >= 10:
                        detrended = vals - np.poly1d(
                            np.polyfit(np.arange(len(vals)), vals, 1)
                        )(np.arange(len(vals)))
                        autocorr = np.correlate(detrended, detrended, mode="full")
                        autocorr = autocorr[autocorr.size // 2:]
                        autocorr /= autocorr[0] if autocorr[0] != 0 else 1
                        peaks_ac = [
                            i for i in range(2, min(len(autocorr) - 1, len(vals) // 2))
                            if autocorr[i] > autocorr[i - 1]
                            and autocorr[i] > autocorr[i + 1]
                            and autocorr[i] > 0.1
                        ]
                        if peaks_ac:
                            st.markdown(
                                f"<div style='padding:6px 12px;margin:6px 0;"
                                f"background:rgba(108,63,199,0.1);"
                                f"border-left:2px solid {C_PURPLE};"
                                f"border-radius:0 5px 5px 0;font-size:0.8rem;'>"
                                f"🔁 <b>Seasonality detected</b>: Dominant cycle every "
                                f"~<b>{peaks_ac[0]} {valid_freqs.get(freq,'').lower()} periods</b>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                except Exception:
                    pass

            if split_by == "None":
                st.markdown(
                    f"<div style='display:inline-block;padding:2px 10px;margin-bottom:6px;"
                    f"background:rgba(0,0,0,0.2);border:1px solid {trend_clr}33;"
                    f"border-radius:4px;font-size:0.75rem;color:{trend_clr};font-weight:700;'>"
                    f"Trend Direction: {trend_dir}</div>",
                    unsafe_allow_html=True,
                )

            panel_header(
                f"Trend — {', '.join(multi_metrics) if multi_metrics else metric_col} "
                f"({valid_freqs.get(freq, freq)} · {tagg})"
                + (f"  |  {trend_dir}" if split_by == "None" else "")
                + ("  |  📅 Synthetic Index" if using_synthetic else "")
            )
            st.plotly_chart(style_fig(fig_trend, 440), use_container_width=True)

        except Exception as e:
            st.error(f"Trend error: {e}")

    # ════════════════════════════════════════════════════════
    # DISPLAY SECTION 2 — TOP & LOWEST PERIODS TABLE
    # ════════════════════════════════════════════════════════
    if show_tb3 and not agg_df.empty and metric_col and metric_col in agg_df.columns:
        section_label(f"🏆 Top & Lowest {tb3_n} Periods", C_GREEN)
        t_a, t_b = st.columns(2)
        with t_a:
            section_label(f"Top {tb3_n} Periods", C_GREEN)
            top_t = agg_df.nlargest(tb3_n, metric_col)[[date_col_t, metric_col]].reset_index(drop=True)
            top_t.index += 1; top_t.columns = ["Date", metric_col]
            st.dataframe(top_t, use_container_width=True, height=200)
        with t_b:
            section_label(f"Lowest {tb3_n} Periods", C_RED)
            bot_t = agg_df.nsmallest(tb3_n, metric_col)[[date_col_t, metric_col]].reset_index(drop=True)
            bot_t.index += 1; bot_t.columns = ["Date", metric_col]
            st.dataframe(bot_t, use_container_width=True, height=200)

    # ════════════════════════════════════════════════════════
    # DISPLAY SECTION 3 — PERIOD COMPARISON CHART
    # ════════════════════════════════════════════════════════
    if metric_col and period_cmp:
        st.markdown("---")
        section_label(f"📅 {period_cmp} Comparison", C_PURPLE)
        try:
            tmp2 = df_trend_src.dropna(subset=[date_col_t]).copy()
            if period_cmp == "Year-over-Year":
                tmp2["Period"] = tmp2[date_col_t].dt.year.astype(str)
                tmp2["Sub"]    = tmp2[date_col_t].dt.month
                grp = tmp2.groupby(["Sub", "Period"])[metric_col].agg(tagg).reset_index()
                fig_yoy = px.line(
                    grp, x="Sub", y=metric_col, color="Period",
                    color_discrete_sequence=pal_seq or [C_BLUE, C_ORANGE, C_PURPLE, C_CYAN],
                )
                fig_yoy.update_xaxes(
                    title="Month", tickvals=list(range(1, 13)),
                    ticktext=["Jan","Feb","Mar","Apr","May","Jun",
                              "Jul","Aug","Sep","Oct","Nov","Dec"],
                )
            elif period_cmp == "Month-over-Month":
                tmp2["Period"] = tmp2[date_col_t].dt.to_period("M").astype(str)
                grp = tmp2.groupby("Period")[metric_col].agg(tagg).reset_index()
                grp["growth"] = grp[metric_col].pct_change() * 100
                fig_yoy = px.bar(
                    grp, x="Period", y="growth",
                    color="growth", color_continuous_scale="RdYlGn",
                    labels={"growth": "MoM Growth %"},
                )
            else:
                tmp2["Period"] = tmp2[date_col_t].dt.to_period("Q").astype(str)
                grp = tmp2.groupby("Period")[metric_col].agg(tagg).reset_index()
                fig_yoy = px.bar(grp, x="Period", y=metric_col, color_discrete_sequence=[C_PURPLE])

            panel_header(period_cmp)
            st.plotly_chart(style_fig(fig_yoy, 300), use_container_width=True)
        except Exception as e:
            st.warning(f"Period comparison error: {e}")

    # ════════════════════════════════════════════════════════
    # DISPLAY SECTION 4 — FUTURE PREDICTION
    # ════════════════════════════════════════════════════════
    if run_pred:
        st.markdown("---")
        section_label("🔮 Forecast Results", C_CYAN)

        if not metric_col:
            st.warning("Select at least one metric column above.")
        else:
            try:
                base_df  = df_trend_src.dropna(subset=[date_col_t]).copy()
                base_agg = (
                    base_df.resample("D", on=date_col_t)[metric_col]
                    .agg(tagg).reset_index().dropna()
                )

                if len(base_agg) < 10:
                    st.warning("Not enough data points for forecasting (need at least 10).")
                else:
                    hist_x = np.arange(len(base_agg))
                    hist_y = base_agg[metric_col].values
                    future_dates = pd.date_range(
                        start=base_agg[date_col_t].iloc[-1] + pd.Timedelta(days=1),
                        periods=horizon_days, freq="D",
                    )
                    future_x = np.arange(len(base_agg), len(base_agg) + horizon_days)

                    fig_pred = go.Figure()
                    fig_pred.add_trace(go.Scatter(
                        x=base_agg[date_col_t], y=hist_y,
                        mode="lines", name="Historical",
                        line=dict(color=C_BLUE, width=2),
                        fill="tozeroy", fillcolor="rgba(26,109,212,0.06)",
                    ))
                    fig_pred.add_vline(
                        x=base_agg[date_col_t].iloc[-1],
                        line_dash="dash", line_color=C_MUTED, line_width=1,
                        annotation_text="Forecast Start", annotation_font_color=C_MUTED,
                    )

                    forecast_colors = [C_CYAN, C_ORANGE, C_GREEN, C_PURPLE]
                    metrics_rows = []

                    for m_idx, pred_model in enumerate(selected_models):
                        fclr = forecast_colors[m_idx % len(forecast_colors)]
                        try:
                            if pred_model == "Linear Regression" and SKLEARN_OK:
                                from sklearn.linear_model import LinearRegression
                                from sklearn.metrics import mean_absolute_error, r2_score
                                lr = LinearRegression()
                                lr.fit(hist_x.reshape(-1,1), hist_y)
                                pred_y  = lr.predict(future_x.reshape(-1,1))
                                fitted  = lr.predict(hist_x.reshape(-1,1))
                                std_err = np.std(hist_y - fitted)
                                ci_upper, ci_lower = pred_y + 1.96*std_err, pred_y - 1.96*std_err
                                mae_out  = mean_absolute_error(hist_y, fitted)
                                rmse_out = np.sqrt(np.mean((hist_y - fitted)**2))
                                r2_out   = r2_score(hist_y, fitted)
                                model_name = "Linear Regression"

                            elif pred_model == "ARIMA" and STATSMODELS_OK:
                                from statsmodels.tsa.arima.model import ARIMA
                                from sklearn.metrics import mean_absolute_error, r2_score
                                arima_fit = ARIMA(hist_y, order=(2,1,2)).fit()
                                fc_arima  = arima_fit.get_forecast(steps=horizon_days)
                                pred_y    = fc_arima.predicted_mean
                                ci_bounds = fc_arima.conf_int(alpha=0.05)
                                ci_lower, ci_upper = ci_bounds.iloc[:,0].values, ci_bounds.iloc[:,1].values
                                fitted    = arima_fit.fittedvalues
                                mae_out   = mean_absolute_error(hist_y[1:], fitted[1:])
                                rmse_out  = np.sqrt(np.mean((hist_y[1:] - fitted[1:])**2))
                                r2_out    = r2_score(hist_y[1:], fitted[1:])
                                model_name = "ARIMA(2,1,2)"

                            elif pred_model == "Exponential Smoothing" and STATSMODELS_OK:
                                from sklearn.metrics import mean_absolute_error, r2_score
                                hw_fit   = ExponentialSmoothing(hist_y, trend="add", seasonal=None).fit()
                                pred_y   = hw_fit.forecast(horizon_days)
                                std_err  = np.std(hist_y - hw_fit.fittedvalues)
                                ci_upper, ci_lower = pred_y + 1.96*std_err, pred_y - 1.96*std_err
                                mae_out  = mean_absolute_error(hist_y, hw_fit.fittedvalues)
                                rmse_out = np.sqrt(np.mean((hist_y - hw_fit.fittedvalues)**2))
                                r2_out   = r2_score(hist_y, hw_fit.fittedvalues)
                                model_name = "Exponential Smoothing"

                            elif pred_model == "Simple Moving Average":
                                window   = min(30, len(hist_y)//2)
                                last_ma  = np.mean(hist_y[-window:])
                                pred_y   = np.full(horizon_days, last_ma)
                                std_err  = np.std(hist_y[-window:])
                                ci_upper, ci_lower = pred_y + 1.96*std_err, pred_y - 1.96*std_err
                                fitted_sma = np.convolve(hist_y, np.ones(window)/window, mode="same")
                                if SKLEARN_OK:
                                    from sklearn.metrics import mean_absolute_error, r2_score
                                    mae_out  = mean_absolute_error(hist_y[window:], fitted_sma[window:])
                                    rmse_out = np.sqrt(np.mean((hist_y[window:] - fitted_sma[window:])**2))
                                    r2_out   = 0.0
                                else:
                                    mae_out, rmse_out, r2_out = std_err, std_err, 0.0
                                model_name = f"SMA ({window}-period)"

                            elif pred_model == "Prophet" and PROPHET_OK:
                                from sklearn.metrics import mean_absolute_error, r2_score
                                p_df = base_agg.rename(columns={date_col_t:"ds", metric_col:"y"})
                                p_df["ds"] = p_df["ds"].dt.tz_localize(None)
                                m  = Prophet(interval_width=0.95)
                                m.fit(p_df)
                                future_p = m.make_future_dataframe(periods=horizon_days)
                                fc = m.predict(future_p)
                                pred_y   = fc["yhat"].tail(horizon_days).values
                                ci_upper = fc["yhat_upper"].tail(horizon_days).values
                                ci_lower = fc["yhat_lower"].tail(horizon_days).values
                                mae_out  = mean_absolute_error(p_df["y"], fc["yhat"][:len(p_df)])
                                rmse_out = np.sqrt(np.mean((p_df["y"].values - fc["yhat"][:len(p_df)].values)**2))
                                r2_out   = r2_score(p_df["y"], fc["yhat"][:len(p_df)])
                                model_name = "Prophet"

                            else:
                                continue

                            if show_ci:
                                r, g, b = int(fclr[1:3],16), int(fclr[3:5],16), int(fclr[5:7],16)
                                fig_pred.add_trace(go.Scatter(
                                    x=list(future_dates)+list(future_dates[::-1]),
                                    y=list(ci_upper)+list(ci_lower[::-1]),
                                    fill="toself",
                                    fillcolor=f"rgba({r},{g},{b},0.10)",
                                    line=dict(color="rgba(0,0,0,0)"),
                                    name=f"{model_name} 95% CI",
                                ))

                            fig_pred.add_trace(go.Scatter(
                                x=future_dates, y=pred_y,
                                mode="lines", name=f"Forecast: {model_name}",
                                line=dict(color=fclr, width=2, dash="dot"),
                            ))
                            metrics_rows.append({
                                "Model": model_name, "MAE": f"{mae_out:,.2f}",
                                "RMSE": f"{rmse_out:,.2f}", "R²": f"{r2_out:.4f}",
                                "Horizon": f"{horizon_days} days",
                            })

                        except Exception as model_err:
                            st.warning(f"{pred_model} failed: {model_err}")

                    panel_header(f"Forecast — {', '.join(selected_models)} — Next {horizon_days} Days")
                    st.plotly_chart(style_fig(fig_pred, 460), use_container_width=True)

                    if metrics_rows:
                        section_label("Model Performance Metrics", C_CYAN)
                        st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Forecast error: {e}")


# ══════════════════════════════════════════════════════════════
# TAB 6 — RAW DATA  (ENHANCED)
# ══════════════════════════════════════════════════════════════

with tab_raw:
    tab_title("Raw Data Explorer & Editor")

    # ── Session state for undo/redo ──
    if "raw_undo_stack" not in st.session_state:
        st.session_state.raw_undo_stack = []
    if "raw_redo_stack" not in st.session_state:
        st.session_state.raw_redo_stack = []
    if "raw_col_order" not in st.session_state:
        st.session_state.raw_col_order = df.columns.tolist()
    if "raw_highlighted" not in st.session_state:
        st.session_state.raw_highlighted = set()

    # Sync column order with current df if columns changed
    existing_order = [c for c in st.session_state.raw_col_order if c in df.columns]
    new_cols = [c for c in df.columns if c not in existing_order]
    st.session_state.raw_col_order = existing_order + new_cols

    # ── Controls Row 1: Search + Column selector + Rows per page ──
    raw_c1, raw_c2, raw_c3 = st.columns([2, 2, 1])
    with raw_c1:
        search_raw = st.text_input("🔍 Search", placeholder="Search any value…", label_visibility="collapsed")
    with raw_c2:
        show_cols_raw = st.multiselect(
            "Show Columns", st.session_state.raw_col_order,
            default=st.session_state.raw_col_order,
            label_visibility="collapsed", key="raw_cols"
        )
    with raw_c3:
        rows_per_page = st.selectbox("Rows", [10, 25, 50, 100, 500], index=1, label_visibility="collapsed")

    # ── Column Reorder ──
    with st.expander("↕ Reorder Columns", expanded=False):
        st.caption("Drag columns to reorder — then click Apply")
        reorder_cols = st.multiselect(
            "Column order (select in desired order)",
            df.columns.tolist(),
            default=st.session_state.raw_col_order,
            key="raw_reorder"
        )
        if st.button("✅ Apply Order", key="apply_col_order"):
            st.session_state.raw_col_order = reorder_cols
            st.rerun()

    # ── Sort Controls ──
    sort_c1, sort_c2 = st.columns([2, 1])
    with sort_c1:
        sort_col = st.selectbox("Sort by", ["— None —"] + df.columns.tolist(), label_visibility="collapsed", key="raw_sort")
    with sort_c2:
        sort_asc = st.radio("Order", ["Ascending","Descending"], horizontal=True, label_visibility="collapsed")

    # ── Build display df ──
    display_cols = show_cols_raw if show_cols_raw else st.session_state.raw_col_order
    display_df = df[display_cols].copy()

    if search_raw:
        mask = display_df.astype(str).apply(
            lambda col: col.str.contains(search_raw, case=False, na=False)
        ).any(axis=1)
        display_df = display_df[mask]

    if sort_col != "— None —" and sort_col in display_df.columns:
        display_df = display_df.sort_values(sort_col, ascending=(sort_asc == "Ascending"))

    # ── Pagination ──
    total_rows_raw = len(display_df)
    total_pages = max(1, (total_rows_raw - 1) // rows_per_page + 1)
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, key="raw_page")
    start_r = (page - 1) * rows_per_page
    end_r   = start_r + rows_per_page
    paged_df = display_df.iloc[start_r:end_r].copy()

    st.caption(
        f"Showing rows {start_r+1}–{min(end_r, total_rows_raw)} of {total_rows_raw:,}  ·  "
        f"Page {page}/{total_pages}  ·  {len(display_cols)} columns visible"
    )

    panel_header("Editable Data Table — click any cell to edit · add/delete rows below")

    # ── Editable data table ──
    edited = st.data_editor(
        paged_df,
        use_container_width=True,
        height=420,
        num_rows="dynamic",
        key="raw_editor",
        column_config={
            col: st.column_config.Column(disabled=False)
            for col in paged_df.columns
        }
    )

    # Detect edited cells for highlighting indicator
    if not edited.equals(paged_df):
        changed_mask = ~(edited == paged_df)
        changed_cells = changed_mask.sum().sum()
        if changed_cells > 0:
            st.markdown(
                f"<div style='padding:4px 10px;margin:4px 0;background:rgba(212,116,10,0.1);"
                f"border-left:2px solid {C_ORANGE};border-radius:0 5px 5px 0;font-size:0.78rem;color:{C_ORANGE};'>"
                f"⚡ {changed_cells} cell(s) edited — click <b>Save Edits</b> to persist changes</div>",
                unsafe_allow_html=True
            )

            # Validate data types
            validation_errors = []
            for col in edited.columns:
                orig_dtype = paged_df[col].dtype
                try:
                    edited[col].astype(orig_dtype)
                except Exception:
                    validation_errors.append(f"Column <b>{col}</b> expects {orig_dtype} — type mismatch detected")
            if validation_errors:
                for err in validation_errors:
                    st.markdown(
                        f"<div style='padding:4px 10px;margin:3px 0;background:rgba(239,68,68,0.1);"
                        f"border-left:2px solid {C_RED};border-radius:0 5px 5px 0;font-size:0.75rem;color:{C_RED};'>"
                        f"⚠️ {err}</div>",
                        unsafe_allow_html=True
                    )

    # ── Action Buttons ──
    raw_act_c1, raw_act_c2, raw_act_c3, raw_act_c4, raw_act_c5 = st.columns(5)

    with raw_act_c1:
        if st.button("💾 Save Edits", key="save_edits"):
            # Push to undo stack before saving
            if st.session_state.df_edited is not None:
                st.session_state.raw_undo_stack.append(st.session_state.df_edited.copy())
            else:
                st.session_state.raw_undo_stack.append(df_raw.copy())
            st.session_state.raw_redo_stack.clear()

            if st.session_state.df_edited is None:
                st.session_state.df_edited = df_raw.copy()
            # Replace page slice with edited values
            idx_slice = display_df.index[start_r:end_r]
            for col in edited.columns:
                if col in st.session_state.df_edited.columns:
                    st.session_state.df_edited.loc[idx_slice, col] = edited[col].values[:len(idx_slice)]
            st.success(f"✅ Edits saved! ({len(idx_slice)} rows updated)")

    with raw_act_c2:
        undo_disabled = len(st.session_state.raw_undo_stack) == 0
        if st.button(f"↩ Undo ({len(st.session_state.raw_undo_stack)})", key="undo_edits", disabled=undo_disabled):
            if st.session_state.df_edited is not None:
                st.session_state.raw_redo_stack.append(st.session_state.df_edited.copy())
            prev = st.session_state.raw_undo_stack.pop()
            st.session_state.df_edited = prev
            st.success("↩ Undone!")
            st.rerun()

    with raw_act_c3:
        redo_disabled = len(st.session_state.raw_redo_stack) == 0
        if st.button(f"↪ Redo ({len(st.session_state.raw_redo_stack)})", key="redo_edits", disabled=redo_disabled):
            nxt = st.session_state.raw_redo_stack.pop()
            if st.session_state.df_edited is not None:
                st.session_state.raw_undo_stack.append(st.session_state.df_edited.copy())
            st.session_state.df_edited = nxt
            st.success("↪ Redone!")
            st.rerun()

    with raw_act_c4:
        if st.button("↺ Reset All", key="reset_edits"):
            st.session_state.df_edited = None
            st.session_state.raw_undo_stack.clear()
            st.session_state.raw_redo_stack.clear()
            st.success("Reverted to original data.")
            st.rerun()

    with raw_act_c5:
        pass  # Spacer

    # ── Export Options ──
    section_label("📥 Export Data", C_CYAN)
    exp_c1, exp_c2, exp_c3 = st.columns(3)

    with exp_c1:
        csv_out = display_df.to_csv(index=False).encode()
        st.download_button(
            "📥 Download CSV", data=csv_out,
            file_name="data_export.csv", mime="text/csv",
            use_container_width=True
        )

    with exp_c2:
        try:
            import io as _io
            buf = _io.BytesIO()
            display_df.to_excel(buf, index=False)
            st.download_button(
                "📥 Download Excel", data=buf.getvalue(),
                file_name="data_export.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        except Exception:
            st.caption("Excel export unavailable (openpyxl not installed)")

    with exp_c3:
        json_out = display_df.to_json(orient="records", date_format="iso").encode()
        st.download_button(
            "📥 Download JSON", data=json_out,
            file_name="data_export.json", mime="application/json",
            use_container_width=True
        )

    # ── Column Summary ──
    section_label("Column Visibility Summary", C_MUTED)
    st.markdown(
        " ".join([
            f'<span style="display:inline-block;margin:2px;padding:2px 8px;'
            f'background:rgba(26,109,212,0.15);border:1px solid rgba(26,109,212,0.3);'
            f'border-radius:3px;font-size:0.7rem;color:{C_TEXT};">{c}</span>'
            for c in display_df.columns
        ]),
        unsafe_allow_html=True
    )

    # ── Quick Stats for visible numeric columns ──
    vis_nc = [c for c in display_cols if c in nc]
    if vis_nc:
        section_label("Quick Stats — Visible Numeric Columns", C_BLUE)
        stats_df = display_df[vis_nc].describe().T.round(3)
        stats_df.insert(0, "Column", stats_df.index)
        stats_df = stats_df.reset_index(drop=True)
        st.dataframe(stats_df, use_container_width=True, height=200)

# ============================================================
# CHATBOT — Sidebar panel
# ============================================================

if st.sidebar.button("💬 Open AI Chatbot", key="open_chat"):
    st.session_state.chat_open = not st.session_state.chat_open

if st.session_state.chat_open:
    with st.sidebar:
        st.markdown("---")
        sidebar_label("💬 AI Chat Assistant")

        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_msgs:
                role_color = C_BLUE if msg["role"] == "user" else C_CYAN
                align = "flex-end" if msg["role"] == "user" else "flex-start"
                bg = "rgba(26,109,212,0.15)" if msg["role"] == "user" else "rgba(0,220,200,0.08)"
                border = "rgba(26,109,212,0.4)" if msg["role"] == "user" else "rgba(0,220,200,0.2)"
                st.markdown(
                    f"""<div style="display:flex;justify-content:{align};margin:4px 0;">
                        <div style="background:{bg};border:1px solid {border};
                            border-radius:10px;padding:7px 10px;max-width:90%;
                            font-size:0.78rem;color:{C_TEXT};">
                            <b style="color:{role_color};font-size:0.65rem;">
                                {'You' if msg['role']=='user' else '🤖 Assistant'}
                            </b><br>{msg['content']}
                        </div></div>""",
                    unsafe_allow_html=True
                )

        user_input = st.text_input("Ask about your data...", key="chat_input", label_visibility="collapsed")

        if st.button("Send", key="chat_send") and user_input.strip():
            st.session_state.chat_msgs.append({"role": "user", "content": user_input})
            msg_l = user_input.lower()
            if any(w in msg_l for w in ["rows","row","size","how many"]):
                reply = f"Your dataset has **{df.shape[0]:,} rows** and **{df.shape[1]} columns**."
            elif any(w in msg_l for w in ["column","columns","field","fields"]):
                reply = f"Columns: {', '.join(df.columns.tolist()[:10])}{'…' if len(df.columns) > 10 else ''}."
            elif any(w in msg_l for w in ["missing","null","nan","empty"]):
                reply = f"There are **{df.isnull().sum().sum():,} missing cells** ({df.isnull().sum().sum()/df.size*100:.1f}% of total)."
            elif any(w in msg_l for w in ["numeric","number","int","float"]):
                reply = f"Numeric columns ({len(nc)}): {', '.join(nc[:8])}."
            elif any(w in msg_l for w in ["category","categorical","text","string"]):
                reply = f"Categorical columns ({len(cc)}): {', '.join(cc[:8])}."
            elif any(w in msg_l for w in ["mean","average","avg"]):
                if nc:
                    means = {c: round(df[c].mean(), 2) for c in nc[:4]}
                    reply = "Means: " + ", ".join([f"{k}={v}" for k, v in means.items()])
                else:
                    reply = "No numeric columns found."
            elif any(w in msg_l for w in ["max","maximum","highest","top"]):
                if nc:
                    maxs = {c: df[c].max() for c in nc[:4]}
                    reply = "Max values: " + ", ".join([f"{k}={v}" for k, v in maxs.items()])
                else:
                    reply = "No numeric columns found."
            elif any(w in msg_l for w in ["min","minimum","lowest","bottom"]):
                if nc:
                    mins = {c: df[c].min() for c in nc[:4]}
                    reply = "Min values: " + ", ".join([f"{k}={v}" for k, v in mins.items()])
                else:
                    reply = "No numeric columns found."
            elif any(w in msg_l for w in ["duplicate","duplicates"]):
                reply = f"There are **{df.duplicated().sum()}** duplicate rows."
            elif any(w in msg_l for w in ["date","time","datetime"]):
                reply = f"Datetime columns: {', '.join(dc) if dc else 'None found'}."
            elif any(w in msg_l for w in ["predict","forecast","future","arima","prophet"]):
                reply = "Go to the **Trends** tab → Future Prediction. Available models: Linear Regression, ARIMA, Exponential Smoothing, Prophet, SMA. You can run multi-model comparison!"
            elif any(w in msg_l for w in ["correlat"]):
                reply = "Go to the **Correlation** tab to analyse relationships using Pearson, Spearman, or Kendall methods."
            elif any(w in msg_l for w in ["trend","seasonality","rising","falling"]):
                reply = "The **Trends** tab shows trend direction (Rising/Falling/Stable), moving averages, rolling std dev, and seasonality detection."
            elif any(w in msg_l for w in ["edit","undo","redo","save"]):
                reply = "The **Raw Data** tab supports live cell editing, Undo/Redo history, data type validation, and export to CSV/Excel/JSON."
            elif any(w in msg_l for w in ["hello","hi","hey"]):
                reply = f"Hello! 👋 Dataset: {df.shape[0]:,} rows × {df.shape[1]} cols. What would you like to explore?"
            else:
                reply = (
                    f"I can help explore your dataset ({df.shape[0]:,} rows, {df.shape[1]} cols). "
                    f"Try asking about: rows/columns, missing values, numeric/categorical, mean/max/min, "
                    f"duplicates, trends, predictions (ARIMA/Prophet), or correlation."
                )

            st.session_state.chat_msgs.append({"role": "assistant", "content": reply})
            st.rerun()

        if st.button("🗑 Clear Chat", key="clear_chat"):
            st.session_state.chat_msgs = []
            st.rerun()

        if st.button("✕ Close Chat", key="close_chat"):
            st.session_state.chat_open = False
            st.rerun()
