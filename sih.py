"""
=============================================================================
 MPLADS Transparency Dashboard + AI Risk Intelligence Layer — Streamlit
=============================================================================
Laid out like the public "Empowered Indian" MPLADS dashboard
(empoweredindian.in/mplads) — a national overview, a browsable list of
states, and per-MP profile pages.

This version is FULLY SELF-CONTAINED and PUBLICLY DEPLOYABLE:
  • No external API calls, no network connection required at runtime.
  • No API keys / secrets of any kind (a previous draft of this file
    embedded a live third-party API call and a hardcoded bearer token —
    both have been removed. Never ship a real key inside client-visible
    source code; if you later add a private backend, read the key from
    an environment variable / st.secrets, never hardcode it).
  • All data comes from the public MPLADS export CSVs that were
    downloaded from the official MPLADS/eSAKSHI portal
    (https://mplads.mospi.gov.in/digigov/dashboard.html) and pre-scored
    by the companion Colab notebook into ./data/*.csv + meta.json.

On top of the plain browsing experience (Home / Browse by State / MP
Profiles) this adds an 🤖 AI RISK INTELLIGENCE LAYER — a priority queue
of works and MPs worth a closer look, with plain-English reasons and a
verification checklist. That layer is clearly marked throughout as an
addition on top of the official data, not a replacement for it.

Run locally with:
    pip install -r requirements.txt
    streamlit run app.py

Folder layout expected:
    streamlit_app/
    ├── app.py
    ├── risk_engine.py
    └── data/
        ├── mp_risk_scores.csv
        ├── work_risk_scores.csv
        ├── vendor_features.csv
        └── meta.json
=============================================================================
"""

# -----------------------------------------------------------------------
# STEP 1 — Imports & page config  (NOTE: no `requests` import anywhere —
# this app never makes an outbound network call)
# -----------------------------------------------------------------------
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="MPLADS Transparency + AI Risk Intelligence",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent / "data"
RISK_COLORS = {"High": "#f87171", "Medium": "#fbbf24", "Low": "#34d399"}

# -----------------------------------------------------------------------
# STEP 2 — Dark theme styling
# -----------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root{
        --bg:#0b0f14; --bg-panel:#11161d; --bg-card:#161c25; --border:#232b36;
        --text:#e6e9ef; --text-dim:#93a0b4; --accent:#5eead4; --accent2:#60a5fa;
    }
    .stApp{ background: radial-gradient(1200px 800px at 10% -10%, #101823 0%, #0b0f14 55%) fixed; color:var(--text); }
    section[data-testid="stSidebar"]{ background:#0d1218; border-right:1px solid var(--border); }
    section[data-testid="stSidebar"] * { color: var(--text) !important; }
    h1,h2,h3,h4 { color:var(--text) !important; font-weight:700 !important; }
    p, span, label, div { color:var(--text); }

    .hero {
        background: linear-gradient(135deg, #0f2e57 0%, #143a73 60%, #0b3d91 100%);
        border: 1px solid var(--border);
        padding: 2rem 2rem; border-radius: 14px; color: white; margin-bottom: 1.2rem;
        box-shadow: 0 8px 30px rgba(0,0,0,0.45);
    }
    .hero h1 { margin: 0; font-size: 2rem; color:white !important; }
    .hero p { margin: .5rem 0 0 0; opacity: .85; color:#dbe6ff; }

    .state-card {
        border: 1px solid var(--border); border-radius: 12px; padding: .9rem 1rem;
        background: linear-gradient(180deg,var(--bg-card) 0%, #10151c 100%);
        margin-bottom: .6rem; box-shadow:0 4px 16px rgba(0,0,0,0.3);
    }
    .ai-banner {
        background: rgba(96,165,250,0.08); border: 1px solid rgba(96,165,250,0.35);
        border-radius: 12px; padding: 1rem 1.2rem; margin: 1rem 0; color:var(--text-dim);
    }
    .kpi-card{
        background:linear-gradient(180deg,var(--bg-card) 0%, #10151c 100%);
        border:1px solid var(--border); border-radius:14px; padding:16px 18px;
        box-shadow:0 4px 24px rgba(0,0,0,0.35);
    }
    .kpi-label{ color:var(--text-dim); font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em;}
    .kpi-value{ font-size:1.7rem; font-weight:800; color:var(--text); margin-top:4px;}

    .pill{ display:inline-block; padding:3px 12px; border-radius:999px; font-weight:700; font-size:0.72rem;}
    .pill-high{ background:rgba(248,113,113,0.15); color:#f87171; border:1px solid rgba(248,113,113,0.4);}
    .pill-med{ background:rgba(251,191,36,0.15); color:#fbbf24; border:1px solid rgba(251,191,36,0.4);}
    .pill-low{ background:rgba(52,211,153,0.15); color:#34d399; border:1px solid rgba(52,211,153,0.4);}

    .stDataFrame{ border:1px solid var(--border); border-radius:10px; }
    hr{ border-color:var(--border) !important; }
    [data-testid="stMetricValue"]{ color:var(--text); }
    </style>
    """,
    unsafe_allow_html=True,
)


def pill(level: str) -> str:
    cls = {"High": "pill-high", "Medium": "pill-med", "Low": "pill-low"}.get(level, "pill-low")
    return f'<span class="pill {cls}">{level.upper()} RISK</span>'


# -----------------------------------------------------------------------
# STEP 3 — Load the local public dataset (pre-scored by the Colab notebook)
# No network calls of any kind: this is what makes the app safe to deploy
# publicly without any backend, API key, or live connection to maintain.
# -----------------------------------------------------------------------
@st.cache_data(show_spinner="Loading public MPLADS dataset...")
def load_dataset():
    mp_df = pd.read_csv(DATA_DIR / "mp_risk_scores.csv")
    work_df = pd.read_csv(DATA_DIR / "work_risk_scores.csv")
    vendor_df = pd.read_csv(DATA_DIR / "vendor_features.csv")
    meta = json.load(open(DATA_DIR / "meta.json"))

    # Split the pipe-joined reason/checklist strings back into lists for bullet display
    mp_df["reasons_list"] = mp_df["reasons_text"].apply(
        lambda s: [] if not isinstance(s, str) or s.startswith("No significant") else s.split(" | "))
    mp_df["checklist_list"] = mp_df["checklist_text"].apply(
        lambda s: [] if not isinstance(s, str) or s.startswith("General random") else s.split(" | "))
    work_df["reasons_list"] = work_df["reasons_text"].apply(
        lambda s: [] if not isinstance(s, str) or s.startswith("No red flags") else s.split(" | "))
    return mp_df, work_df, vendor_df, meta


mp_df, work_df, vendor_df, meta = load_dataset()

# -----------------------------------------------------------------------
# STEP 4 — Aggregates derived on the fly from the local dataset
# (replaces the separate national_overview / aggregate_by_state /
# aggregate_by_worktype JSON files — one less thing to keep in sync)
# -----------------------------------------------------------------------
state_agg = (
    work_df.groupby("State")["risk_level"].apply(lambda s: (s == "High").sum())
    .rename("high_risk").reset_index().sort_values("high_risk", ascending=False)
)
worktype_agg = (
    work_df.groupby("Category")["risk_level"].apply(lambda s: (s == "High").sum())
    .rename("high_risk").reset_index().sort_values("high_risk", ascending=False)
)

if "nav" not in st.session_state:
    st.session_state.nav = "🏠 Home"
if "selected_state" not in st.session_state:
    st.session_state.selected_state = None
if "selected_mp" not in st.session_state:
    st.session_state.selected_mp = None

# -----------------------------------------------------------------------
# STEP 5 — Sidebar navigation
# -----------------------------------------------------------------------
st.sidebar.markdown("## 🏛️ MPLADS Dashboard")
st.sidebar.caption("· Public dataset ·")
st.sidebar.markdown("---")
nav_options = ["🏠 Home", "🗺️ Browse by State", "🏛️ MP Profiles", "🤖 AI Risk Intelligence", "📊 Analytics"]
st.session_state.nav = st.sidebar.radio("Navigate", nav_options, index=nav_options.index(st.session_state.nav))
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div class="ai-banner">⚠️ AI risk flags indicate <b>patterns that deserve verification</b>, '
    'not proof of wrongdoing.</div>', unsafe_allow_html=True,
)

# -----------------------------------------------------------------------
# STEP 6 — Hero header
# -----------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🏛️ MPLADS Transparency Dashboard</h1>
        <p>Explore how MP Local Area Development funds are allocated, spent, and completed —
        with an added AI layer that tells officials which works deserve a closer look, and why.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("Data source: **public MPLADS Dashboard export**.")

# =========================================================================
# PAGE: HOME
# =========================================================================
if st.session_state.nav == "🏠 Home":
    total_allocated = mp_df["Allocated Amount (₹)"].sum()
    total_expenditure = mp_df["Total Expenditure (₹)"].sum()
    utilization = mp_df["Utilization %"].mean()
    completion_rate = mp_df["Completion Rate %"].mean()
    total_mps = len(mp_df)

    st.subheader("National Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, value in [
        (c1, "Total Allocated", f"₹{total_allocated/1e7:,.0f} Cr"),
        (c2, "Total Expenditure", f"₹{total_expenditure/1e7:,.0f} Cr"),
        (c3, "Avg. Utilisation", f"{utilization:.1f}%"),
        (c4, "Avg. Completion Rate", f"{completion_rate:.1f}%"),
        (c5, "MPs Tracked", f"{total_mps:,}"),
    ]:
        with col:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
                        f'<div class="kpi-value">{value}</div></div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="ai-banner">
        🤖 <strong>Built into this dashboard:</strong> an AI Risk Intelligence layer flags works and
        MPs whose spending patterns are statistically unusual and worth verifying first —
        {meta.get('high_risk_mps', 0)} MPs and {meta.get('high_risk_works', 0):,} individual works are
        currently flagged High risk. See the <strong>"🤖 AI Risk Intelligence"</strong> tab in the sidebar.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Browse by State")
    state_summary = (
        mp_df.groupby("State")
        .agg(mps=("MP Name", "count"), allocated=("Allocated Amount (₹)", "sum"))
        .reset_index().sort_values("allocated", ascending=False).head(12)
    )
    cols = st.columns(4)
    for i, row in enumerate(state_summary.itertuples()):
        with cols[i % 4]:
            st.markdown(
                f'<div class="state-card"><strong>{row.State}</strong><br>'
                f'<span style="color:var(--text-dim)">{row.mps} MP(s) · ₹{row.allocated/1e7:,.0f} Cr allocated</span></div>',
                unsafe_allow_html=True,
            )
            if st.button("View →", key=f"state_btn_{row.State}"):
                st.session_state.selected_state = row.State
                st.session_state.nav = "🗺️ Browse by State"
                st.rerun()

# =========================================================================
# PAGE: BROWSE BY STATE
# =========================================================================
elif st.session_state.nav == "🗺️ Browse by State":
    st.subheader("Browse by State")

    states = sorted(mp_df["State"].dropna().unique())
    default_idx = states.index(st.session_state.selected_state) if st.session_state.selected_state in states else 0
    picked_state = st.selectbox("Select a state", states, index=default_idx)
    st.session_state.selected_state = picked_state

    state_df = mp_df[mp_df["State"] == picked_state]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MPs in this state", f"{len(state_df):,}")
    c2.metric("Total Allocated", f"₹{state_df['Allocated Amount (₹)'].sum()/1e7:,.1f} Cr")
    c3.metric("Avg. Completion Rate", f"{state_df['Completion Rate %'].mean():.1f}%")
    c4.metric("High-risk MPs here", f"{(state_df['risk_level'] == 'High').sum():,}")

    display_cols = ["MP Name", "Constituency", "House", "Allocated Amount (₹)",
                     "Total Expenditure (₹)", "Utilization %", "Completion Rate %", "risk_score", "risk_level"]
    st.dataframe(
        state_df[display_cols].sort_values("Allocated Amount (₹)", ascending=False),
        use_container_width=True, height=380,
        column_config={"risk_score": st.column_config.ProgressColumn("AI Risk", min_value=0, max_value=100, format="%.0f")},
    )

    mp_names = state_df["MP Name"].tolist()
    if mp_names:
        st.markdown("##### Jump to an MP's profile")
        picked_mp = st.selectbox("Select an MP", mp_names, key="state_page_mp_pick")
        if st.button("View full profile →"):
            st.session_state.selected_mp = picked_mp
            st.session_state.nav = "🏛️ MP Profiles"
            st.rerun()

# =========================================================================
# PAGE: MP PROFILES
# =========================================================================
elif st.session_state.nav == "🏛️ MP Profiles":
    st.subheader("MP Profiles")

    all_names = sorted(mp_df["MP Name"].dropna().unique())
    query = st.text_input("Search for an MP by name")
    matches = [n for n in all_names if query.lower() in n.lower()] if query else all_names

    if matches:
        default_idx = matches.index(st.session_state.selected_mp) if st.session_state.selected_mp in matches else 0
        picked = st.selectbox("Select an MP", matches, index=default_idx)
        st.session_state.selected_mp = picked
        row = mp_df[mp_df["MP Name"] == picked].iloc[0]

        st.markdown(f"### {row['MP Name']} &nbsp; {pill(row['risk_level'])}", unsafe_allow_html=True)
        st.caption(f"{row.get('Constituency', '')} · {row.get('State', '')} · {row.get('House', '')}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Allocated", f"₹{row['Allocated Amount (₹)']/1e7:,.2f} Cr")
        c2.metric("Expenditure", f"₹{row['Total Expenditure (₹)']/1e7:,.2f} Cr")
        c3.metric("Utilisation", f"{row['Utilization %']:.1f}%")
        c4.metric("Completion", f"{row['Completion Rate %']:.1f}%")

        # AI risk assessment inline on the MP profile
        st.markdown('<div class="ai-banner">', unsafe_allow_html=True)
        st.markdown(f"🤖 **AI Risk Assessment** — score **{row['risk_score']:.0f}/100** ({row['risk_level']} risk)")
        if row["reasons_list"]:
            st.markdown("**Why flagged:**")
            for reason in row["reasons_list"]:
                st.markdown(f"- {reason}")
            st.markdown("**What to verify first:**")
            for item in row["checklist_list"]:
                st.markdown(f"- [ ] {item}")
        else:
            st.markdown("No significant risk patterns detected for this MP in the current dataset.")
        st.markdown("</div>", unsafe_allow_html=True)

        mp_works = work_df[work_df["MP Name"] == picked]
        if not mp_works.empty:
            st.markdown("##### Flagged completed works for this MP")
            st.dataframe(
                mp_works[["Work Description", "Category", "Final Amount (₹)", "risk_score", "risk_level"]]
                .sort_values("risk_score", ascending=False).head(20),
                use_container_width=True,
                column_config={"risk_score": st.column_config.ProgressColumn("AI Risk", min_value=0, max_value=100, format="%.0f")},
            )
    else:
        st.info("No MPs match that search.")

# =========================================================================
# PAGE: 🤖 AI RISK INTELLIGENCE
# =========================================================================
elif st.session_state.nav == "🤖 AI Risk Intelligence":
    st.markdown(
        """
        <div class="ai-banner">
        🤖 <strong>This section is our addition</strong> — it does not exist on the standard
        MPLADS transparency dashboard. It answers three questions officials can't get answered
        elsewhere: which works to look at first, why they look unusual, and what to verify.
        A flagged item is <strong>not</strong> a finding of fraud — it's a statistical anomaly
        worth a human look.
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Works scored", f"{meta.get('n_works', len(work_df)):,}")
    m2.metric("🔴 High-risk works", f"{meta.get('high_risk_works', 0):,}")
    m3.metric("🟠 Medium-risk works", f"{int((work_df['risk_level']=='Medium').sum()):,}")
    m4.metric("🔴 High-risk MPs", f"{meta.get('high_risk_mps', 0):,}")

    st.sidebar.header("AI Risk Filters")
    states = sorted(work_df["State"].dropna().unique())
    worktypes = sorted(work_df["Category"].dropna().unique())
    selected_states = st.sidebar.multiselect("State", states, default=[])
    selected_worktypes = st.sidebar.multiselect("Work category", worktypes, default=[])
    selected_bands = st.sidebar.multiselect("Risk band", ["High", "Medium", "Low"], default=["High", "Medium"])
    min_amount = st.sidebar.number_input("Minimum work amount (₹)", value=0, step=50_000)

    def apply_filters(df, band_col, amount_col=None):
        out = df.copy()
        if selected_states:
            out = out[out["State"].isin(selected_states)]
        if selected_worktypes and "Category" in out.columns:
            out = out[out["Category"].isin(selected_worktypes)]
        if selected_bands:
            out = out[out[band_col].isin(selected_bands)]
        if min_amount and amount_col and amount_col in out.columns:
            out = out[out[amount_col] >= min_amount]
        return out

    sub_queue, sub_mp = st.tabs(["🎯 Priority Queue (Works)", "🏛️ MP / Fund Risk"])

    with sub_queue:
        filtered_works = apply_filters(work_df, "risk_level", "Final Amount (₹)")
        st.caption(f"Showing {len(filtered_works):,} of {len(work_df):,} scored works "
                   "(narrow with the sidebar filters)")
        if filtered_works.empty:
            st.warning("No works match the current filters.")
        else:
            top = filtered_works.sort_values("risk_score", ascending=False).head(200)
            for _, wrow in top.iterrows():
                band = wrow.get("risk_level", "Low")
                dot = "red" if band == "High" else "orange" if band == "Medium" else "green"
                with st.expander(
                    f":{dot}[●] **{wrow['risk_score']:.0f}/100** — "
                    f"{str(wrow['Work Description'])[:90]}  ·  {wrow['MP Name']}  ·  {wrow['State']}"
                ):
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        st.markdown("**Why flagged:**")
                        for reason in wrow.get("reasons_list", []):
                            st.markdown(f"- {reason}")
                        if not wrow.get("reasons_list"):
                            st.markdown("- No specific rule triggered; flagged by overall pattern deviation.")
                    with c2:
                        st.markdown(f"**Amount:** ₹{wrow['Final Amount (₹)']:,.0f}")
                        st.markdown(f"**Category:** {wrow['Category']}")
                        st.markdown(f"**Constituency:** {wrow['Constituency']}")
                        st.markdown(f"**Work ID:** {wrow['Work ID']}")

    with sub_mp:
        filtered_mps = apply_filters(mp_df, "risk_level")
        mp_view = (filtered_mps if not filtered_mps.empty else mp_df).sort_values("risk_score", ascending=False)
        display_cols = [c for c in [
            "MP Name", "State", "Constituency", "risk_score", "risk_level",
            "Utilization %", "Completion Rate %", "top_vendor", "top_vendor_share_pct",
        ] if c in mp_view.columns]
        st.dataframe(
            mp_view[display_cols].head(100),
            use_container_width=True, height=420,
            column_config={"risk_score": st.column_config.ProgressColumn("AI Risk", min_value=0, max_value=100, format="%.0f")},
        )
        mp_names = mp_view["MP Name"].head(100).tolist()
        if mp_names:
            picked = st.selectbox("Select an MP to see details", mp_names, key="ai_mp_pick")
            mp_row = mp_view[mp_view["MP Name"] == picked].iloc[0]
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown("**Why flagged:**")
                for reason in mp_row.get("reasons_list", []):
                    st.markdown(f"- {reason}")
                st.markdown("**What to verify first:**")
                for item in mp_row.get("checklist_list", []):
                    st.markdown(f"- [ ] {item}")
            with c2:
                st.metric("Risk score", f"{mp_row['risk_score']:.0f}/100")
                st.metric("Risk band", mp_row["risk_level"])

# =========================================================================
# PAGE: ANALYTICS
# =========================================================================
elif st.session_state.nav == "📊 Analytics":
    st.subheader("Where the AI-flagged risk is concentrated")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(state_agg.head(15), x="State", y="high_risk",
                     title="High-risk works by state (top 15)",
                     color_discrete_sequence=[RISK_COLORS["High"]], template="plotly_dark")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.bar(worktype_agg.head(15), x="Category", y="high_risk",
                      title="High-risk works by category (top 15)",
                      color_discrete_sequence=[RISK_COLORS["High"]], template="plotly_dark")
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.histogram(work_df, x="risk_score", nbins=40, color="risk_level",
                         color_discrete_map=RISK_COLORS, template="plotly_dark",
                         title="Distribution of AI risk scores across all scored works")
    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig3, use_container_width=True)

    st.caption("Points far above the diagonal = money reported spent, little to show for it on the ground.")
    fig4 = px.scatter(mp_df, x="Completion Rate %", y="Utilization %",
                      color="risk_level", color_discrete_map=RISK_COLORS,
                      hover_data=["MP Name", "State"], template="plotly_dark",
                      title="Utilisation % vs Completion Rate % by MP")
    fig4.add_shape(type="line", x0=0, y0=0, x1=100, y1=100, line=dict(dash="dash", color="gray"))
    fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig4, use_container_width=True)

# -----------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------
st.divider()
st.caption(
    "Browsing pages modeled on the public Empowered Indian MPLADS dashboard "
    "Reference:'https://mplads.mospi.gov.in/digigov/dashboard.html, "
     "The AI Risk Intelligence tab is scored "
    "by a hybrid rule-based + Isolation Forest model (see the companion Colab notebook)."
)
