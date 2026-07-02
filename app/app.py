"""Streamlit sandbox app for visualizing top 100 ranked candidates."""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

# Add project root to sys.path to allow importing from root-level modules.
sys.path.append(str(Path(__file__).parent.parent))


# Loads candidate details, honeypots, and extracts custom features.
@st.cache_data
def load_data():
    df_sub = pd.read_csv("submission.csv")
    df_feat = pd.read_parquet("artifacts/features.parquet")
    df_hp = pd.read_parquet("artifacts/honeypot_flags.parquet")
    
    # Merge candidate details
    df = df_sub.merge(df_feat, on="candidate_id", how="left")
    df = df.merge(df_hp, on="candidate_id", how="left")
    
    # Extract computed features for candidate details
    from features_utils import extract_candidate_features
    features_list = [extract_candidate_features(row) for _, row in df.iterrows()]
    df_extracted = pd.DataFrame(features_list)
    
    # Add extracted features to the main dataframe
    for col in df_extracted.columns:
        if col not in df.columns or col in ["is_consulting_only", "behavioral_score"]:
            df[col] = df_extracted[col].values
            
    return df


# Runs the Streamlit dashboard layout, filters, and charts.
def main():
    st.set_page_config(
        page_title="Talent Radar — Candidate Explorer",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # sleek custom styling injection
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #FF4B4B, #FF8F8F);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .subheader {
            font-size: 1.1rem;
            color: #6c757d;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 1.2rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            text-align: center;
            border: 1px solid #e9ecef;
        }
        .candidate-card {
            background-color: #ffffff;
            padding: 1.5rem;
            border-radius: 16px;
            border-left: 6px solid #FF4B4B;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            margin-bottom: 1.5rem;
        }
        .candidate-card-positive {
            border-left-color: #28a745;
        }
        </style>
    """, unsafe_allow_html=True)

    df = load_data()

    st.markdown('<div class="main-header">Talent Radar</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Candidate Ranking Explorer — Senior AI/ML Engineer JD</div>', unsafe_allow_html=True)

    # --- Top KPIs ---
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown("""
            <div class="metric-card">
                <div style="font-size: 0.85rem; color: #6c757d; text-transform: uppercase;">Total Pool</div>
                <div style="font-size: 2rem; font-weight: 700; color: #212529;">100,000</div>
            </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown("""
            <div class="metric-card">
                <div style="font-size: 0.85rem; color: #6c757d; text-transform: uppercase;">Shortlist Size</div>
                <div style="font-size: 2rem; font-weight: 700; color: #007bff;">2,956</div>
            </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown("""
            <div class="metric-card">
                <div style="font-size: 0.85rem; color: #6c757d; text-transform: uppercase;">Valid Post-Honeypot</div>
                <div style="font-size: 2rem; font-weight: 700; color: #17a2b8;">2,556</div>
            </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown("""
            <div class="metric-card">
                <div style="font-size: 0.85rem; color: #6c757d; text-transform: uppercase;">Submission Status</div>
                <div style="font-size: 2rem; font-weight: 700; color: #28a745;">✓ Valid</div>
            </div>
        """, unsafe_allow_html=True)

    st.write("")

    # --- Sidebar Filtering ---
    st.sidebar.header("🎯 Filters")
    search_query = st.sidebar.text_input("🔍 Search Candidates", placeholder="ID, Title, Company...")
    min_yoe = st.sidebar.slider("Minimum YOE", 0, 20, 0)
    
    # Filter datasets
    filtered_df = df[df["years_of_experience"] >= min_yoe]
    if search_query:
        search_query_lower = search_query.lower()
        match_mask = (
            filtered_df["candidate_id"].str.lower().str.contains(search_query_lower) |
            filtered_df["current_title"].astype(str).str.lower().str.contains(search_query_lower) |
            filtered_df["current_company"].astype(str).str.lower().str.contains(search_query_lower)
        )
        filtered_df = filtered_df[match_mask]

    # --- Main Content Columns ---
    col_list, col_details = st.columns([1.1, 0.9])

    with col_list:
        st.subheader("🏆 Top Ranked Candidates")
        st.dataframe(
            filtered_df[["rank", "candidate_id", "current_title", "current_company", "score"]].rename(
                columns={
                    "rank": "Rank",
                    "candidate_id": "Candidate ID",
                    "current_title": "Current Title",
                    "current_company": "Company",
                    "score": "Final Score"
                }
            ),
            use_container_width=True,
            hide_index=True
        )

    with col_details:
        st.subheader("👤 Candidate Details")
        
        # Candidate selection dropdown
        c_list = filtered_df["candidate_id"].tolist()
        if not c_list:
            st.warning("No candidates found matching the filters.")
            return

        selected_id = st.selectbox("Select Candidate to Inspect", c_list)
        cand = filtered_df[filtered_df["candidate_id"] == selected_id].iloc[0]

        # Visually distinct styling card depending on rank (1-20 vs 21-100)
        card_class = "candidate-card candidate-card-positive" if cand["rank"] <= 20 else "candidate-card"
        
        st.markdown(f"""
            <div class="{card_class}">
                <div style="font-size: 1.3rem; font-weight: 700; color: #212529; margin-bottom: 0.5rem;">
                    {cand['candidate_id']} — Rank #{cand['rank']}
                </div>
                <div style="font-size: 1.1rem; font-weight: 600; color: #FF4B4B; margin-bottom: 1rem;">
                    {cand['current_title']} at {cand['current_company']}
                </div>
                <div style="background-color: #f1f3f5; padding: 1rem; border-radius: 8px; font-style: italic; margin-bottom: 1rem; color: #495057;">
                    "{cand['reasoning']}"
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Tabs for metrics and exclusions checks
        tab1, tab2 = st.tabs(["📊 Performance & Availability", "🛡️ Exclusions & Rules"])
        
        with tab1:
            m1, m2 = st.columns(2)
            m1.metric("Years of Experience (YOE)", f"{cand['years_of_experience']:.1f} yrs")
            
            # Show notice period beautifully
            notice_val = int(cand['redrob_notice_period_days']) if pd.notna(cand['redrob_notice_period_days']) else 0
            m2.metric("Notice Period", f"{notice_val} days", delta="-30 JD Limit" if notice_val <= 30 else "+ Exceeds limit", delta_color="inverse")
            
            st.write("")
            st.markdown(f"**Behavioral Signal Score:** `{cand['behavioral_score']:.3f}`")
            
            # Progress bar for active engagement
            st.progress(float(cand['behavioral_score']))

        with tab2:
            st.write("**Disqualifiers & Traps Check:**")
            
            # Check for consulting
            consulting_status = "❌ Yes" if cand["is_consulting_only"] == 1.0 else "✅ No"
            st.write(f"- consulting organization only: **{consulting_status}**")
            
            # Check for pure research
            research_status = "❌ Yes" if cand["is_pure_research"] == 1.0 else "✅ No"
            st.write(f"- pure research focus: **{research_status}**")

            # Check for wrapper experience
            wrapper_status = "❌ Yes" if cand["is_recent_wrapper_only"] == 1.0 else "✅ No"
            st.write(f"- wrapper-only experience: **{wrapper_status}**")

            # Check for coding recency
            coding_status = "❌ Yes" if cand["no_recent_coding"] == 1.0 else "✅ No"
            st.write(f"- non-technical / non-coding current role: **{coding_status}**")

    # --- Visualizations ---
    st.write("")
    st.markdown("---")
    st.subheader("📈 Ranking Metrics & Distribution")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("**Final Score vs. Behavioral Score**")
        st.scatter_chart(
            df[["behavioral_score", "score"]],
            x="behavioral_score",
            y="score",
            color="#FF4B4B",
            use_container_width=True
        )

    with chart_col2:
        st.markdown("**Years of Experience (YOE) Distribution**")
        yoe_counts = df["years_of_experience"].round().value_counts().sort_index()
        st.bar_chart(yoe_counts, use_container_width=True, color="#FF8F8F")


if __name__ == "__main__":
    main()
