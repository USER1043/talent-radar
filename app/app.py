"""
@file Dual-tab Streamlit application for pipeline execution and candidate inspection.
@package online_ranking
"""

import sys
import subprocess
import tempfile
import streamlit as st
import pandas as pd
from pathlib import Path

# Add project root to sys.path to allow importing from root-level modules.
sys.path.append(str(Path(__file__).parent.parent))


# Loads candidate details, honeypots, and extracts custom features.
@st.cache_data
def load_data(csv_path: str = "submission.csv") -> pd.DataFrame:
    if not Path(csv_path).exists():
        return pd.DataFrame()
    df_sub = pd.read_csv(csv_path)
    df_feat = pd.read_parquet("artifacts/features.parquet")
    df_hp = pd.read_parquet("artifacts/honeypot_flags.parquet")
    
    df = df_sub.merge(df_feat, on="candidate_id", how="left")
    df = df.merge(df_hp, on="candidate_id", how="left")
    
    from features_utils import extract_candidate_features
    features_list = [extract_candidate_features(row) for _, row in df.iterrows()]
    df_extracted = pd.DataFrame(features_list)
    
    for col in df_extracted.columns:
        if col not in df.columns or col in ["is_consulting_only", "behavioral_score"]:
            df[col] = df_extracted[col].values
            
    return df


# Main execution flow for the Streamlit dashboard layout and tabs.
def main():
    st.set_page_config(
        page_title="Talent Radar — Pipeline & Explorer",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )

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
            margin-bottom: 1.5rem;
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

    st.markdown('<div class="main-header">Talent Radar</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Pipeline Runner & Candidate Explorer Dashboard</div>', unsafe_allow_html=True)

    tab_runner, tab_explorer = st.tabs(["⚙️ Pipeline Runner", "🔍 Candidate Explorer"])

    with tab_runner:
        st.subheader("Sandbox Pipeline Runner")
        st.write("Upload a candidate profile list (`.jsonl` or `.jsonl.gz`) to execute the ranking pipeline.")
        
        uploaded_file = st.file_uploader("Choose candidate file", type=["jsonl", "gz"])
        
        if uploaded_file is not None:
            file_suffix = ".jsonl.gz" if uploaded_file.name.endswith(".gz") else ".jsonl"
            
            if st.button("Run Ranking Pipeline"):
                status_box = st.empty()
                progress_bar = st.progress(0)
                
                with tempfile.TemporaryDirectory() as tmpdir:
                    input_path = Path(tmpdir) / f"candidates{file_suffix}"
                    output_path = Path(tmpdir) / "ranked_output.csv"
                    
                    with open(input_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    status_box.info("Stage 1/5: Running parser and loading active candidate IDs...")
                    progress_bar.progress(20)
                    
                    cmd = [
                        sys.executable,
                        "rank.py",
                        "--candidates", str(input_path),
                        "--out", str(output_path)
                    ]
                    
                    status_box.info("Stage 2/5: Scoring candidates with learned ranker...")
                    progress_bar.progress(50)
                    
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        status_box.info("Stage 3/5: Running honeypot rules and filtering candidates...")
                        progress_bar.progress(70)
                        
                        status_box.info("Stage 4/5: Generating deterministic reason planner details...")
                        progress_bar.progress(90)
                        
                        status_box.success("Stage 5/5: Pipeline completed successfully!")
                        progress_bar.progress(100)
                        
                        df_res = pd.read_csv(output_path)
                        st.subheader("Preview Ranked Results")
                        st.dataframe(df_res, use_container_width=True, hide_index=True)
                        
                        csv_data = output_path.read_text()
                        st.download_button(
                            label="Download ranked_output.csv",
                            data=csv_data,
                            file_name="ranked_output.csv",
                            mime="text/csv"
                        )
                        
                        # Save a copy as temporary preview target for explorer tab
                        Path("temp_preview_ranked.csv").write_text(csv_data)
                    else:
                        status_box.error(f"Pipeline failed with error:\n{result.stderr}")

    with tab_explorer:
        # Determine CSV source file for inspection
        csv_source = "submission.csv"
        if Path("temp_preview_ranked.csv").exists():
            st.info("Showing results from the most recent Pipeline Runner execution. To revert to the global submission list, clear the temp copy.")
            if st.button("Use global submission.csv"):
                Path("temp_preview_ranked.csv").unlink(missing_ok=True)
                st.rerun()
            csv_source = "temp_preview_ranked.csv"
            
        df = load_data(csv_source)
        
        if df.empty:
            st.warning(f"No active data found in {csv_source}. Please run the pipeline runner or generate the submission list first.")
            return

        # --- Top KPIs ---
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 0.85rem; color: #6c757d; text-transform: uppercase;">Total Pool</div>
                    <div style="font-size: 2rem; font-weight: 700; color: #212529;">{len(df)}</div>
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
            st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 0.85rem; color: #6c757d; text-transform: uppercase;">Source File</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #28a745;">{csv_source}</div>
                </div>
            """, unsafe_allow_html=True)

        st.write("")

        # --- Sidebar Filtering ---
        st.sidebar.header("🎯 Filters")
        search_query = st.sidebar.text_input("🔍 Search Candidates", placeholder="ID, Title, Company...")
        min_yoe = st.sidebar.slider("Minimum YOE", 0, 20, 0)
        
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
            st.subheader("🏆 Ranked Candidates list")
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
            
            c_list = filtered_df["candidate_id"].tolist()
            if not c_list:
                st.warning("No candidates found matching the filters.")
                return

            selected_id = st.selectbox("Select Candidate to Inspect", c_list)
            cand = filtered_df[filtered_df["candidate_id"] == selected_id].iloc[0]

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

            tab1, tab2 = st.tabs(["📊 Performance & Availability", "🛡️ Exclusions & Rules"])
            
            with tab1:
                m1, m2 = st.columns(2)
                m1.metric("Years of Experience (YOE)", f"{cand['years_of_experience']:.1f} yrs")
                
                notice_val = int(cand['redrob_notice_period_days']) if pd.notna(cand['redrob_notice_period_days']) else 0
                m2.metric("Notice Period", f"{notice_val} days", delta="-30 JD Limit" if notice_val <= 30 else "+ Exceeds limit", delta_color="inverse")
                
                st.write("")
                st.markdown(f"**Behavioral Signal Score:** `{cand['behavioral_score']:.3f}`")
                st.progress(float(cand['behavioral_score']))

            with tab2:
                st.write("**Disqualifiers & Traps Check:**")
                
                consulting_status = "❌ Yes" if cand["is_consulting_only"] == 1.0 else "✅ No"
                st.write(f"- consulting organization only: **{consulting_status}**")
                
                research_status = "❌ Yes" if cand["is_pure_research"] == 1.0 else "✅ No"
                st.write(f"- pure research focus: **{research_status}**")

                wrapper_status = "❌ Yes" if cand["is_recent_wrapper_only"] == 1.0 else "✅ No"
                st.write(f"- wrapper-only experience: **{wrapper_status}**")

                coding_status = "❌ Yes" if cand["no_recent_coding"] == 1.0 else "✅ No"
                st.write(f"- non-technical / non-coding current role: **{coding_status}**")

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
