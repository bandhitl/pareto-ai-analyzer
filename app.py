import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai

st.set_page_config(layout="wide")
openai.api_key = st.secrets["openai_api_key"]

st.title("PVC Pipe Production Problem Analyzer & AI Action Plan")
st.markdown("Enter manufacturing problems (with machine numbers). The system will map problems to machine parts for Heatmap analysis, and the AI expert will provide a detailed textual action plan.")

# --- Static Data ---
problem_options = [
    "Uneven wall thickness", "Rough surface (internal/external)", "Cracks or longitudinal splits",
    "Pipe warping or bending (ovality)", "Color inconsistency / streaks", "Bubbles, voids, or blisters",
    "Underfilled / short pipe", "Burning / degradation marks", "Poor gelation / unmelted particles",
    "Contamination (black specks)", "Die lines / flow marks", "Excessive flash or burrs",
    "Low impact strength", "Dimensional instability", "Bell end deformation / defects",
    "Other (please specify)"
]

# ✨ NEW: List of machine parts for heatmap columns and mapping reference ✨
machine_parts_list_for_heatmap = [
    "Feeding system", "Extruder head", "Die/mandrel", "Cooling system",
    "Haul Off unit", "Cutting unit", "Belling Unit", "Unmapped/Other" # Added for problems not in mapping
]

# ✨ NEW: Pre-defined mapping of problems to machine parts ✨
# หนึ่งปัญหาสามารถ map กับหลายส่วนของเครื่องจักรได้
default_problem_to_part_mapping = {
    "Uneven wall thickness": ["Die/mandrel", "Extruder head"],
    "Rough surface (internal/external)": ["Die/mandrel", "Cooling system", "Extruder head"],
    "Cracks or longitudinal splits": ["Cooling system", "Haul Off unit", "Die/mandrel"],
    "Pipe warping or bending (ovality)": ["Cooling system", "Die/mandrel", "Haul Off unit"],
    "Color inconsistency / streaks": ["Feeding system", "Extruder head"],
    "Bubbles, voids, or blisters": ["Extruder head", "Die/mandrel", "Feeding system"], # e.g., moisture, volatiles
    "Underfilled / short pipe": ["Extruder head", "Feeding system"],
    "Burning / degradation marks": ["Extruder head", "Die/mandrel"],
    "Poor gelation / unmelted particles": ["Extruder head", "Feeding system"],
    "Contamination (black specks)": ["Feeding system"],
    "Die lines / flow marks": ["Die/mandrel"],
    "Excessive flash or burrs": ["Die/mandrel", "Cutting unit"],
    "Low impact strength": ["Extruder head", "Cooling system"], # Assuming processing related
    "Dimensional instability": ["Die/mandrel", "Cooling system", "Haul Off unit"],
    "Bell end deformation / defects": ["Belling Unit", "Cutting unit"]
}

# --- Input Section ---
num_problem_entries = st.number_input(
    "Number of problem entries:", min_value=1, max_value=30, value=3,
    help="Enter the number of distinct problem occurrences or types you want to log."
)

problem_data = [] 

for i in range(num_problem_entries):
    st.markdown(f"--- \n**Problem Entry {i+1}**")
    col1, col2, col3 = st.columns([6, 2, 3]) # Problem Desc, Count, Machine No. (Machine Part input removed)
    
    with col1:
        problem_desc_label = f"Problem Description {i+1}"
        selected_problem = st.selectbox(problem_desc_label, problem_options, key=f"prob_desc_{i}", index=i % len(problem_options))
        if selected_problem == "Other (please specify)":
            problem_name = st.text_input(f"Specify other problem for Entry {i+1}", key=f"custom_prob_{i}")
        else:
            problem_name = selected_problem
    with col2:
        count = st.number_input(f"Count {i+1}", min_value=1, step=1, key=f"pcount_{i}", value=1)
    with col3:
        machine_no = st.text_input(f"Machine No. (Optional) {i+1}", key=f"machine_{i}")

    if problem_name.strip():
        problem_data.append({
            "Problem": problem_name.strip(),
            "Count": count,
            "Machine No.": machine_no.strip() if machine_no.strip() else "N/A"
            # "Machine Part" is no longer manually inputted here
        })

# --- Analysis Button and Logic ---
if st.button("🔍 Analyze PVC Problems & Generate Reports"):
    if not problem_data:
        st.warning("Please enter at least one problem entry.")
    else:
        df_raw = pd.DataFrame(problem_data)

        # --- Pareto Analysis (unchanged) ---
        st.header("📊 Pareto Analysis")
        # ... (โค้ด Pareto Chart เหมือนเดิมจากครั้งที่แล้วที่แก้ไขถูกต้องแล้ว) ...
        df_pareto = df_raw.groupby("Problem", as_index=False)["Count"].sum()
        df_pareto = df_pareto[df_pareto["Count"] > 0].sort_values(by="Count", ascending=False).reset_index(drop=True)
        if df_pareto.empty:
            st.error("No problems with valid counts for Pareto analysis.")
        else:
            df_pareto["Cumulative %"] = df_pareto["Count"].cumsum() / df_pareto["Count"].sum() * 100
            fig_pareto, ax1_pareto = plt.subplots(figsize=(12, 6))
            default_color = 'deepskyblue'; highlight_color = 'crimson'
            bar_colors = [default_color] * len(df_pareto)
            first_over_80_idx = -1
            for idx, cum_perc in enumerate(df_pareto["Cumulative %"]):
                if cum_perc > 80: first_over_80_idx = idx; break
            if first_over_80_idx != -1:
                for k in range(first_over_80_idx + 1): bar_colors[k] = highlight_color
            else:
                bar_colors = [highlight_color] * len(df_pareto)
            ax1_pareto.bar(df_pareto["Problem"], df_pareto["Count"], color=bar_colors)
            ax2_pareto = ax1_pareto.twinx()
            ax2_pareto.plot(df_pareto["Problem"], df_pareto["Cumulative %"], color="darkorange", marker='o', linewidth=2, linestyle='--')
            ax1_pareto.set_xlabel("Problem Description", fontweight='bold', fontsize=12)
            ax1_pareto.set_ylabel("Frequency Count", fontweight='bold', fontsize=12)
            ax2_pareto.set_ylabel("Cumulative Percentage (%)", fontweight='bold', color="darkorange", fontsize=12)
            ax1_pareto.set_xticklabels(df_pareto["Problem"], rotation=45, ha='right', fontsize=10)
            ax1_pareto.tick_params(axis='y', labelsize=10); ax2_pareto.tick_params(axis='y', labelcolor="darkorange", labelsize=10)
            ax1_pareto.grid(axis='y', linestyle='--', alpha=0.7)
            ax2_pareto.axhline(80, color='dimgray', linestyle=':', linewidth=1.5, label='80% Line'); ax2_pareto.legend(loc="upper right")
            fig_pareto.suptitle("Pareto Analysis of Production Problems", fontsize=16, fontweight='bold'); fig_pareto.tight_layout(rect=[0, 0.03, 1, 0.95])
            st.pyplot(fig_pareto)
        st.markdown("---")


        # --- ✨ Heatmap Analysis (using pre-defined mapping) ✨ ---
        st.header("🔥 Heatmap: Problem vs. Machine Part (Based on Pre-defined Mapping)")
        
        heatmap_entries = []
        for index, row in df_raw.iterrows():
            problem_name = row["Problem"]
            count = row["Count"]
            # machine_no = row["Machine No."] # Not directly used in this heatmap's aggregation by part, but kept in df_raw
            
            mapped_parts = default_problem_to_part_mapping.get(problem_name, ["Unmapped/Other"]) # Get mapped parts or default
            for part in mapped_parts:
                heatmap_entries.append({"Problem": problem_name, "Machine Part": part, "Count": count})
        
        if not heatmap_entries:
            st.info("No data available to generate heatmap based on problem mapping.")
        else:
            df_for_heatmap = pd.DataFrame(heatmap_entries)
            # Aggregate counts for heatmap: sum counts if a problem maps to multiple parts and occurs multiple times
            heatmap_pivot_data = df_for_heatmap.groupby(["Problem", "Machine Part"])["Count"].sum().unstack(fill_value=0)
            
            # Ensure all defined machine parts are columns, even if they have no data for the current problems
            heatmap_pivot_data = heatmap_pivot_data.reindex(columns=machine_parts_list_for_heatmap, fill_value=0)
            # Filter out rows (Problems) that only have zeros across all specified machine parts (optional, makes heatmap cleaner)
            heatmap_pivot_data = heatmap_pivot_data.loc[(heatmap_pivot_data != 0).any(axis=1)]


            if not heatmap_pivot_data.empty:
                fig_heatmap, ax_heatmap = plt.subplots(figsize=(14, max(8, len(heatmap_pivot_data.index) * 0.6))) # Adjusted figsize
                sns.heatmap(heatmap_pivot_data, annot=True, fmt="d", cmap="viridis", linewidths=.5, ax=ax_heatmap, cbar=True) # Changed cmap
                ax_heatmap.set_title("Heatmap of Problem Frequency by Mapped Machine Part", fontsize=15, fontweight='bold')
                ax_heatmap.set_xlabel("Machine Part", fontsize=12, fontweight='bold')
                ax_heatmap.set_ylabel("Problem Description", fontsize=12, fontweight='bold')
                plt.xticks(rotation=45, ha="right", fontsize=10)
                plt.yticks(rotation=0, fontsize=10)
                fig_heatmap.tight_layout()
                st.pyplot(fig_heatmap)
            else:
                st.info("No problem occurrences could be mapped to specific machine parts for the heatmap display.")
        st.markdown("---")


        # --- AI Textual Analysis ---
        st.header("🤖 AI Expert Textual Analysis & Action Plan")
        num_top_problems_to_analyze = min(len(df_pareto), 3)
        if num_top_problems_to_analyze > 0 and not df_pareto.empty: # Check if df_pareto is not empty
            top_problem_names = df_pareto.head(num_top_problems_to_analyze)["Problem"].tolist()
            
            problem_details_for_ai = (
                "Please analyze the following top PVC pipe production problems. "
                "For each problem, consider its total occurrences, any associated machine numbers, and typically related machine parts based on general knowledge. "
                "Provide a detailed textual report for each problem including potential causes, "
                "suggested solutions, responsible department, a short-term action plan (1-3 months), "
                "and a long-term action plan (6-12 months). Make your analysis specific to PVC pipe production.\n\n"
            )

            for i, prob_name in enumerate(top_problem_names):
                total_count = df_pareto[df_pareto["Problem"] == prob_name]["Count"].iloc[0]
                problem_details_for_ai += f"**Problem {i+1}: {prob_name}**\n"
                problem_details_for_ai += f"- Total Occurrences: {total_count}\n"
                
                occurrences = df_raw[df_raw["Problem"] == prob_name]
                machine_numbers_involved = occurrences[occurrences["Machine No."] != "N/A"]["Machine No."].unique()
                
                # ✨ Add pre-mapped machine parts to AI prompt for context ✨
                likely_machine_parts = default_problem_to_part_mapping.get(prob_name, ["Not specifically mapped"])
                problem_details_for_ai += f"- Typically Associated Machine Part(s): {', '.join(likely_machine_parts)}\n"

                if len(machine_numbers_involved) > 0:
                    problem_details_for_ai += f"- Specific Machine Nos. Logged: {', '.join(machine_numbers_involved)}\n\n"
                else:
                    problem_details_for_ai += f"- No specific machine number logged for individual occurrences.\n\n"
            
            # ... (AI call and response display - same as before)
            with st.spinner("Consulting PVC Pipe Production AI Expert and drafting detailed textual plan..."):
                system_prompt_content_text = (
                    "You are an AI assistant acting as an Expert in PVC pipe production with over 20 years of experience. "
                    "Your task is to analyze the provided manufacturing problems (including their total occurrences, any specific machine numbers logged, and typically associated machine parts based on general knowledge for context). "
                    "For each problem, provide a comprehensive textual analysis. Structure your response clearly for each problem. "
                    "For each problem, you MUST discuss all the following aspects in a narrative or bulleted text format (do NOT use tables):\n"
                    "1.  **Problem Statement:** Briefly restate the problem name, its total occurrences. Mention associated machine numbers if logged, and acknowledge the typically related machine parts provided for context.\n"
                    "2.  **Potential Causes (PVC Specific):** Detail specific potential causes rooted in PVC pipe manufacturing, considering the typically associated machine parts and specific machine numbers if relevant.\n"
                    "3.  **Suggested Solutions (PVC Specific):** Propose actionable solutions tailored to PVC pipe production, referencing typically associated machine parts and specific machine numbers where applicable.\n"
                    "4.  **Responsible Department:** Identify the primary department(s) accountable.\n"
                    "5.  **Short-Term Action/Plan (1-3 months):** Outline concrete, immediate actions.\n"
                    "6.  **Long-Term Action/Plan (6-12 months):** Describe strategic actions for sustained improvement.\n\n"
                    "Ensure your response is a well-organized textual report. Use markdown formatting like bold headings and bullet points. Do NOT output any JSON or table structures."
                )
                ai_response_text = ""
                try:
                    response = openai.chat.completions.create(
                        model="gpt-4o", 
                        messages=[{"role": "system", "content": system_prompt_content_text}, {"role": "user", "content": problem_details_for_ai}],
                        temperature=0.3
                    )
                    ai_response_text = response.choices[0].message.content
                    st.markdown(ai_response_text)
                except openai.APIError as e: st.error(f"OpenAI API error: {e}")
                except Exception as e: st.error(f"Unexpected error: {e}"); st.text(f"AI response: {ai_response_text}")
        else:
            st.info("Not enough distinct problems identified from Pareto analysis to forward to AI.")

        # --- Download Buttons (unchanged) ---
        st.markdown("---"); st.subheader("📥 Download Data")
        col_dl1, col_dl2 = st.columns(2)
        # ... (Download buttons code - same as before)
        with col_dl1:
            if not df_pareto.empty:
                csv_pareto = df_pareto.to_csv(index=False).encode('utf-8')
                st.download_button("Download Pareto Analysis Data (CSV)", data=csv_pareto, file_name="pvc_pareto_analysis_data.csv", mime="text/csv", key="dl_pareto")
        with col_dl2:
            csv_raw = df_raw.to_csv(index=False).encode('utf-8')
            st.download_button("Download Raw Input Data (CSV)", data=csv_raw, file_name="pvc_raw_input_data.csv", mime="text/csv", key="dl_raw")
