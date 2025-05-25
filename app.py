import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Import seaborn for heatmap
import openai

# --- Page Config and OpenAI Key ---
st.set_page_config(layout="wide")
openai.api_key = st.secrets["openai_api_key"]

# --- App Title and Description ---
st.title("PVC Pipe Production Problem Analyzer & AI Action Plan")
st.markdown("Enter manufacturing problems (with machine numbers and parts). The AI expert will analyze, and the system will show Pareto & Heatmap analyses, plus a detailed textual action plan.")

# --- Input Section ---
num_problem_entries = st.number_input(
    "Number of problem entries:", 
    min_value=1, 
    max_value=30,  # Increased max entries for more heatmap data
    value=3,
    help="Enter the number of distinct problem occurrences or types you want to log. You can log the same problem type multiple times if it occurs on different machines/parts or at different times."
)

problem_data = [] 

problem_options = [
    "Uneven wall thickness", "Rough surface (internal/external)", "Cracks or longitudinal splits",
    "Pipe warping or bending (ovality)", "Color inconsistency / streaks", "Bubbles, voids, or blisters",
    "Underfilled / short pipe", "Burning / degradation marks", "Poor gelation / unmelted particles",
    "Contamination (black specks)", "Die lines / flow marks", "Excessive flash or burrs",
    "Low impact strength", "Dimensional instability", "Bell end deformation / defects",
    "Other (please specify)"
]

# ✨ NEW: Machine parts options ✨
machine_parts_options = [
    "N/A or Other",
    "Feeding system",
    "Extruder head",
    "Die/mandrel",
    "Cooling system",
    "Haul Off unit",
    "Cutting unit",
    "Belling Unit"
]

for i in range(num_problem_entries):
    st.markdown(f"--- \n**Problem Entry {i+1}**")
    # Adjusted column layout for new input field
    col1, col2, col3, col4 = st.columns([6, 2, 3, 4]) 
    
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
    with col4: # ✨ NEW: Input for Machine Part ✨
        machine_part = st.selectbox(f"Machine Part Involved {i+1}", machine_parts_options, key=f"mpart_{i}")

    if problem_name.strip():
        problem_data.append({
            "Problem": problem_name.strip(),
            "Count": count,
            "Machine No.": machine_no.strip() if machine_no.strip() else "N/A",
            "Machine Part": machine_part # Store the selected machine part
        })

# --- Analysis Button and Logic ---
if st.button("🔍 Analyze PVC Problems & Generate Reports"):
    if not problem_data:
        st.warning("Please enter at least one problem entry.")
    else:
        df_raw = pd.DataFrame(problem_data)

        # --- Pareto Analysis ---
        st.header("📊 Pareto Analysis")
        df_pareto = df_raw.groupby("Problem", as_index=False)["Count"].sum()
        df_pareto = df_pareto[df_pareto["Count"] > 0].sort_values(by="Count", ascending=False).reset_index(drop=True)

        if df_pareto.empty:
            st.error("No problems with valid counts found for Pareto analysis.")
        else:
            df_pareto["Cumulative %"] = df_pareto["Count"].cumsum() / df_pareto["Count"].sum() * 100
            # ... (Pareto chart plotting code - unchanged from previous correct version)
            fig_pareto, ax1_pareto = plt.subplots(figsize=(12, 6))
            default_color = 'deepskyblue'
            highlight_color = 'crimson'
            bar_colors = [default_color] * len(df_pareto)
            first_over_80_idx = -1
            for idx, cum_perc in enumerate(df_pareto["Cumulative %"]):
                if cum_perc > 80:
                    first_over_80_idx = idx
                    break
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
            ax1_pareto.tick_params(axis='y', labelsize=10)
            ax2_pareto.tick_params(axis='y', labelcolor="darkorange", labelsize=10)
            ax1_pareto.grid(axis='y', linestyle='--', alpha=0.7)
            ax2_pareto.axhline(80, color='dimgray', linestyle=':', linewidth=1.5, label='80% Line')
            ax2_pareto.legend(loc="upper right")
            fig_pareto.suptitle("Pareto Analysis of Production Problems", fontsize=16, fontweight='bold')
            fig_pareto.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
            st.pyplot(fig_pareto)
            st.markdown("---") # Separator

        # --- ✨ NEW: Heatmap Analysis ✨ ---
        st.header("🔥 Heatmap: Problem vs. Machine Part")
        if "Machine Part" in df_raw.columns and not df_raw[df_raw["Machine Part"] != "N/A or Other"].empty:
            # Filter out "N/A or Other" for a cleaner heatmap, or include it if preferred
            df_heatmap_source = df_raw[df_raw["Machine Part"] != "N/A or Other"]
            
            if not df_heatmap_source.empty:
                heatmap_data = df_heatmap_source.groupby(["Problem", "Machine Part"])["Count"].sum().unstack(fill_value=0)
                
                if not heatmap_data.empty:
                    fig_heatmap, ax_heatmap = plt.subplots(figsize=(12, max(8, len(heatmap_data.index) * 0.5))) # Adjust height based on number of problems
                    sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlGnBu", linewidths=.5, ax=ax_heatmap, cbar=True)
                    ax_heatmap.set_title("Heatmap of Problem Frequency by Machine Part", fontsize=15, fontweight='bold')
                    ax_heatmap.set_xlabel("Machine Part", fontsize=12, fontweight='bold')
                    ax_heatmap.set_ylabel("Problem Description", fontsize=12, fontweight='bold')
                    plt.xticks(rotation=45, ha="right")
                    plt.yticks(rotation=0)
                    fig_heatmap.tight_layout()
                    st.pyplot(fig_heatmap)
                else:
                    st.info("Not enough data (after filtering 'N/A or Other') to generate a heatmap of problems by specific machine parts.")
            else:
                st.info("No problems were associated with specific machine parts (excluding 'N/A or Other'). Heatmap cannot be generated.")
        else:
            st.info("Machine part information is missing or insufficient to generate a heatmap.")
        st.markdown("---") # Separator


        # --- AI Textual Analysis ---
        st.header("🤖 AI Expert Textual Analysis & Action Plan")
        # ... (AI analysis part - unchanged from previous correct version)
        num_top_problems_to_analyze = min(len(df_pareto), 3)
        if num_top_problems_to_analyze > 0:
            top_problem_names = df_pareto.head(num_top_problems_to_analyze)["Problem"].tolist()
            problem_details_for_ai = (
                "Please analyze the following top PVC pipe production problems. "
                "For each problem, consider its total occurrences and any associated machine numbers/parts. " # Added parts
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
                machine_parts_involved = occurrences[occurrences["Machine Part"] != "N/A or Other"]["Machine Part"].unique() # Get machine parts
                
                if len(machine_numbers_involved) > 0:
                    problem_details_for_ai += f"- Associated Machine Nos.: {', '.join(machine_numbers_involved)}\n"
                if len(machine_parts_involved) > 0: # Add machine parts to AI prompt
                    problem_details_for_ai += f"- Associated Machine Parts: {', '.join(machine_parts_involved)}\n\n"
                elif len(machine_numbers_involved) == 0 and len(machine_parts_involved) == 0 :
                     problem_details_for_ai += f"- No specific machine number or part provided for individual occurrences.\n\n"
                else:
                    problem_details_for_ai += "\n"


            with st.spinner("Consulting PVC Pipe Production AI Expert and drafting detailed textual plan..."):
                system_prompt_content_text = (
                    "You are an AI assistant acting as an Expert in PVC pipe production with over 20 years of experience. "
                    "Your task is to analyze the provided manufacturing problems (including their total occurrences and any associated machine numbers or machine parts). " # Mention machine parts
                    "For each problem, provide a comprehensive textual analysis. Structure your response clearly for each problem. "
                    "For each problem, you MUST discuss all the following aspects in a narrative or bulleted text format (do NOT use tables):\n"
                    "1.  **Problem Statement:** Briefly restate the problem name, its total occurrences, and mention associated machine numbers/parts if any.\n" # Mention machine parts
                    "2.  **Potential Causes (PVC Specific):** Detail specific potential causes rooted in PVC pipe manufacturing, considering machine numbers/parts where relevant.\n" # Mention machine parts
                    "3.  **Suggested Solutions (PVC Specific):** Propose actionable solutions tailored to PVC pipe production, referencing machine numbers/parts where applicable.\n" # Mention machine parts
                    "4.  **Responsible Department:** Identify the primary department(s) accountable.\n"
                    "5.  **Short-Term Action/Plan (1-3 months):** Outline concrete, immediate actions.\n"
                    "6.  **Long-Term Action/Plan (6-12 months):** Describe strategic actions for sustained improvement.\n\n"
                    "Ensure your response is a well-organized textual report. Use markdown formatting like bold headings and bullet points. Do NOT output any JSON or table structures."
                )
                
                ai_response_text = ""
                try:
                    response = openai.chat.completions.create(
                        model="gpt-4o", 
                        messages=[
                            {"role": "system", "content": system_prompt_content_text},
                            {"role": "user", "content": problem_details_for_ai}
                        ],
                        temperature=0.3
                    )
                    ai_response_text = response.choices[0].message.content
                    st.markdown(ai_response_text)

                except openai.APIError as e:
                    st.error(f"An OpenAI API error occurred: {e}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
                    if ai_response_text: st.text(f"Raw AI response during error: {ai_response_text}")
        else:
            st.info("Not enough problems identified from Pareto analysis to forward to AI.")
            
        # --- Download Buttons ---
        st.markdown("---") # Separator
        st.subheader("📥 Download Data")
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            if not df_pareto.empty:
                csv_pareto = df_pareto.to_csv(index=False).encode('utf-8')
                st.download_button("Download Pareto Analysis Data (CSV)", data=csv_pareto, file_name="pvc_pareto_analysis_data.csv", mime="text/csv", key="dl_pareto")
        with col_dl2:
            csv_raw = df_raw.to_csv(index=False).encode('utf-8')
            st.download_button("Download Raw Input Data (CSV)", data=csv_raw, file_name="pvc_raw_input_data.csv", mime="text/csv", key="dl_raw")
