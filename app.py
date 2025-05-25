import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import openai
# import json # ไม่จำเป็นต้องใช้ json อีกต่อไป

st.set_page_config(layout="wide")

openai.api_key = st.secrets["openai_api_key"]

st.title("PVC Pipe Production Problem Analyzer & AI Action Plan")
st.markdown("Enter manufacturing problems (optionally with machine numbers). The AI expert will analyze, show an enhanced Pareto chart, and provide a detailed textual action plan.")

# --- 1. ส่วนรับข้อมูลปัญหา และเพิ่ม "เลขที่เครื่องจักร" ---
num_problems = st.number_input("Number of problem entries:", min_value=1, max_value=20, value=3,
                               help="Enter the number of distinct problem occurrences or types you want to log. You can log the same problem type multiple times if it occurs on different machines or at different times.")

problem_data = [] # ใช้เก็บข้อมูลแต่ละรายการปัญหา

problem_options = [
    "Uneven wall thickness", "Rough surface (internal/external)", "Cracks or longitudinal splits",
    "Pipe warping or bending (ovality)", "Color inconsistency / streaks", "Bubbles, voids, or blisters",
    "Underfilled / short pipe", "Burning / degradation marks", "Poor gelation / unmelted particles",
    "Contamination (black specks)", "Die lines / flow marks", "Excessive flash or burrs",
    "Low impact strength", "Dimensional instability", "Bell end deformation / defects",
    "Other (please specify)"
]

for i in range(num_problems):
    st.markdown(f"--- \n**Problem Entry {i+1}**")
    col1, col2, col3 = st.columns([3, 1, 2]) # ปรับสัดส่วนคอลัมน์
    
    with col1:
        problem_desc_label = f"Problem Description {i+1}"
        selected_problem = st.selectbox(problem_desc_label, problem_options, key=f"prob_desc_{i}", index=i % len(problem_options))
        if selected_problem == "Other (please specify)":
            problem_name = st.text_input(f"Specify other problem for Entry {i+1}", key=f"custom_prob_{i}")
        else:
            problem_name = selected_problem
    with col2:
        count = st.number_input(f"Count {i+1}", min_value=1, step=1, key=f"pcount_{i}", value=1) # Default count to 1
    with col3:
        machine_no = st.text_input(f"Machine No. (Optional) for Entry {i+1}", key=f"machine_{i}")

    if problem_name.strip(): # ตรวจสอบว่ามีการระบุชื่อปัญหา
        problem_data.append({
            "Problem": problem_name.strip(),
            "Count": count,
            "Machine No.": machine_no.strip() if machine_no.strip() else "N/A"
        })

if st.button("🔍 Analyze PVC Problems & Generate Action Plan"):
    if not problem_data:
        st.warning("Please enter at least one problem entry.")
    else:
        df_raw = pd.DataFrame(problem_data)

        # สร้าง DataFrame สำหรับ Pareto Chart (รวมปัญหาชื่อเดียวกัน)
        df_pareto = df_raw.groupby("Problem", as_index=False)["Count"].sum()
        df_pareto = df_pareto[df_pareto["Count"] > 0].sort_values(by="Count", ascending=False).reset_index(drop=True)

        if df_pareto.empty:
            st.error("No problems with valid counts found for Pareto analysis.")
        else:
            df_pareto["Cumulative %"] = df_pareto["Count"].cumsum() / df_pareto["Count"].sum() * 100

            # --- 2. การเน้นสีแท่ง Pareto สำหรับ 80% แรก ---
            st.subheader("Pareto Chart of PVC Pipe Production Problems")
            fig, ax1 = plt.subplots(figsize=(12, 6)) # ปรับขนาดกราฟให้กว้างขึ้นเล็กน้อย

            default_color = 'deepskyblue'
            highlight_color = 'crimson' # สีแดงสำหรับเน้น
            
            bar_colors = [default_color] * len(df_pareto)
            first_over_80_idx = -1
            for idx, cum_perc in enumerate(df_pareto["Cumulative %"]):
                if cum_perc > 80:
                    first_over_80_idx = idx
                    break
            
            if first_over_80_idx != -1: # ถ้ามีปัญหาที่ทำให้เกิน 80%
                for i in range(first_over_80_idx + 1):
                    bar_colors[i] = highlight_color
            else: # ถ้าทุกปัญหารวมกันแล้วยังไม่เกิน 80% (หรือเท่ากับ 80% พอดี)
                bar_colors = [highlight_color] * len(df_pareto)

            ax1.bar(df_pareto["Problem"], df_pareto["Count"], color=bar_colors)
            
            ax2 = ax1.twinx()
            ax2.plot(df_pareto["Problem"], df_pareto["Cumulative %"], color="darkorange", marker='o', linewidth=2, linestyle='--') #เปลี่ยนสีเส้น%
            
            ax1.set_xlabel("Problem Description", fontweight='bold', fontsize=12)
            ax1.set_ylabel("Frequency Count", fontweight='bold', fontsize=12)
            ax2.set_ylabel("Cumulative Percentage (%)", fontweight='bold', color="darkorange", fontsize=12)
            
            ax1.tick_params(axis='x', rotation=45, ha='right', labelsize=10)
            ax1.tick_params(axis='y', labelsize=10)
            ax2.tick_params(axis='y', labelcolor="darkorange", labelsize=10)

            ax1.grid(axis='y', linestyle='--', alpha=0.7)
            ax2.axhline(80, color='dimgray', linestyle=':', linewidth=1.5, label='80% Line') #ปรับเส้น 80%
            ax2.legend(loc="upper right")
            
            plt.title("Pareto Analysis of Production Problems", fontsize=16, fontweight='bold')
            fig.tight_layout()
            st.pyplot(fig)

            # --- 3. เตรียมข้อมูลสำหรับ AI (รวมเลขที่เครื่องจักร) และเปลี่ยน AI Output เป็น Text ---
            num_top_problems_to_analyze = min(len(df_pareto), 3) # วิเคราะห์ Top 3 ปัญหา
            top_problem_names = df_pareto.head(num_top_problems_to_analyze)["Problem"].tolist()

            problem_details_for_ai = (
                "Please analyze the following top PVC pipe production problems. "
                "For each problem, consider its total occurrences and any associated machine numbers. "
                "Provide a detailed textual report for each problem including potential causes, "
                "suggested solutions, responsible department, a short-term action plan (1-3 months), "
                "and a long-term action plan (6-12 months). Make your analysis specific to PVC pipe production.\n\n"
            )

            for i, prob_name in enumerate(top_problem_names):
                total_count = df_pareto[df_pareto["Problem"] == prob_name]["Count"].iloc[0]
                problem_details_for_ai += f"**Problem {i+1}: {prob_name}**\n"
                problem_details_for_ai += f"- Total Occurrences: {total_count}\n"
                
                # ดึงข้อมูลเลขเครื่องจักรที่เกี่ยวข้องกับปัญหานี้จาก df_raw
                occurrences = df_raw[df_raw["Problem"] == prob_name]
                machine_numbers_involved = occurrences[occurrences["Machine No."] != "N/A"]["Machine No."].unique()
                
                if len(machine_numbers_involved) > 0:
                    problem_details_for_ai += f"- Associated Machine Nos.: {', '.join(machine_numbers_involved)}\n\n"
                else:
                    problem_details_for_ai += f"- No specific machine number provided for individual occurrences.\n\n"

            st.subheader("⚙️ AI Expert Textual Analysis & Action Plan for PVC Pipe Production")
            with st.spinner("Consulting PVC Pipe Production AI Expert and drafting detailed textual plan..."):
                
                system_prompt_content_text = (
                    "You are an AI assistant acting as an Expert in PVC pipe production with over 20 years of experience. "
                    "Your task is to analyze the provided manufacturing problems (including their total occurrences and any associated machine numbers). "
                    "For each problem, provide a comprehensive textual analysis. Structure your response clearly for each problem. "
                    "For each problem, you MUST discuss all the following aspects in a narrative or bulleted text format (do NOT use tables):\n"
                    "1.  **Problem Statement:** Briefly restate the problem name, its total occurrences, and mention associated machine numbers if any.\n"
                    "2.  **Potential Causes (PVC Specific):** Detail specific potential causes deeply rooted in PVC pipe manufacturing processes, materials, or equipment (considering the machine numbers if relevant).\n"
                    "3.  **Suggested Solutions (PVC Specific):** Propose actionable, practical solutions tailored to PVC pipe production. These should include adjustments in formulation, process parameters, equipment maintenance/calibration, tooling, or quality control, referencing machine numbers where applicable.\n"
                    "4.  **Responsible Department:** Identify the primary department(s) accountable (e.g., Production, Maintenance, Quality Assurance (QA), Process Engineering, Formulation/R&D, Raw Material Procurement, Tooling, Management).\n"
                    "5.  **Short-Term Action/Plan (1-3 months):** Outline concrete, immediate actions or a concise plan for mitigation or resolution. Be specific.\n"
                    "6.  **Long-Term Action/Plan (6-12 months):** Describe strategic actions or a comprehensive plan for sustained improvement or permanent solutions (e.g., equipment upgrades, process re-engineering, training programs).\n\n"
                    "Ensure your response is a well-organized textual report. Use markdown formatting like bold headings (e.g., **Potential Causes (PVC Specific):**) and bullet points (e.g., using '-' or '*') for readability where appropriate within your textual explanation for each problem. Do NOT output any JSON or table structures."
                )
                
                ai_response_text = "" # Initialize for debugging
                try:
                    response = openai.chat.completions.create(
                        model="gpt-4o", 
                        messages=[
                            {"role": "system", "content": system_prompt_content_text},
                            {"role": "user", "content": problem_details_for_ai}
                        ],
                        temperature=0.3 # Temperature ปานกลางสำหรับ text generation
                    )
                    ai_response_text = response.choices[0].message.content
                    
                    st.markdown(ai_response_text) # แสดงผลลัพธ์ AI เป็น Text

                except openai.APIError as e:
                    st.error(f"An OpenAI API error occurred: {e}")
                    st.error("Please check your API key, account quota, model access, and network connection.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
                    if ai_response_text:
                        st.text("Raw AI response during unexpected error:")
                        st.text(ai_response_text)
            
            # Download button (ข้อมูลดิบ Pareto)
            csv_pareto = df_pareto.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Pareto Analysis Data (CSV)", data=csv_pareto, file_name="pvc_pareto_analysis_data.csv", mime="text/csv")
            
            # Optionally, allow download of raw input data
            csv_raw = df_raw.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Raw Input Data (CSV)", data=csv_raw, file_name="pvc_raw_input_data.csv", mime="text/csv")
