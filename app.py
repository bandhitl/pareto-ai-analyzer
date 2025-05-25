import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import openai

# Set OpenAI API key from secrets
openai.api_key = st.secrets["openai_api_key"]

st.title("PVC Pipe Production Problem Analyzer & AI Action Plan") # ปรับ Title
st.markdown("Enter manufacturing problems. The AI expert in PVC pipe production will analyze, show a Pareto chart, and provide a detailed action plan table with short and long-term solutions.")

num_problems = st.number_input("Number of problems:", min_value=1, max_value=10, value=3) # อาจลด max_value ลงถ้าตารางจะยาวมาก

problem_names = []
problem_counts = []

# ปัญหาที่พบบ่อยในการผลิตท่อ PVC (สามารถปรับปรุง/เพิ่มเติมได้)
problem_options = [
    "Uneven wall thickness",             # ความหนาผนังไม่สม่ำเสมอ
    "Rough surface (internal/external)", # ผิวหยาบ (ภายใน/ภายนอก)
    "Cracks or longitudinal splits",     # รอยแตกหรือการแยกตามยาว
    "Pipe warping or bending (ovality)", # ท่อบิดงอหรือเสียรูป (ความรี)
    "Color inconsistency / streaks",     # สีไม่สม่ำเสมอ / เป็นเส้น
    "Bubbles, voids, or blisters",       # ฟองอากาศ, โพรง, หรือรอยพุพอง
    "Underfilled / short pipe",          # ฉีดไม่เต็ม / ท่อสั้น
    "Burning / degradation marks",       # รอยไหม้ / การเสื่อมสภาพของพลาสติก
    "Poor gelation / unmelted particles",# การหลอมละลายไม่สมบูรณ์ / มีอนุภาคไม่หลอมละลาย
    "Contamination (black specks)",      # การปนเปื้อน (จุดดำ)
    "Die lines / flow marks",            # เส้นจากดาย / รอยการไหล
    "Excessive flash or burrs",          # ครีบส่วนเกิน
    "Low impact strength",               # ความทนแรงกระแทกต่ำ
    "Dimensional instability",           # ความไม่เสถียรของมิติ
    "Bell end deformation / defects",    # ปลายบานเสียรูป / มีตำหนิ
    "Other (please specify)"
]


for i in range(num_problems):
    col1, col2 = st.columns([3, 1])
    with col1:
        selected = st.selectbox(f"Problem {i+1}", problem_options, key=f"prob_{i}", index=i % len(problem_options)) #กระจาย default selection
        if selected == "Other (please specify)":
            name = st.text_input(f"Specify problem {i+1}", key=f"custom_prob_{i}")
        else:
            name = selected
    with col2:
        count = st.number_input(f"Count {i+1}", min_value=0, step=1, key=f"pcount_{i}")
    problem_names.append(name)
    problem_counts.append(count)

if st.button("🔍 Analyze PVC Problems & Generate Detailed Action Plan"): # ปรับชื่อปุ่ม
    df_raw = pd.DataFrame({"Problem": problem_names, "Count": problem_counts})
    df = df_raw.groupby("Problem", as_index=False).sum()
    df = df[df["Count"] > 0].sort_values(by="Count", ascending=False).reset_index(drop=True)
    
    if df.empty:
        st.warning("Please enter at least one problem with a count greater than 0.")
    else:
        df["Cumulative %"] = df["Count"].cumsum() / df["Count"].sum() * 100

        st.subheader("Pareto Chart of PVC Pipe Production Problems")
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.bar(df["Problem"], df["Count"], color="deepskyblue") # เปลี่ยนสี
        ax2 = ax1.twinx()
        ax2.plot(df["Problem"], df["Cumulative %"], color="crimson", marker='o', linewidth=2) # เปลี่ยนสีและเพิ่มความหนาเส้น
        ax1.set_ylabel("Frequency Count", fontweight='bold')
        ax2.set_ylabel("Cumulative Percentage (%)", fontweight='bold')
        ax1.tick_params(axis='x', rotation=45, ha='right')
        ax1.grid(axis='y', linestyle='--', alpha=0.7) # เพิ่ม gridline
        ax2.axhline(80, color='gray', linestyle='--', label='80% Vital Few Line')
        ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=1) # ปรับตำแหน่ง legend
        fig.tight_layout() #ปรับ layout ให้พอดี
        st.pyplot(fig)

        # --- ส่วนของการเรียก AI และแสดงผลเป็นตาราง ---
        # เลือก top N ปัญหาจาก Pareto chart (เช่น top 3)
        num_top_problems_to_analyze = min(len(df), 3) # วิเคราะห์ไม่เกิน 3 ปัญหา หรือน้อยกว่าถ้ามีไม่ถึง
        top_problems_df = df.head(num_top_problems_to_analyze)
        
        problem_list_for_ai = f"Analyze the following top {num_top_problems_to_analyze} PVC pipe production problems and provide a detailed action plan:\n"
        for idx, row in top_problems_df.iterrows():
            problem_list_for_ai += f"{idx+1}. Problem: \"{row['Problem']}\" (occurred {row['Count']} times)\n"

        st.subheader("⚙️ AI Expert Analysis & Action Plan for PVC Pipe Production") # ปรับ Subheader
        with st.spinner("Consulting PVC Pipe Production AI Expert and drafting detailed action plan... This may take some time."): # ปรับ Spinner text
            
            system_prompt_content = (
                "You are an AI assistant acting as an Expert in PVC pipe production with over 20 years of experience. "
                "Your task is to analyze the provided manufacturing problems, which are specific to PVC pipe production. "
                "For each problem, provide a detailed analysis. Present your entire analysis ONLY as a markdown table with the following columns:\n"
                "1. 'Problem': The name of the PVC pipe production problem as provided.\n"
                "2. 'Potential Cause (PVC Specific)': Specific potential causes deeply rooted in PVC pipe manufacturing processes (e.g., extrusion issues, die design, formulation imbalances - PVC resin, stabilizers, lubricants, plasticizers, fillers, pigments), raw material quality, or equipment condition (e.g., extruder screw wear, die head contamination, vacuum tank calibration, haul-off synchronization, cutter precision).\n"
                "3. 'Suggested Solution (PVC Specific)': Actionable, practical solutions tailored to PVC pipe production. These should include adjustments in PVC formulation, process parameter optimization (e.g., melt/die temperatures, screw speed, line speed, vacuum levels, cooling rates), equipment maintenance/calibration, tooling adjustments, or enhanced quality control procedures at various stages.\n"
                "4. 'Responsible Department': The primary department accountable (e.g., Production, Maintenance, Quality Assurance (QA), Process Engineering, Formulation/R&D, Raw Material Procurement, Tooling).\n"
                "5. 'Short-Term Action/Plan (1-3 months)': Concrete, immediate actions or a concise plan that can be implemented within 1 to 3 months to mitigate or resolve the problem. Include specific checks or trials.\n"
                "6. 'Long-Term Action/Plan (6-12 months)': Strategic actions or a comprehensive plan for sustained improvement or permanent solutions. This may involve equipment upgrades, process re-engineering, supplier development, comprehensive training programs, or implementation of advanced quality systems (e.g., SPC).\n\n"
                "The possible departments are: Production, Maintenance, Quality Assurance (QA), Process Engineering, Formulation/R&D, Raw Material Procurement, Tooling, Management.\n"
                "Ensure your response contains only the markdown table and no other introductory or concluding text. Your advice should be highly practical and reflect deep expertise in PVC pipe manufacturing challenges."
            )
            
            try:
                response = openai.chat.completions.create(
                    model="gpt-4o",  # หรือ "gpt-4-turbo" หรือ "gpt-4" (gpt-4o อาจจะเร็วและคุ้มค่ากว่า)
                    messages=[
                        {"role": "system", "content": system_prompt_content},
                        {"role": "user", "content": problem_list_for_ai}
                    ],
                    temperature=0.25 # รักษาอุณหภูมิต่ำเพื่อความแม่นยำและสอดคล้องกับโครงสร้าง
                )
                ai_response_table_markdown = response.choices[0].message.content
                
                st.markdown(ai_response_table_markdown)

            except openai.APIError as e:
                st.error(f"An OpenAI API error occurred: {e}")
                st.error("Please check your API key, account quota, model access, and network connection.")
            except Exception as e:
                st.error(f"An unexpected error occurred while fetching AI insights: {e}")
        # --- สิ้นสุดส่วน AI ---

        # Download button (ข้อมูลดิบ Pareto)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Pareto Data (CSV)", data=csv, file_name="pvc_pareto_data.csv", mime="text/csv")
