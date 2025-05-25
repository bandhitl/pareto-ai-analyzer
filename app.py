import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import openai

# Set OpenAI API key from secrets
openai.api_key = st.secrets["openai_api_key"]

st.title("Pareto Problem Analyzer with AI Action Plan") #ปรับชื่อ Title เล็กน้อย
st.markdown("Enter manufacturing problems and their frequency. The system will show a Pareto chart and suggest insights as an action plan table.")

num_problems = st.number_input("Number of problems:", min_value=1, max_value=15, value=5)

problem_names = []
problem_counts = []

problem_options = [
    "Uneven wall thickness",
    "Rough surface",
    "Cracks or longitudinal splits",
    "Pipe warping or bending",
    "Color inconsistency",
    "Bubbles or voids",
    "Underfilled / short pipe",
    "Ring marks / scoring",
    "Bell end deformation",
    "Temperature instability",
    "Other (please specify)"
]

for i in range(num_problems):
    col1, col2 = st.columns([3, 1])
    with col1:
        selected = st.selectbox(f"Problem {i+1}", problem_options, key=f"prob_{i}")
        if selected == "Other (please specify)":
            name = st.text_input(f"Specify problem {i+1}", key=f"custom_prob_{i}")
        else:
            name = selected
    with col2:
        count = st.number_input(f"Count {i+1}", min_value=0, step=1, key=f"pcount_{i}")
    problem_names.append(name)
    problem_counts.append(count)

if st.button("🔍 Analyze Problems & Generate Action Plan"): # ปรับชื่อปุ่ม
    df_raw = pd.DataFrame({"Problem": problem_names, "Count": problem_counts})
    df = df_raw.groupby("Problem", as_index=False).sum()
    df = df[df["Count"] > 0].sort_values(by="Count", ascending=False).reset_index(drop=True)
    df["Cumulative %"] = df["Count"].cumsum() / df["Count"].sum() * 100

    st.subheader("Pareto Chart")
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(df["Problem"], df["Count"], color="skyblue")
    ax2 = ax1.twinx()
    ax2.plot(df["Problem"], df["Cumulative %"], color="red", marker='o')
    ax1.set_ylabel("Count")
    ax2.set_ylabel("Cumulative %")
    ax1.set_xticklabels(df["Problem"], rotation=45, ha='right')
    ax2.axhline(80, color='gray', linestyle='--', label='80%')
    ax2.legend(loc="lower right")
    st.pyplot(fig)

    # --- ส่วนของการเรียก AI และแสดงผลเป็นตาราง ---
    top_problems = df.head(3) # หรือจะเลือกจำนวนปัญหาอื่นก็ได้ตามต้องการ
    
    # สร้างรายการปัญหาสำหรับส่งให้ AI
    problem_list_for_ai = "Analyze the following top manufacturing problems and provide an action plan:\n"
    for idx, row in top_problems.iterrows():
        problem_list_for_ai += f"{idx+1}. Problem: {row['Problem']} (occurred {row['Count']} times)\n"

    # ปรับปรุง Subheader
    st.subheader("💡 AI Insight & Action Plan Table")
    with st.spinner("🤖 Generating AI analysis and action plan table... This might take a moment."):
        # กำหนด System Prompt ใหม่เพื่อให้ AI สร้างตาราง
        system_prompt_content = (
            "You are a manufacturing quality expert. Your task is to analyze the provided manufacturing problems. "
            "For each problem, identify potential causes, suggest actionable solutions, and assign a primary responsible department "
            "for implementing those solutions. Present your entire analysis ONLY as a markdown table with the following columns: "
            "'Problem', 'Potential Cause', 'Suggested Solution', 'Responsible Department'. "
            "The possible departments are: Production Team, Maintenance Team, Quality Assurance (QA) Team, "
            "Engineering Team, Raw Materials/Procurement Team, Logistics Team, R&D Team, Management. "
            "For the 'Problem' column, list the problem name as provided. "
            "Ensure your response contains only the markdown table and no other introductory or concluding text."
        )
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4",  # GPT-4 เหมาะสำหรับงานที่ต้องการความเข้าใจคำสั่งซับซ้อนและการสร้างตาราง
                messages=[
                    {"role": "system", "content": system_prompt_content},
                    {"role": "user", "content": problem_list_for_ai} # ส่งรายการปัญหาให้ AI
                ],
                temperature=0.2 # ลด temperature เพื่อให้ได้ผลลัพธ์ที่ตรงประเด็นและเป็นโครงสร้าง
            )
            ai_response_table_markdown = response.choices[0].message.content
            
            # แสดงผลตาราง Markdown ที่ AI สร้างขึ้น
            st.markdown(ai_response_table_markdown)

        except openai.APIError as e:
            st.error(f"An OpenAI API error occurred: {e}")
            st.error("Please check your API key, quota, and network connection.")
        except Exception as e:
            st.error(f"An unexpected error occurred while fetching AI insights: {e}")
    # --- สิ้นสุดส่วน AI ---

    # Download button (ข้อมูลดิบ Pareto)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Pareto Data (CSV)", data=csv, file_name="pareto_data.csv", mime="text/csv")
