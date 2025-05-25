import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import openai

# Set OpenAI API key from secrets
openai.api_key = st.secrets["openai_api_key"]

st.title("Pareto Problem Analyzer with AI")
st.markdown("Enter manufacturing problems and their frequency. The system will show a Pareto chart and suggest insights.")

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

if st.button("🔍 Analyze Problems"):
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

    # Prepare top 3 for analysis
    top_problems = df.head(3)
    text_prompt = "Analyze the top 3 problems and suggest potential causes and solutions based on manufacturing best practices:\n"
    for idx, row in top_problems.iterrows():
        text_prompt += f"{idx+1}. {row['Problem']} - {row['Count']} times\n"

    st.subheader("💡 AI Insight")
    with st.spinner("Thinking like an industrial engineer..."):
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a manufacturing quality expert."},
                {"role": "user", "content": text_prompt}
            ],
            temperature=0.3
        )
        result_text = response.choices[0].message.content
        st.markdown(result_text)

        # --- ✨ ส่วนที่เพิ่มเข้ามา: สร้างและแสดงรูปภาพจาก AI Insight ✨ ---
        st.subheader("🖼️ Visual Aid from AI")
        with st.spinner("🎨 Generating illustrative image... This may take a moment."):
            try:
                # สร้าง prompt สำหรับ DALL-E จาก result_text
                # คุณสามารถปรับปรุง prompt นี้เพื่อให้ได้ผลลัพธ์ที่ดีขึ้น
                # ใช้ส่วนหนึ่งของ result_text เพื่อไม่ให้ prompt ยาวเกินไป
                image_prompt_detail = result_text[:400] # จำกัดความยาวเพื่อประสิทธิภาพ
                
                if len(image_prompt_detail.strip()) < 20: # ตรวจสอบว่าข้อความไม่สั้นเกินไป
                    st.warning("AI insight text is too short to generate a meaningful image.")
                else:
                    # สร้าง prompt ที่กระชับและสื่อความหมายสำหรับ DALL-E
                    image_prompt = (
                        f"Create a clear and simple visual illustration for manufacturing workers. "
                        f"The image should explain solutions or key concepts based on the following advice: '{image_prompt_detail}...'. "
                        f"Focus on visual clarity and ease of understanding for a factory setting."
                    )

                    image_response = openai.images.generate(
                        model="dall-e-3",  # หรือ "dall-e-2" (DALL-E 3 ให้ผลลัพธ์ที่ดีกว่า แต่ DALL-E 2 อาจมีค่าใช้จ่ายต่ำกว่า)
                        prompt=image_prompt,
                        n=1, # จำนวนรูปภาพที่ต้องการ
                        size="1024x1024",  # ขนาดรูปภาพ, DALL-E 3 รองรับ "1024x1024", "1792x1024", "1024x1792"
                        response_format="url" # ขอ URL ของรูปภาพ (เป็น default สำหรับ DALL-E 3)
                        # quality="standard" # or "hd" for DALL-E 3
                    )
                    generated_image_url = image_response.data[0].url
                    st.image(generated_image_url, caption="AI Generated Illustration based on Insights")
            except openai.APIError as e:
                st.error(f"An OpenAI API error occurred while generating the image: {e}")
            except Exception as e:
                st.error(f"Could not generate image due to an unexpected error: {e}")
        # --- ✨ สิ้นสุดส่วนที่เพิ่มเข้ามา ✨ ---

    # Download button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Pareto Data", data=csv, file_name="pareto_data.csv", mime="text/csv")
