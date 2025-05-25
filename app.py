import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import openai
import json

st.set_page_config(layout="wide")

openai.api_key = st.secrets["openai_api_key"]

st.title("PVC Pipe Production Problem Analyzer & AI Action Plan")
st.markdown("Enter manufacturing problems. The AI expert in PVC pipe production will analyze, show a Pareto chart, and provide a detailed action plan table with short and long-term solutions.")

num_problems = st.number_input("Number of problems:", min_value=1, max_value=10, value=3)

problem_names = []
problem_counts = []

problem_options = [
    "Uneven wall thickness", "Rough surface (internal/external)", "Cracks or longitudinal splits",
    "Pipe warping or bending (ovality)", "Color inconsistency / streaks", "Bubbles, voids, or blisters",
    "Underfilled / short pipe", "Burning / degradation marks", "Poor gelation / unmelted particles",
    "Contamination (black specks)", "Die lines / flow marks", "Excessive flash or burrs",
    "Low impact strength", "Dimensional instability", "Bell end deformation / defects",
    "Other (please specify)"
]

for i in range(num_problems):
    col1, col2 = st.columns([3, 1])
    with col1:
        selected = st.selectbox(f"Problem {i+1}", problem_options, key=f"prob_{i}", index=i % len(problem_options))
        if selected == "Other (please specify)":
            name = st.text_input(f"Specify problem {i+1}", key=f"custom_prob_{i}")
        else:
            name = selected
    with col2:
        count = st.number_input(f"Count {i+1}", min_value=0, step=1, key=f"pcount_{i}")
    problem_names.append(name)
    problem_counts.append(count)

if st.button("🔍 Analyze PVC Problems & Generate Detailed Action Plan"):
    df_raw = pd.DataFrame({"Problem": problem_names, "Count": problem_counts})
    df = df_raw.groupby("Problem", as_index=False).sum()
    df = df[df["Count"] > 0].sort_values(by="Count", ascending=False).reset_index(drop=True)
    
    if df.empty:
        st.warning("Please enter at least one problem with a count greater than 0.")
    else:
        df["Cumulative %"] = df["Count"].cumsum() / df["Count"].sum() * 100

        st.subheader("Pareto Chart of PVC Pipe Production Problems")
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.bar(df["Problem"], df["Count"], color="deepskyblue")
        ax2 = ax1.twinx()
        ax2.plot(df["Problem"], df["Cumulative %"], color="crimson", marker='o', linewidth=2)
        ax1.set_ylabel("Frequency Count", fontweight='bold')
        ax2.set_ylabel("Cumulative Percentage (%)", fontweight='bold')
        ax1.set_xticklabels(df["Problem"], rotation=45, ha='right')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        ax2.axhline(80, color='gray', linestyle='--', label='80% Vital Few Line')
        ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=1)
        fig.tight_layout()
        st.pyplot(fig)

        num_top_problems_to_analyze = min(len(df), 3)
        top_problems_df = df.head(num_top_problems_to_analyze)
        
        problem_list_for_ai = f"Analyze the following top {num_top_problems_to_analyze} PVC pipe production problems and provide a detailed action plan according to the specified JSON format:\n"
        for idx, row in top_problems_df.iterrows():
            problem_list_for_ai += f"{idx+1}. Problem: \"{row['Problem']}\" (occurred {row['Count']} times)\n"

        st.subheader("⚙️ AI Expert Analysis & Action Plan for PVC Pipe Production")
        with st.spinner("Consulting PVC Pipe Production AI Expert and drafting detailed action plan... This may take some time."):
            
            system_prompt_content = (
                "You are an AI assistant acting as an Expert in PVC pipe production with over 20 years of experience. "
                "Your task is to analyze the provided manufacturing problems specific to PVC pipe production. "
                "Present your entire analysis ONLY as a single, valid JSON string. This JSON string should represent a list of objects. "
                "Each object in the list corresponds to one analyzed problem and must contain the following keys (ensure keys are exactly as written):\n"
                "1. 'Problem' (string): The name of the PVC pipe production problem as provided.\n"
                "2. 'Potential_Cause_PVC_Specific' (string): Specific potential causes. For multiple points or lines, use '\\n' for newlines within the string.\n"
                "3. 'Suggested_Solution_PVC_Specific' (string): Actionable solutions. For multiple points or lines, use '\\n' for newlines. For bullet points, start each point with '- ' followed by the text, all within a single string separated by '\\n'.\n"
                "4. 'Responsible_Department' (string): The primary department accountable (choose from: Production, Maintenance, Quality Assurance (QA), Process Engineering, Formulation/R&D, Raw Material Procurement, Tooling, Management).\n"
                "5. 'Short_Term_Action_Plan' (string): Concrete actions for 1-3 months. Use '\\n' for newlines or for separating bullet points (e.g., '- Action 1\\n- Action 2').\n"
                "6. 'Long_Term_Action_Plan' (string): Strategic actions for 6-12 months. Use '\\n' for newlines or for separating bullet points.\n\n"
                "Example for a field with bullet points: \"- Point 1 related to the solution.\\n- Point 2 another detail.\\n- Yet another check to perform.\"\n"
                "Ensure the output is ONLY the JSON string, without any surrounding text, markdown, or explanations. Do not use markdown table formatting."
            )
            
            ai_response_content = ""
            try:
                response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt_content},
                        {"role": "user", "content": problem_list_for_ai}
                    ],
                    temperature=0.1
                )
                ai_response_content = response.choices[0].message.content
                
                if ai_response_content.strip().startswith("```json"):
                    json_str = ai_response_content.strip()[7:-3].strip()
                elif ai_response_content.strip().startswith("```"):
                    json_str = ai_response_content.strip()[3:-3].strip()
                else:
                    json_str = ai_response_content.strip()

                if not json_str:
                    st.error("AI returned an empty response. Cannot parse JSON.")
                else:
                    parsed_data = json.loads(json_str)
                    
                    if isinstance(parsed_data, list):
                        df_analysis = pd.DataFrame(parsed_data)
                        
                        column_mapping = {
                            "Problem": "Problem",
                            "Potential_Cause_PVC_Specific": "Potential Cause (PVC Specific)",
                            "Suggested_Solution_PVC_Specific": "Suggested Solution (PVC Specific)",
                            "Responsible_Department": "Responsible Department",
                            "Short_Term_Action_Plan": "Short-Term Action/Plan (1-3 months)",
                            "Long_Term_Action_Plan": "Long-Term Action/Plan (6-12 months)"
                        }
                        
                        df_analysis.rename(columns={k: v for k, v in column_mapping.items() if k in df_analysis.columns}, inplace=True)
                        
                        desired_column_order = [
                            "Problem", "Potential Cause (PVC Specific)", "Suggested Solution (PVC Specific)",
                            "Responsible Department", "Short-Term Action/Plan (1-3 months)", 
                            "Long-Term Action/Plan (6-12 months)"
                        ]
                        display_columns = [col for col in desired_column_order if col in df_analysis.columns]
                        
                        # ✨ แก้ไขปัญหาการแสดงผลตาราง ✨
                        column_configs = {}
                        text_heavy_columns = [
                            "Potential Cause (PVC Specific)", 
                            "Suggested Solution (PVC Specific)", 
                            "Short-Term Action/Plan (1-3 months)", 
                            "Long-Term Action/Plan (6-12 months)"
                        ]
                        
                        for col_name in display_columns:
                            if col_name in text_heavy_columns:
                                column_configs[col_name] = st.column_config.TextColumn(
                                    label=col_name,
                                    width="large",
                                    help=f"Click to expand {col_name}"
                                )
                            elif col_name == "Problem":
                                column_configs[col_name] = st.column_config.TextColumn(
                                    label=col_name,
                                    width="medium"
                                )
                            elif col_name == "Responsible Department":
                                column_configs[col_name] = st.column_config.TextColumn(
                                    label=col_name,
                                    width="small"
                                )
                        
                        # แสดงตารางแบบใหม่ที่อ่านง่ายขึ้น
                        st.dataframe(
                            df_analysis[display_columns], 
                            use_container_width=True,
                            column_config=column_configs,
                            height=800  # เพิ่มความสูงของตาราง
                        )
                        
                        # เพิ่มส่วนแสดงรายละเอียดแบบ Card สำหรับอ่านง่ายขึ้น
                        st.subheader("📋 Detailed Analysis (Expanded View)")
                        
                        for idx, row in df_analysis.iterrows():
                            with st.expander(f"🔍 {row['Problem']}", expanded=False):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**🎯 Potential Causes:**")
                                    # แปลง \n เป็น bullet points
                                    causes = row['Potential Cause (PVC Specific)'].replace('\\n', '\n')
                                    if not causes.startswith('- '):
                                        causes = '- ' + causes.replace('\n', '\n- ')
                                    st.markdown(causes)
                                    
                                    st.markdown("**💡 Suggested Solutions:**")
                                    solutions = row['Suggested Solution (PVC Specific)'].replace('\\n', '\n')
                                    st.markdown(solutions)
                                    
                                    st.markdown(f"**👥 Responsible Department:** {row['Responsible Department']}")
                                
                                with col2:
                                    st.markdown("**⏰ Short-Term Actions (1-3 months):**")
                                    short_term = row['Short-Term Action/Plan (1-3 months)'].replace('\\n', '\n')
                                    st.markdown(short_term)
                                    
                                    st.markdown("**🎯 Long-Term Actions (6-12 months):**")
                                    long_term = row['Long-Term Action/Plan (6-12 months)'].replace('\\n', '\n')
                                    st.markdown(long_term)

                    else:
                        st.error("AI did not return data in the expected list format.")
                        st.text("Raw AI response:")
                        st.text(ai_response_content)

            except json.JSONDecodeError as e:
                st.error(f"Error decoding AI response as JSON: {e}")
                st.text("Raw AI response that failed to parse:")
                st.text(ai_response_content)
            except openai.APIError as e:
                st.error(f"An OpenAI API error occurred: {e}")
                st.error("Please check your API key, account quota, model access, and network connection.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                if ai_response_content:
                    st.text("Raw AI response during unexpected error:")
                    st.text(ai_response_content)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Pareto Data (CSV)", data=csv, file_name="pvc_pareto_data.csv", mime="text/csv")
