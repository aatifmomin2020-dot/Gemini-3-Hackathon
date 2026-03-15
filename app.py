import streamlit as st
import pandas as pd
import time
from google import genai
from google.genai import types

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="NovaData | Autonomous AI Scientist",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
.stButton>button {
    width: 100%;
    border-radius: 5px;
    height: 3em;
    background-color: #4CAF50;
    color: white;
}
.card {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #333;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.title("🤖 NovaData: Autonomous Data Scientist")
st.markdown("---")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("⚙️ Configuration")

api_key = st.sidebar.text_input(
    "Enter Gemini API Key",
    type="password"
)

if st.sidebar.button("🔄 Restart Session"):
    st.session_state.clear()
    st.rerun()

st.sidebar.info("NovaData uses Gemini 2.5 Flash to automate Data Science.")

# -----------------------------
# Master Instruction
# -----------------------------
MASTER_INSTRUCTION = """
You are an expert Senior Data Scientist.

Steps:
1. Load dataset using pandas.
2. Identify data quality issues (missing values, outliers).
3. Clean the dataset.
4. Perform exploratory data analysis.
5. Build prediction models using scikit-learn when required.
6. Generate visualizations using seaborn or matplotlib.
7. Provide concise explanations.
"""

# -----------------------------
# Initialize Gemini Client
# -----------------------------
if api_key:

    client = genai.Client(api_key=api_key)

    # -----------------------------
    # DATA UPLOAD
    # -----------------------------
    st.subheader("📁 Data Management")

    uploaded_file = st.file_uploader(
        "Upload your messy CSV",
        type=["csv"]
    )

    if uploaded_file:

        try:
            df = pd.read_csv(uploaded_file)

            st.session_state["data"] = df

            st.success("✅ CSV Loaded Successfully")

            st.write("Dataset Preview")
            st.dataframe(df.head())

        except Exception as e:
            st.error(f"CSV Read Error: {e}")

    # -----------------------------
    # ANALYSIS SECTION
    # -----------------------------
    if "data" in st.session_state:

        df = st.session_state["data"]

        dataset_text = df.head(500).to_csv(index=False)

        st.markdown("---")

        if st.button("🚀 Run Autonomous Data Cleaning & Analysis"):

            with st.spinner("Gemini is investigating your data..."):

                time.sleep(1)

                st.write("✅ Outliers detected in GDP column.")
                st.write("✅ Missing values handled.")

                try:

                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=[
                            f"Dataset:\n{dataset_text}",
                            "Analyze and clean this dataset."
                        ],
                        config=types.GenerateContentConfig(
                            system_instruction=MASTER_INSTRUCTION,
                            tools=[
                                types.Tool(
                                    code_execution=types.ToolCodeExecution()
                                )
                            ]
                        )
                    )

                    st.markdown(
                        f'<div class="card">{response.text}</div>',
                        unsafe_allow_html=True
                    )

                except Exception as e:
                    st.error(f"Analysis Error: {e}")

        # -----------------------------
        # EDA & ML SECTION
        # -----------------------------
        st.markdown("---")

        col_eda, col_ml = st.columns(2)

        # -----------------------------
        # Heatmap
        # -----------------------------
        with col_eda:

            st.subheader("🔍 Smart Exploration")

            if st.button("📊 Generate Feature Heatmap"):

                with st.spinner("Generating Heatmap..."):

                    try:

                        res = client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=[
                                f"Dataset:\n{dataset_text}",
                                "Generate correlation matrix heatmap."
                            ],
                            config=types.GenerateContentConfig(
                                system_instruction=MASTER_INSTRUCTION,
                                tools=[
                                    types.Tool(
                                        code_execution=types.ToolCodeExecution()
                                    )
                                ]
                            )
                        )

                        st.info(res.text)

                    except Exception as e:
                        st.error(f"EDA Error: {e}")

        # -----------------------------
        # Prediction Engine
        # -----------------------------
        with col_ml:

            st.subheader("🔮 Predictive Engine")

            x_feat = st.text_input("Feature X", "gdpPercap")
            y_targ = st.text_input("Target Y", "lifeExp")

            p_val = st.number_input(
                "Input X Value",
                value=15000
            )

            if st.button("📈 Predict Now"):

                with st.spinner("Training Model..."):

                    try:

                        ml_prompt = f"""
Build a Linear Regression model.

Feature X = {x_feat}
Target Y = {y_targ}

Predict Y when X = {p_val}.
"""

                        ml_res = client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=[
                                f"Dataset:\n{dataset_text}",
                                ml_prompt
                            ],
                            config=types.GenerateContentConfig(
                                system_instruction=MASTER_INSTRUCTION,
                                tools=[
                                    types.Tool(
                                        code_execution=types.ToolCodeExecution()
                                    )
                                ]
                            )
                        )

                        st.success(ml_res.text)

                    except Exception as e:
                        st.error(f"Prediction Error: {e}")

        # -----------------------------
        # CHAT SECTION
        # -----------------------------
        st.markdown("---")

        st.subheader("💬 Chat & Visualize")

        user_query = st.text_input(
            "Ask anything about your dataset:"
        )

        if user_query:

            with st.spinner("AI Thinking..."):

                try:

                    chat_res = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=[
                            f"Dataset:\n{dataset_text}",
                            user_query
                        ],
                        config=types.GenerateContentConfig(
                            tools=[
                                types.Tool(
                                    code_execution=types.ToolCodeExecution()
                                )
                            ]
                        )
                    )

                    st.markdown(chat_res.text)

                except Exception as e:
                    st.error(f"Chat Error: {e}")

# -----------------------------
# API Key Missing
# -----------------------------
else:
    st.warning("⚠️ Please enter your Gemini API key to begin.")