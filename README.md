# 🤖 NovaData: Autonomous AI Data Scientist

**NovaData** is an agentic data science platform built for the Gemini 3 Hackathon. It leverages the reasoning capabilities of **Gemini 2.5 Flash** and its **Native Code Execution** feature to automate the end-to-end data science lifecycle.

---

## 🚀 Key Features

* **Autonomous Data Auditing**: Automatically identifies statistical properties like bimodal distributions, skewness, and missing values.
* **Intelligent Cleaning**: Writes and executes Python code to normalize outliers using the Interquartile Range (IQR) method and handles data type casting.
* **Predictive Engine**: Dynamically trains **Scikit-Learn** Linear Regression models in a sandboxed environment to forecast future trends.
* **Natural Language Visualization**: Translates simple chat queries into complex, filtered, and aggregated charts using Pandas and Streamlit.
* **Smart Exploration**: Generates correlation matrices to help users identify meaningful feature relationships before modeling.

---

## 🛠️ Tech Stack

* **Core AI**: [Google Gemini 2.5 Flash API](https://aistudio.google.com/)
* **Frontend**: [Streamlit](https://streamlit.io/)
* **Data Processing**: Pandas, NumPy
* **Machine Learning**: Scikit-Learn
* **Visualization**: Matplotlib, Seaborn

---

## 🏗️ Architecture

NovaData operates on a **ReAct (Reasoning and Acting) Pattern**:
1.  **Input**: The user uploads a CSV and provides a natural language goal.
2.  **Reasoning**: Gemini 2.5 Flash analyzes the metadata and determines the required Python libraries and logic.
3.  **Action**: The agent utilizes **Native Code Execution** to run code in a secure sandbox.
4.  **Observation**: The output (cleaned data, model coefficients, or chart data) is returned to the dashboard.

---

## 💻 Getting Started

### Prerequisites
- Python 3.9+
- A Google Gemini API Key

### Installation

1. **Clone the repository**:
   ```bash
  git clone https://github.com/aatifmomin2020-dot/Gemini-3-Hackathon.git
cd Gemini-3-Hackathon
