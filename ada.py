import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np

st.set_page_config(page_title="Bias Detection Tool", layout="wide")

# Sidebar
st.sidebar.title("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

st.title("Bias Detection Tool")
st.write("Analyze your dataset for potential biases in sampling, historical trends, and more.")

# Sampling Bias Detection
def detect_sampling_bias(df):
    male_count = df[df['gender'] == 'male'].shape[0]
    female_count = df[df['gender'] == 'female'].shape[0]
    
    total_count = male_count + female_count
    if total_count == 0:
        return {'male': 0, 'female': 0}
    
    male_percent = (male_count / total_count) * 100
    female_percent = (female_count / total_count) * 100
    
    return {'male': male_percent, 'female': female_percent}

# Historical Bias Detection
def detect_historical_bias(df, gender_column, reference_distribution):
    current_distribution = df[gender_column].value_counts(normalize=True) * 100
    deviation = (current_distribution - pd.Series(reference_distribution)).abs()
    return deviation.to_dict()

# Proxy Bias Detection
def detect_proxy_bias(df, gender_column, proxy_columns):
    le = LabelEncoder()
    df[gender_column] = le.fit_transform(df[gender_column])
    
    for col in proxy_columns:
        if df[col].dtype == "object":
            df[col] = le.fit_transform(df[col])

    correlations = df[proxy_columns].corrwith(df[gender_column])
    return correlations.to_dict()

# Observer Bias Detection
def detect_observer_bias(df, label_column, observer_column):
    observer_groups = df.groupby(observer_column)[label_column].value_counts(normalize=True)
    inconsistencies = observer_groups.groupby(level=0).std()
    return inconsistencies.mean() if not inconsistencies.empty else 0

# Default Male Bias Detection
def detect_default_male_bias(df, gender_column, default_value):
    default_males = df[df[gender_column] == default_value].shape[0]
    return {"Default Male Count": default_males}

# Score Calculations
def get_sampling_score(sampling_result):
    bias_difference = abs(sampling_result.get('male', 0) - sampling_result.get('female', 0))
    return max(1, 10 - (bias_difference // 5))

def get_historical_score(current_distribution, reference_distribution):
    male_deviation = abs(current_distribution.get('male', 0) - reference_distribution['male'])
    female_deviation = abs(current_distribution.get('female', 0) - reference_distribution['female'])
    deviation = max(male_deviation, female_deviation)
    return max(1, 10 - (deviation // 10))

def get_proxy_score(proxy_result):
    if not proxy_result:
        return 10
    max_corr = max(abs(v) for v in proxy_result.values())
    return max(1, 10 - (max_corr * 10))

def get_observer_score(observer_result):
    return max(1, 10 - (observer_result * 10)) if observer_result != "N/A" else 10

def get_default_male_score(default_male_result):
    default_male_count = default_male_result.get("Default Male Count", 0)
    return max(1, 10 - (default_male_count // 10))

# Barometer Visualization
def draw_barometer(score):
    fig, ax = plt.subplots(figsize=(7, 2))
    ax.barh([0], [10], color='lightgray', height=0.2)
    color = 'red' if score <= 3 else 'orange' if score <= 7 else 'green'
    ax.barh([0], [score], color=color, height=0.2)
    ax.set_xlim(0, 10)
    ax.set_yticks([])
    ax.text(score + 0.1, 0, f"Score: {score}", va='center', fontsize=12, color='black', fontweight='bold')
    ax.set_xticks(np.arange(1, 11, 1))
    ax.set_xticklabels([str(i) for i in range(1, 11)])
    return fig

# Generate Full Bias Report
def generate_bias_report(df, reference_distribution):
    sampling_result = detect_sampling_bias(df)
    historical_result = detect_historical_bias(df, "gender", reference_distribution)
    proxy_result = detect_proxy_bias(df.copy(), "gender", [col for col in df.columns if col != "gender"])
    observer_result = detect_observer_bias(df, "label", "observer") if "label" in df.columns and "observer" in df.columns else "N/A"
    default_male_result = detect_default_male_bias(df, "gender", "male")

    report = {
        "Sampling Bias": {
            "explanation": "Measures overrepresentation of gender in the sample.",
            "result": sampling_result,
            "score": get_sampling_score(sampling_result),
            "interpretation": f"Gender split is Male: {sampling_result.get('male', 0):.2f}%, Female: {sampling_result.get('female', 0):.2f}%."
        },
        "Historical Bias": {
            "explanation": "Compares current data to a historical baseline.",
            "result": historical_result,
            "score": get_historical_score(sampling_result, reference_distribution),
            "interpretation": f"Reference: M({reference_distribution['male']}%), F({reference_distribution['female']}%) | Your data: M({sampling_result.get('male', 0):.2f}%), F({sampling_result.get('female', 0):.2f}%)."
        },
        "Proxy Bias": {
            "explanation": "Looks for variables correlated with gender.",
            "result": proxy_result,
            "score": get_proxy_score(proxy_result),
            "interpretation": "Some features correlate strongly with gender. Review needed." if any(abs(v) > 0.5 for v in proxy_result.values()) else "Minimal proxy bias detected."
        },
        "Observer Bias": {
            "explanation": "Evaluates inconsistencies in labels between observers.",
            "result": observer_result,
            "score": get_observer_score(observer_result),
            "interpretation": "No observer bias detected." if observer_result == "N/A" else "Some variation in labeling between observers."
        },
        "Default Male Bias": {
            "explanation": "Checks if 'male' is used as a default gender.",
            "result": default_male_result,
            "score": get_default_male_score(default_male_result),
            "interpretation": "Some entries default to 'male'. Consider reviewing your data structure." if default_male_result["Default Male Count"] > 0 else "No default male bias detected."
        }
    }
    return report

# Streamlit UI Logic
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Standardize gender column
    if 'gender' in df.columns:
        df['gender'] = df['gender'].astype(str).str.lower().str.strip()

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    reference_distribution = {"male": 50, "female": 50}
    results = generate_bias_report(df, reference_distribution)

    st.subheader("Bias Report")

    for bias_type, data in results.items():
        with st.expander(f"{bias_type}", expanded=(bias_type == "Sampling Bias")):
            st.write(data["explanation"])
            if bias_type == "Proxy Bias":
                proxy_df = pd.DataFrame(data["result"].items(), columns=["Variable", "Correlation"]).round(2)
                st.table(proxy_df)
            elif bias_type == "Default Male Bias":
                st.write(f"Male defaults detected: {data['result'].get('Default Male Count', 0)}")
            else:
                st.write(data["result"])
            st.write(f"Score: {data['score']}")
            st.write(data["interpretation"])
            st.pyplot(draw_barometer(data["score"]))
else:
    st.info("📂 Please upload a CSV file to analyze.")
