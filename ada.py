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
    gender_column = detect_gender_column(df)
    if gender_column is None:
        return {'male': 0, 'female': 0}

    df[gender_column] = df[gender_column].astype(str).str.lower()
    male_count = df[df[gender_column] == 'male'].shape[0]
    female_count = df[df[gender_column] == 'female'].shape[0]

    total_count = male_count + female_count
    if total_count == 0:
        return {'male': 0, 'female': 0}

    male_percent = (male_count / total_count) * 100
    female_percent = (female_count / total_count) * 100

    return {'male': male_percent, 'female': female_percent}

# Historical Bias Detection
def detect_historical_bias(df, gender_column, reference_distribution):
    if gender_column not in df.columns:
        return {"male": 0, "female": 0}

    current_distribution = df[gender_column].astype(str).str.lower().value_counts(normalize=True) * 100
    deviation = (current_distribution - pd.Series(reference_distribution)).abs()
    return deviation.to_dict()

# Proxy Bias Detection
def detect_proxy_bias(df, gender_column, proxy_columns):
    le = LabelEncoder()
    df[gender_column] = le.fit_transform(df[gender_column].astype(str).str.lower())

    for col in proxy_columns:
        if df[col].dtype == "object":
            df[col] = le.fit_transform(df[col].astype(str))

    correlations = df[proxy_columns].corrwith(df[gender_column])
    return correlations.to_dict()

# Observer Bias Detection
def detect_observer_bias(df, label_column, observer_column):
    observer_groups = df.groupby(observer_column)[label_column].value_counts(normalize=True)
    inconsistencies = observer_groups.groupby(level=0).std()
    return inconsistencies.mean()

# Default Male Bias Detection
def detect_default_male_bias(df, gender_column, default_value):
    df[gender_column] = df[gender_column].astype(str).str.lower()
    default_males = df[df[gender_column] == default_value].shape[0]
    return {"Default Male Count": default_males}

# Gender column detection
def detect_gender_column(df):
    for col in df.columns:
        if col.strip().lower() == "gender":
            return col
    return None

# Generate Bias Report
def generate_bias_report(df, reference_distribution):
    gender_column = detect_gender_column(df)
    if gender_column is None:
        st.error("Gender column not found in dataset.")
        return {}

    sampling_result = detect_sampling_bias(df)
    historical_result = detect_historical_bias(df, gender_column, reference_distribution)
    proxy_result = detect_proxy_bias(df, gender_column, [col for col in df.columns if col != gender_column])
    observer_result = detect_observer_bias(df, "label", "observer") if "label" in df.columns and "observer" in df.columns else "N/A"
    default_male_result = detect_default_male_bias(df, gender_column, "male")

    report = {
        "📊 Sampling Bias": {
            "explanation": "This measures whether one gender is overrepresented compared to others.",
            "result": sampling_result,
            "score": get_sampling_score(sampling_result),
            "interpretation": f"Your gender split is: Male: {sampling_result.get('male', 0)}%, Female: {sampling_result.get('female', 0)}%. Your data indicates that your sample is biased towards {'men' if sampling_result.get('male', 0) > sampling_result.get('female', 0) else 'women'}. Try increasing the sample size with more {'female' if sampling_result.get('male', 0) > sampling_result.get('female', 0) else 'male'} participants to improve the gender representation."
        },
        "📜 Historical Bias": {
            "explanation": "This compares your current dataset to historical data to highlight past inequalities.",
            "result": historical_result,
            "score": get_historical_score(sampling_result, reference_distribution),
            "interpretation": f"Historical data: Male({reference_distribution['male']}), Female({reference_distribution['female']}). Uploaded data: Male({sampling_result.get('male', 0)}), Female({sampling_result.get('female', 0)}). Your data indicates a stronger bias towards {'male' if sampling_result.get('male', 0) > reference_distribution['male'] else 'female'} participants when compared with the historical data. Consider examining the figures to rectify this."
        },
        "🔗 Proxy Bias": {
            "explanation": "This checks if other variables are strongly correlated with gender, indicating indirect discrimination. Scores operate between -1 and 1, and 0 shows no gender correlation.",
            "result": proxy_result,
            "score": get_proxy_score(proxy_result),
            "interpretation": f"Your variables indicate high levels of correlation with gender, suggesting inherent biases that you should consider in your research results." if any(abs(v) > 0.5 for v in proxy_result.values()) else "Your variables show low correlation with gender, indicating minimal proxy bias."
        },
        "👀 Observer Bias": {
            "explanation": "This identifies inconsistencies in how different people categorize data, which can introduce bias.",
            "result": observer_result,
            "score": get_observer_score(observer_result),
            "interpretation": "There are no indications of observer bias in your data. Good job!" if observer_result == "N/A" else "Observer bias detected: Variations exist in how different observers categorize data, which may require review."
        },
        "🙹 Default Male Bias": {
            "explanation": "This checks if 'male' is used as the default gender category, which can lead to bias.",
            "result": default_male_result,
            "score": get_default_male_score(default_male_result),
            "interpretation": "No 'default male' bias detected. The gender distribution in your dataset is balanced." if default_male_result["Default Male Count"] == 0 else "Your dataset defaults to 'male' in some cases. Consider ensuring neutral representation."
        }
    }
    return report

def get_sampling_score(sampling_result):
    male_percent = sampling_result.get('male', 0)
    female_percent = sampling_result.get('female', 0)

    # Total % might not equal 100 if other genders exist, so use just male and female
    total = male_percent + female_percent
    if total == 0:
        return 1  # No data to assess

    # Normalize percentages
    male_ratio = male_percent / total
    female_ratio = female_percent / total

    # Calculate imbalance as deviation from 0.5 (perfect balance)
    imbalance = abs(male_ratio - 0.5) * 2  # Ranges from 0 to 1
    score = max(1, round(10 * (1 - imbalance), 2))
    return score

def get_historical_score(current_distribution, reference_distribution):
    male_deviation = abs(current_distribution.get('male', 0) - reference_distribution['male'])
    female_deviation = abs(current_distribution.get('female', 0) - reference_distribution['female'])
    deviation = max(male_deviation, female_deviation)
    return max(1, 10 - int(deviation // 10))

def get_proxy_score(proxy_result):
    max_correlation = max(abs(v) for v in proxy_result.values())
    return max(1, 10 - int(max_correlation * 10))

def get_observer_score(observer_result):
    if observer_result == "N/A":
        return 10
    else:
        return max(1, 10 - int(observer_result * 10))

def get_default_male_score(default_male_result):
    default_male_count = default_male_result.get("Default Male Count", 0)
    return max(1, 10 - int(default_male_count // 10))

# Draw barometer
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

# Streamlit UI
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    reference_distribution = {"male": 50, "female": 50}
    results = generate_bias_report(df, reference_distribution)

    st.subheader("Bias Report")
    for bias_type, data in results.items():
        with st.expander(f"{bias_type}", expanded=(bias_type == "📊 Sampling Bias")):
            st.markdown(f"**What this measures:** {data['explanation']}")
            st.markdown("---")

            if bias_type == "🔗 Proxy Bias":
                proxy_df = pd.DataFrame(data["result"].items(), columns=["Variable", "Correlation"]).round(2)
                st.write("**Correlation with gender:**")
                st.table(proxy_df)
            elif bias_type == "🙹 Default Male Bias":
                st.write(f"**Male defaults detected:** {data['result'].get('Default Male Count', 0)}")
            else:
                st.write("**Raw Results:**")
                st.write(data["result"])

            st.markdown("**Interpretation:**")
            st.info(data["interpretation"])

            st.markdown("**Bias Score:**")
            st.pyplot(draw_barometer(data["score"]))
else:
    st.info("📂 Please upload a CSV file to analyze.")
