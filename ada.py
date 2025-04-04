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

# Bias Detection Functions
def detect_sampling_bias(df):
    male_count = df[df['gender'] == 'male'].shape[0]
    female_count = df[df['gender'] == 'female'].shape[0]
    
    total_count = male_count + female_count
    male_percent = (male_count / total_count) * 100 if total_count else 0
    female_percent = (female_count / total_count) * 100 if total_count else 0
    
    return {'male': male_percent, 'female': female_percent}

def detect_historical_bias(df, gender_column, reference_distribution):
    current_distribution = df[gender_column].value_counts(normalize=True) * 100
    deviation = (current_distribution - pd.Series(reference_distribution)).abs()
    return deviation.to_dict()

def detect_proxy_bias(df, gender_column, proxy_columns):
    df_copy = df.copy()
    le = LabelEncoder()
    df_copy[gender_column] = le.fit_transform(df_copy[gender_column])
    
    for col in proxy_columns:
        if df_copy[col].dtype == "object":
            df_copy[col] = le.fit_transform(df_copy[col])
    
    correlations = df_copy[proxy_columns].corrwith(df_copy[gender_column])
    return correlations.to_dict()

def detect_observer_bias(df, label_column, observer_column):
    observer_groups = df.groupby(observer_column)[label_column].value_counts(normalize=True)
    inconsistencies = observer_groups.groupby(level=0).std()
    return inconsistencies.to_dict()

def detect_default_male_bias(df, gender_column, default_value):
    default_males = df[df[gender_column] == default_value].shape[0]
    return {"Default Male Count": default_males}

# Scoring Functions
def get_sampling_score(sampling_result):
    male_percent = sampling_result.get('male', 0)
    female_percent = sampling_result.get('female', 0)
    bias_difference = abs(male_percent - female_percent)
    return max(1, 10 - (bias_difference // 5))

def get_historical_score(current_distribution, reference_distribution):
    male_deviation = abs(current_distribution.get('male', 0) - reference_distribution['male'])
    female_deviation = abs(current_distribution.get('female', 0) - reference_distribution['female'])
    deviation = max(male_deviation, female_deviation)
    return max(1, 10 - (deviation // 10))

def get_proxy_score(proxy_result):
    max_correlation = max(abs(v) for v in proxy_result.values())
    return max(1, 10 - (max_correlation * 10))

def get_observer_score(observer_result):
    if observer_result == "N/A":
        return 10
    elif isinstance(observer_result, dict):
        avg_std = np.mean(list(observer_result.values()))
        return max(1, 10 - (avg_std * 10))
    return 1

def get_default_male_score(default_male_result):
    default_male_count = default_male_result.get("Default Male Count", 0)
    return max(1, 10 - (default_male_count // 10))

# Bias Report Generator
def generate_bias_report(df, reference_distribution):
    sampling_result = detect_sampling_bias(df)
    historical_result = detect_historical_bias(df, "gender", reference_distribution)
    proxy_result = detect_proxy_bias(df, "gender", [col for col in df.columns if col != "gender"])
    observer_result = detect_observer_bias(df, "label", "observer") if "label" in df.columns and "observer" in df.columns else "N/A"
    default_male_result = detect_default_male_bias(df, "gender", "male")

    report = {
        "Sampling Bias": {
            "explanation": "This measures whether one gender is overrepresented compared to others.",
            "result": sampling_result,
            "score": get_sampling_score(sampling_result),
            "interpretation": f"Your gender split is: Male: {sampling_result.get('male', 0):.1f}%, Female: {sampling_result.get('female', 0):.1f}%. "
                             f"Your data indicates that your sample is biased towards {'men' if sampling_result.get('male', 0) > sampling_result.get('female', 0) else 'women'}. "
                             f"Try increasing the sample size with more {'female' if sampling_result.get('male', 0) > sampling_result.get('female', 0) else 'male'} participants to improve the gender representation."
        },
        "Historical Bias": {
            "explanation": "This compares your current dataset to historical data to highlight past inequalities.",
            "result": historical_result,
            "score": get_historical_score(sampling_result, reference_distribution),
            "interpretation": f"Historical data: Male({reference_distribution['male']}), Female({reference_distribution['female']}). "
                             f"Uploaded data: Male({sampling_result.get('male', 0):.1f}), Female({sampling_result.get('female', 0):.1f}). "
                             f"Your data indicates a stronger bias towards {'male' if sampling_result.get('male', 0) > reference_distribution['male'] else 'female'} participants when compared with the historical data."
        },
        "Proxy Bias": {
            "explanation": "This checks if other variables are strongly correlated with gender, indicating indirect discrimination. Scores operate between -1 and 1, and 0 shows no gender correlation.",
            "result": proxy_result,
            "score": get_proxy_score(proxy_result),
            "interpretation": "Your variables indicate high levels of correlation with gender, suggesting inherent biases that you should consider in your research results." if any(abs(v) > 0.5 for v in proxy_result.values()) else "Your variables show low correlation with gender, indicating minimal proxy bias."
        },
        "Observer Bias": {
            "explanation": "This identifies inconsistencies in how different people categorize data, which can introduce bias.",
            "result": observer_result,
            "score": get_observer_score(observer_result),
            "interpretation": "There are no indications of observer bias in your data. Good job!" if observer_result == "N/A" else "Observer bias detected: Variations exist in how different observers categorize data, which may require review."
        },
        "Default Male Bias": {
            "explanation": "This checks if 'male' is used as the default gender category, which can lead to bias.",
            "result": default_male_result,
            "score": get_default_male_score(default_male_result),
            "interpretation": "No 'default male' bias detected. The gender distribution in your dataset is balanced." if default_male_result["Default Male Count"] == 0 else "Your dataset defaults to 'male' in some cases. Consider ensuring neutral representation."
        }
    }
    return report

# Barometer Drawing
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

    if 'gender' not in df.columns:
        st.error("The dataset must contain a 'gender' column to perform bias analysis.")
    else:
        reference_distribution = {"male": 50, "female": 50}
        results = generate_bias_report(df, reference_distribution)

        st.subheader("Bias Report")

        with st.expander("📊 Sampling Bias", expanded=True):
            st.write(results["Sampling Bias"]["explanation"])
            male_percent = results["Sampling Bias"]["result"].get("male", 0)
            female_percent = results["Sampling Bias"]["result"].get("female", 0)
            st.metric(label="Male Percentage", value=f"{male_percent:.2f}%")
            st.metric(label="Female Percentage", value=f"{female_percent:.2f}%")
            st.write(f"Score: {results['Sampling Bias']['score']}")
            st.write(results["Sampling Bias"]["interpretation"])
            st.pyplot(draw_barometer(results['Sampling Bias']['score']))

        with st.expander("📜 Historical Bias", expanded=True):
            st.write(results["Historical Bias"]["explanation"])
            st.write(f"Score: {results['Historical Bias']['score']}")
            st.write(results["Historical Bias"]["interpretation"])
            st.pyplot(draw_barometer(results['Historical Bias']['score']))

        with st.expander("🔗 Proxy Bias", expanded=False):
            st.write(results["Proxy Bias"]["explanation"])
            proxy_result = {k: round(v, 2) for k, v in results["Proxy Bias"]["result"].items()}
            st.dataframe(pd.DataFrame(proxy_result.items(), columns=["Variable", "Correlation"]))
            st.write(f"Score: {results['Proxy Bias']['score']}")
            st.write(results["Proxy Bias"]["interpretation"])
            st.pyplot(draw_barometer(results['Proxy Bias']['score']))

        with st.expander("👀 Observer Bias", expanded=False):
            st.write(results["Observer Bias"]["explanation"])
            st.write(f"Score: {results['Observer Bias']['score']}")
            st.write(results["Observer Bias"]["interpretation"])
            st.pyplot(draw_barometer(results['Observer Bias']['score']))

        with st.expander("🚹 Default Male Bias", expanded=False):
            st.write(results["Default Male Bias"]["explanation"])
            st.write(f"Score: {results['Default Male Bias']['score']}")
            st.write(results["Default Male Bias"]["interpretation"])
            st.pyplot(draw_barometer(results['Default Male Bias']['score']))
else:
    st.info("📂 Please upload a CSV file to analyze.")
