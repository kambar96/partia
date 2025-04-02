import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

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
    
    encoded_columns = {}
    for col in proxy_columns:
        if df[col].dtype == "object":  # Convert categorical variables
            df[col] = le.fit_transform(df[col])
            encoded_columns[col] = df[col]

    correlations = df[proxy_columns].corrwith(df[gender_column])
    return correlations.to_dict()

# Observer Bias Detection
def detect_observer_bias(df, label_column, observer_column):
    observer_groups = df.groupby(observer_column)[label_column].value_counts(normalize=True)
    inconsistencies = observer_groups.groupby(level=0).std()
    return inconsistencies.to_dict()

# Default Male Bias Detection
def detect_default_male_bias(df, gender_column, default_value):
    default_males = df[df[gender_column] == default_value].shape[0]
    return {"Default Male Count": default_males}

# Generate Bias Report
def generate_bias_report(df, reference_distribution):
    sampling_result = detect_sampling_bias(df)
    historical_result = detect_historical_bias(df, "gender", reference_distribution)
    proxy_result = detect_proxy_bias(df, "gender", [col for col in df.columns if col != "gender"])
    observer_result = detect_observer_bias(df, "label", "observer") if "label" in df.columns and "observer" in df.columns else "N/A"
    default_male_result = detect_default_male_bias(df, "gender", "male")

def get_sampling_score(sampling_result):
    male_percent = sampling_result.get('male', 0)
    female_percent = sampling_result.get('female', 0)
    bias_difference = abs(male_percent - female_percent)
    
    # A smaller difference gives a higher score
    return max(1, 10 - (bias_difference // 5))  # Adjust the denominator as needed for your scale
    
def get_historical_score(current_distribution, reference_distribution):
    male_deviation = abs(current_distribution.get('male', 0) - reference_distribution['male'])
    female_deviation = abs(current_distribution.get('female', 0) - reference_distribution['female'])
    
    # The lower the deviation, the higher the score
    deviation = max(male_deviation, female_deviation)
    return max(1, 10 - (deviation // 10))  # Adjust as needed

def get_proxy_score(proxy_result):
    max_correlation = max(abs(v) for v in proxy_result.values())  # Max absolute correlation
    
    # The lower the correlation, the higher the score
    return max(1, 10 - (max_correlation * 10))  # Adjust multiplier as needed
    
def get_observer_score(observer_result):
    if observer_result == "N/A":
        return 10  # No observer bias
    else:
        return max(1, 10 - (observer_result * 10))  # Adjust if needed based on how you calculate inconsistencies

def get_default_male_score(default_male_result):
    default_male_count = default_male_result.get("Default Male Count", 0)
    
    # The higher the count, the lower the score
    return max(1, 10 - (default_male_count // 10))  # Adjust as needed
    
    report = {
       "Sampling Bias": {
            "explanation": "This measures whether one gender is overrepresented compared to others.",
            "result": sampling_result,
            "score": get_sampling_score(sampling_result),  # This is where the error occurred
            "interpretation": f"Your gender split is: Male: {sampling_result.get('male', 0)}%, Female: {sampling_result.get('female', 0)}%. Your data indicates that your sample is biased towards {'men' if sampling_result.get('male', 0) > sampling_result.get('female', 0) else 'women'}. Try increasing the sample size with more {'female' if sampling_result.get('male', 0) > sampling_result.get('female', 0) else 'male'} participants to improve the gender representation."
        },
        "Historical Bias": {
            "explanation": "This compares your current dataset to historical data to highlight past inequalities.",
            "result": historical_result,
            "score": get_historical_score(sampling_result, reference_distribution),
            "interpretation": f"Historical data: Male({reference_distribution['male']}), Female({reference_distribution['female']}). Uploaded data: Male({sampling_result.get('male', 0)}), Female({sampling_result.get('female', 0)}). Your data indicates a stronger bias towards {'male' if sampling_result.get('male', 0) > reference_distribution['male'] else 'female'} participants when compared with the historical data. Consider examining the figures to rectify this."
        },
        "Proxy Bias": {
            "explanation": "This checks if other variables are strongly correlated with gender, indicating indirect discrimination. Scores operate between -1 and 1, and 0 shows no gender correlation.",
            "result": proxy_result,
            "score": get_proxy_score(proxy_result),
            "interpretation": f"Your variables indicate high levels of correlation with gender, suggesting inherent biases that you should consider in your research results." if any(abs(v) > 0.5 for v in proxy_result.values()) else "Your variables show low correlation with gender, indicating minimal proxy bias."
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

# Streamlit UI
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    reference_distribution = {"male": 50, "female": 50}  # Example benchmark
    results = generate_bias_report(df, reference_distribution)

    st.subheader("Bias Report")

    # Sampling Bias Section
    with st.expander("📊 Sampling Bias", expanded=True):
        st.write(results["Sampling Bias"]["explanation"])
        male_percent = results["Sampling Bias"]["result"].get("male", 0)
        female_percent = results["Sampling Bias"]["result"].get("female", 0)
        st.metric(label="Male Percentage", value=f"{male_percent:.2f}%")
        st.metric(label="Female Percentage", value=f"{female_percent:.2f}%")
        st.write(results["Sampling Bias"]["interpretation"])

        # Visualization
        fig, ax = plt.subplots()
        sns.barplot(x=["Male", "Female"], y=[male_percent, female_percent], ax=ax)
        ax.set_ylabel("Percentage")
        st.pyplot(fig)

    # Historical Bias Section
    with st.expander("📜 Historical Bias", expanded=True):
        st.write(results["Historical Bias"]["explanation"])
        st.write(results["Historical Bias"]["interpretation"])

    # Proxy Bias Section
    with st.expander("🔗 Proxy Bias", expanded=False):
        st.write(results["Proxy Bias"]["explanation"])
        proxy_result = {k: round(v, 2) for k, v in results["Proxy Bias"]["result"].items()}
        st.table(pd.DataFrame(proxy_result.items(), columns=["Variable", "Correlation"]))
        st.write(results["Proxy Bias"]["interpretation"])

    # Observer Bias Section
    with st.expander("👀 Observer Bias", expanded=False):
        st.write(results["Observer Bias"]["explanation"])
        st.write(results["Observer Bias"]["interpretation"])

    # Default Male Bias Section
    with st.expander("🚹 Default Male Bias", expanded=False):
        st.write(results["Default Male Bias"]["explanation"])
        st.write(results["Default Male Bias"]["interpretation"])
else:
    st.info("📂 Please upload a CSV file to analyze.")
