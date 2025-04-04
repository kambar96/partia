import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np

st.set_page_config(page_title="Partia", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&display=swap');

    html, body, [class*="css"], .stTextInput, .stSelectbox, .stMarkdown, .stDataFrame, .stTable, .stTooltip {
        font-family: 'Montserrat', sans-serif !important;
    }

    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: 'Montserrat', sans-serif !important;
        color: #5e17eb !important;
    }

    .stProgress > div > div > div > div {
        background-color: #5e17eb;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.title("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    st.success("✅ File uploaded!")

    try:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview:")
        st.dataframe(df.head())

        st.write("### Column Mapping")
        gender_column = st.selectbox("Select gender column:", ["None"] + list(df.columns))
        observer_column = st.selectbox("Select observer column:", ["None"] + list(df.columns))

        st.write(f"✅ You selected gender column: `{gender_column}`")
        st.write(f"✅ You selected observer column: `{observer_column}`")

    except Exception as e:
        st.error(f"❌ Failed to read file: {e}")


st.title("Partia")
st.write("Analyze your dataset for potential biases in sampling, historical trends, and more.")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # --- Column Mapping ---
    st.subheader("Step 1: Column Mapping")

    if "gender_column" not in st.session_state:
        st.session_state.gender_column = "None"
    if "observer_column" not in st.session_state:
        st.session_state.observer_column = "None"
    if "allow_observer_analysis" not in st.session_state:
        st.session_state.allow_observer_analysis = True

    gender_column = st.selectbox("Select the gender column (required):", ["None"] + list(df.columns), index=(["None"] + list(df.columns)).index(st.session_state.gender_column))
    observer_column = st.selectbox("Select the observer column (optional):", ["None"] + list(df.columns), index=(["None"] + list(df.columns)).index(st.session_state.observer_column))

    # Save selections
    st.session_state.gender_column = gender_column
    st.session_state.observer_column = observer_column

    # Validate gender column
    if gender_column == "None":
        st.error("🚫 This data is not sex-disaggregated. Please update your file with the required data and upload again.")
        st.stop()

    # Validate observer column (optional)
    if observer_column == "None" and "observer_decision" not in st.session_state:
        st.warning("⚠️ Your data is missing observer information.")
        proceed_without_observer = st.radio("Do you want to continue the analysis without observer bias?", ["Yes", "No"], key="observer_decision")
        if proceed_without_observer == "No":
            st.error("🚫 Please update your file with the required data and upload again.")
            st.stop()
        elif proceed_without_observer == "Yes":
            st.session_state.allow_observer_analysis = False
    elif observer_column != "None":
        st.session_state.allow_observer_analysis = True

    # --- Analysis Functions ---
    def detect_sampling_bias(df):
        df[gender_column] = df[gender_column].astype(str).str.lower()
        male_count = df[df[gender_column] == 'male'].shape[0]
        female_count = df[df[gender_column] == 'female'].shape[0]
        total_count = male_count + female_count
        if total_count == 0:
            return {'male': 0, 'female': 0}
        male_percent = (male_count / total_count) * 100
        female_percent = (female_count / total_count) * 100
        return {'male': male_percent, 'female': female_percent}

    def detect_historical_bias(df, reference_distribution):
        current_distribution = df[gender_column].astype(str).str.lower().value_counts(normalize=True) * 100
        deviation = (current_distribution - pd.Series(reference_distribution)).abs()
        return deviation.to_dict()

    def detect_proxy_bias(df):
        le = LabelEncoder()
        df[gender_column] = le.fit_transform(df[gender_column].astype(str).str.lower())
        proxy_columns = [col for col in df.columns if col != gender_column]
        for col in proxy_columns:
            if df[col].dtype == "object":
                df[col] = le.fit_transform(df[col].astype(str))
        correlations = df[proxy_columns].corrwith(df[gender_column])
        return correlations.to_dict()

    def detect_observer_bias(df):
        observer_groups = df.groupby(observer_column)["label"].value_counts(normalize=True)
        inconsistencies = observer_groups.groupby(level=0).std()
        return inconsistencies.mean()

    def detect_default_male_bias(df):
        df[gender_column] = df[gender_column].astype(str).str.lower()
        default_males = df[df[gender_column] == 'male'].shape[0]
        return {"Default Male Count": default_males}

    def get_sampling_score(result):
        male = result.get('male', 0)
        female = result.get('female', 0)
        total = male + female
        if total == 0:
            return 1
        male_ratio = male / total
        imbalance = abs(male_ratio - 0.5) * 2
        return max(1, round(10 * (1 - imbalance), 2))

    def get_historical_score(current_distribution, reference_distribution):
        male = current_distribution.get('male', 0)
        female = current_distribution.get('female', 0)
        total = male + female
        if total == 0:
            return 1
        male_ratio = male / total
        ref_male_ratio = reference_distribution['male'] / 100
        imbalance = abs(male_ratio - ref_male_ratio) * 2
        return max(1, round(10 * (1 - imbalance), 2))

    def get_proxy_score(result):
        if not result:
            return 1
        max_corr = max(abs(v) for v in result.values())
        return max(1, round(10 * (1 - min(max_corr, 1)), 2))

    def get_observer_score(value):
        if value == "N/A":
            return 10
        return max(1, round(10 * (1 - min(value, 1)), 2))

    def get_default_male_score(result):
        return max(1, 10 - int(result.get("Default Male Count", 0) // 10))

    def draw_barometer(score):
        fig, ax = plt.subplots(figsize=(4, 0.5))
        ax.barh([0], [10], color='lightgray', height=0.2)
        color = 'red' if score <= 3 else 'orange' if score <= 7 else 'green'
        ax.barh([0], [score], color=color, height=0.2)
        ax.set_xlim(0, 10)
        ax.set_yticks([])
        ax.set_xticks(np.arange(1, 11, 1))
        ax.set_xticklabels([str(i) for i in range(1, 11)])
        ax.set_title(f"Score: {score}", fontsize=10, pad=10)
        return fig

    # --- Run Analysis ---
    reference_distribution = {"male": 50, "female": 50}
    sampling_result = detect_sampling_bias(df)
    historical_result = detect_historical_bias(df, reference_distribution)
    proxy_result = detect_proxy_bias(df)
    observer_result = detect_observer_bias(df) if st.session_state.allow_observer_analysis and "label" in df.columns else "N/A"
    default_male_result = detect_default_male_bias(df)

    report = {
        "📊 Sampling Bias": {
            "explanation": "This measures whether one gender is overrepresented compared to others.",
            "result": sampling_result,
            "score": get_sampling_score(sampling_result),
            "interpretation": f"Your gender split is: Male: {sampling_result.get('male', 0)}%, Female: {sampling_result.get('female', 0)}%. Your data indicates that your sample is biased towards {'men' if sampling_result.get('male', 0) > sampling_result.get('female', 0) else 'women'}."
        },
        "📜 Historical Bias": {
            "explanation": "This compares your current dataset to historical data to highlight past inequalities.",
            "result": historical_result,
            "score": get_historical_score(sampling_result, reference_distribution),
            "interpretation": f"Historical reference: Male({reference_distribution['male']}%), Female({reference_distribution['female']}%). Current: Male({sampling_result.get('male', 0)}%), Female({sampling_result.get('female', 0)}%)."
        },
        "🔗 Proxy Bias": {
            "explanation": "This checks if other variables are strongly correlated with gender, indicating indirect discrimination.",
            "result": proxy_result,
            "score": get_proxy_score(proxy_result),
            "interpretation": "High correlation with gender suggests potential proxy bias." if any(abs(v) > 0.5 for v in proxy_result.values()) else "Minimal proxy bias detected."
        },
        "👀 Observer Bias": {
            "explanation": "This identifies inconsistencies in how different people categorize data.",
            "result": observer_result,
            "score": get_observer_score(observer_result),
            "interpretation": "No observer bias detected." if observer_result == "N/A" else "Observer bias may be present in your data."
        },
        "🙹 Default Male Bias": {
            "explanation": "This checks if 'male' is used as the default gender category.",
            "result": default_male_result,
            "score": get_default_male_score(default_male_result),
            "interpretation": "No 'default male' bias detected." if default_male_result["Default Male Count"] == 0 else "Your dataset defaults to 'male' in some cases."
        }
    }

    # --- Display Report ---
    st.subheader("Bias Report")
    for bias_type, data in report.items():
        with st.container():
            col1, col2, col3 = st.columns([2, 3, 1])

            with col1:
                st.markdown(f"### {bias_type}")
                st.markdown(f"*{data['explanation']}*")
                st.markdown("**Interpretation:**")
                st.info(data["interpretation"])

            with col2:
                if bias_type == "🔗 Proxy Bias":
                    proxy_df = pd.DataFrame(data["result"].items(), columns=["Variable", "Correlation"]).round(2)
                    st.markdown("**Correlation with gender:**")
                    st.table(proxy_df)
                elif bias_type == "🙹 Default Male Bias":
                    st.markdown(f"**Male defaults detected:** {data['result'].get('Default Male Count', 0)}")
                else:
                    st.markdown("**Raw Results:**")
                    st.write(data["result"])

            with col3:
                st.markdown("**Bias Score**")
                st.pyplot(draw_barometer(data["score"]))

            st.markdown("<hr style='margin: 2rem 0;'>", unsafe_allow_html=True)

else:
    st.info("📂 Please upload a CSV file to analyze.")
