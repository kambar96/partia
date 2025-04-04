import streamlit as st
import pandas as pd
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

st.title("Partia")
st.write("Analyze your dataset for potential biases in sampling, historical trends, and more.")

uploaded_file = st.sidebar.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        # --- Smart column auto-detection ---
        def find_column(possible_names):
            for col in df.columns:
                if col.strip().lower() in [name.lower() for name in possible_names]:
                    return col
            return "None"

        default_gender_column = find_column(["gender", "sex", "biological sex"])
        default_observer_column = find_column(["observer", "researcher", "owner"])

        if "gender_column" not in st.session_state:
            st.session_state.gender_column = default_gender_column
        if "observer_column" not in st.session_state:
            st.session_state.observer_column = default_observer_column

        # Column Mapping UI
        st.subheader("Step 1: Column Mapping")
        gender_column = st.selectbox("Select gender column (required):", ["None"] + list(df.columns),
                                     index=(["None"] + list(df.columns)).index(st.session_state.gender_column))
        observer_column = st.selectbox("Select observer column (optional):", ["None"] + list(df.columns),
                                       index=(["None"] + list(df.columns)).index(st.session_state.observer_column))

        st.session_state.gender_column = gender_column
        st.session_state.observer_column = observer_column

        if gender_column == "None":
            st.error("🚫 This data is not sex-disaggregated. Please update your file with the required data and upload again.")
            st.stop()

        if observer_column == "None":
            choice = st.radio("⚠️ No observer column selected. Continue without observer bias analysis?", ["Yes", "No"])
            if choice == "No":
                st.error("🚫 Please update your file with the required observer data and upload again.")
                st.stop()
            allow_observer_analysis = False
        else:
            allow_observer_analysis = True

        # Bias detection functions
        def detect_sampling_bias(df):
            df[gender_column] = df[gender_column].astype(str).str.lower()
            value_counts = df[gender_column].value_counts(normalize=True) * 100
            return value_counts.to_dict()

        def detect_historical_bias(df, reference_dist):
            current = df[gender_column].astype(str).str.lower().value_counts(normalize=True) * 100
            deviation = (current - pd.Series(reference_dist)).abs()
            return deviation.to_dict()

        def detect_proxy_bias(df):
            le = LabelEncoder()
            df[gender_column] = le.fit_transform(df[gender_column].astype(str).str.lower())
            proxy_cols = [col for col in df.columns if col != gender_column]
            for col in proxy_cols:
                if df[col].dtype == "object":
                    df[col] = le.fit_transform(df[col].astype(str))
            return df[proxy_cols].corrwith(df[gender_column]).to_dict()

        def detect_observer_bias(df):
            observer_groups = df.groupby(observer_column)["label"].value_counts(normalize=True)
            inconsistencies = observer_groups.groupby(level=0).std()
            return inconsistencies.mean()

        def detect_default_male_bias(df):
            df[gender_column] = df[gender_column].astype(str).str.lower()
            return {"Default Male Count": df[df[gender_column] == 'male'].shape[0]}

        # Scoring functions
        def get_sampling_score(distribution):
            if not distribution:
                return 1
            ratios = np.array(list(distribution.values())) / 100
            ideal = 1 / len(ratios)
            imbalance = np.sum(np.abs(ratios - ideal))
            score = max(1, round(10 * (1 - imbalance), 2))
            return score

        def get_historical_score(current, reference):
            male = current.get('male', 0)
            female = current.get('female', 0)
            total = male + female
            if total == 0: return 1
            imbalance = abs((male / total) - (reference["male"] / 100)) * 2
            return max(1, round(10 * (1 - imbalance), 2))

        def get_proxy_score(res):
            if not res: return 1
            max_corr = max(abs(v) for v in res.values())
            return max(1, round(10 * (1 - min(max_corr, 1)), 2))

        def get_observer_score(value):
            if value == "N/A": return 10
            return max(1, round(10 * (1 - min(value, 1)), 2))

        def get_default_male_score(res):
            return max(1, 10 - int(res.get("Default Male Count", 0) // 10))

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

        def generate_text_report(report):
            lines = ["Partia Bias Detection Report", "=" * 30, ""]
            for bias_name, data in report.items():
                lines.append(f"{bias_name}")
                lines.append("-" * len(bias_name))
                lines.append(f"Explanation: {data['explanation']}")
                lines.append(f"Raw Results: {data['result']}")
                lines.append(f"Bias Score: {data['score']}")
                lines.append(f"Interpretation: {data['interpretation']}")
                lines.append("")  # Spacer line
            return "\n".join(lines)

        # Run analysis
        reference_distribution = {"male": 50, "female": 50}
        sampling_result = detect_sampling_bias(df)
        historical_result = detect_historical_bias(df, reference_distribution)
        proxy_result = detect_proxy_bias(df)
        observer_result = detect_observer_bias(df) if allow_observer_analysis and "label" in df.columns else "N/A"
        default_male_result = detect_default_male_bias(df)

        # Build report
        report = {}

        distribution_text = ', '.join([f"{k.capitalize()}: {v:.1f}%" for k, v in sampling_result.items()])
        dominant_group = max(sampling_result, key=sampling_result.get)

        report["📊 Sampling Bias"] = {
            "explanation": "This measures whether any gender group is overrepresented compared to others.",
            "result": sampling_result,
            "score": get_sampling_score(sampling_result),
            "interpretation": f"Your gender distribution is: {distribution_text}. "
                              f"The dataset is skewed towards {dominant_group}. Consider including more participants from underrepresented gender categories."
        }

        report["📜 Historical Bias"] = {
            "explanation": "This compares your current dataset to historical data to highlight past inequalities.",
            "result": historical_result,
            "score": get_historical_score(sampling_result, reference_distribution),
            "interpretation": f"Historical reference: Male({reference_distribution['male']}%), Female({reference_distribution['female']}%). "
                              f"Current data: Male({sampling_result.get('male', 0):.1f}%), Female({sampling_result.get('female', 0):.1f}%). "
                              f"Your data shows a stronger representation of {'men' if sampling_result.get('male', 0) > reference_distribution['male'] else 'women'} compared to historical benchmarks."
        }

        proxy_bias_flag = any(abs(v) > 0.5 for v in proxy_result.values())
        report["🔗 Proxy Bias"] = {
            "explanation": "This checks if other variables are strongly correlated with gender, indicating indirect discrimination.",
            "result": proxy_result,
            "score": get_proxy_score(proxy_result),
            "interpretation": "Your variables show high correlation with gender, suggesting possible proxy bias." if proxy_bias_flag
                              else "Low correlation with gender suggests minimal proxy bias in your dataset."
        }

        if allow_observer_analysis and "label" in df.columns:
            report["👀 Observer Bias"] = {
                "explanation": "This identifies inconsistencies in how different people categorize data, which can introduce bias.",
                "result": observer_result,
                "score": get_observer_score(observer_result),
                "interpretation": "Your dataset shows variation between how different observers label entries. This may indicate observer bias that requires further review."
            }

        report["🙹 Default Male Bias"] = {
            "explanation": "This checks if 'male' is used as the default gender category, which can lead to bias.",
            "result": default_male_result,
            "score": get_default_male_score(default_male_result),
            "interpretation": "No 'default male' bias detected." if default_male_result["Default Male Count"] == 0
                              else "Your dataset defaults to 'male' in some cases. Consider reviewing default values for neutrality."
        }

        # Display report
        st.subheader("Bias Report")
        
        # Generate downloadable report text
        text_report = generate_text_report(report)

        # Export button
        st.download_button(
        label="📥 Download Bias Report (.txt)",
        data=text_report,
        file_name="partia_bias_report.txt",
        mime="text/plain"
        )



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

    except Exception as e:
        st.error(f"❌ File upload or processing failed: {e}")

else:
    st.info("📂 Please upload a CSV file to analyze.")
