import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np

st.set_page_config(page_title="Partia", layout="wide")

# Custom CSS
st.markdown("""
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
""", unsafe_allow_html=True)

st.title("Partia")
st.write("Analyze your dataset for potential sampling, proxy, and observer bias.")

uploaded_file = st.sidebar.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

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

        st.subheader("Step 1: Column Mapping")
        gender_column = st.selectbox("Select gender column (required):", ["None"] + list(df.columns),
                                     index=(["None"] + list(df.columns)).index(st.session_state.gender_column))
        observer_column = st.selectbox("Select observer column (optional):", ["None"] + list(df.columns),
                                       index=(["None"] + list(df.columns)).index(st.session_state.observer_column))

        st.session_state.gender_column = gender_column
        st.session_state.observer_column = observer_column

        if gender_column == "None":
            st.error("🚫 This data is not sex-disaggregated. Please update your file and upload again.")
            st.stop()

        if observer_column == "None":
            choice = st.radio("⚠️ No observer column selected. Continue without observer bias analysis?", ["Yes", "No"])
            if choice == "No":
                st.error("🚫 Please update your file with the required observer data and upload again.")
                st.stop()
            allow_observer_analysis = False
        else:
            allow_observer_analysis = True

        def detect_sampling_bias(df):
            df[gender_column] = df[gender_column].astype(str).str.lower()
            value_counts = df[gender_column].value_counts(normalize=True) * 100
            return value_counts.to_dict()

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

        def get_sampling_score(distribution):
            if not distribution: return 1
            ratios = np.array(list(distribution.values())) / 100
            ideal = 1 / len(ratios)
            imbalance = np.sum(np.abs(ratios - ideal))
            return max(1, round(10 * (1 - imbalance), 2))

        def get_proxy_score(res):
            if not res: return 1
            max_corr = max(abs(v) for v in res.values())
            return max(1, round(10 * (1 - min(max_corr, 1)), 2))

        def get_observer_score(value):
            if value == "N/A": return 10
            return max(1, round(10 * (1 - min(value, 1)), 2))

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
                if "details" in data:
                    for d in data["details"]:
                        lines.append(f"- {d}")
                lines.append("")
            return "\n".join(lines)

        # Run analysis
        sampling_result = detect_sampling_bias(df)
        proxy_result = detect_proxy_bias(df)
        observer_result = detect_observer_bias(df) if allow_observer_analysis and "label" in df.columns else "N/A"

        # Build report
        report = {}

        # --- Sampling Bias Interpretation ---
        male = sampling_result.get('male', 0)
        female = sampling_result.get('female', 0)
        diff = abs(male - female)
        if diff <= 1:
            sampling_interp = "Your dataset shows near-equal representation of men and women, which supports more inclusive and generalisable insights."
        elif 1 < diff <= 24:
            group = "men" if male > female else "women"
            sampling_interp = f"Your dataset contains slightly more {group} (difference of {diff:.1f}%). Consider increasing participation from the underrepresented group."
        else:
            group = "men" if male > female else "women"
            sampling_interp = f"Your dataset contains significantly more {group} (difference of {diff:.1f}%). This may introduce strong bias and limit representativeness."

        report["📊 Sampling Bias"] = {
            "explanation": "This measures whether any gender group is overrepresented compared to others.",
            "result": sampling_result,
            "score": get_sampling_score(sampling_result),
            "interpretation": sampling_interp
        }

        # --- Proxy Bias Interpretation ---
        max_corr = max(abs(v) for v in proxy_result.values()) if proxy_result else 0
        if max_corr > 0.75:
            proxy_summary = "Your data contains variables strongly correlated with gender, suggesting possible proxy bias."
        elif 0.56 <= max_corr <= 0.75:
            proxy_summary = "Some variables show moderate correlation with gender. Consider reviewing their role in your analysis."
        else:
            proxy_summary = "No variables show strong correlation with gender, indicating low risk of proxy bias."

        proxy_details = []
        for var, corr in proxy_result.items():
            abs_corr = abs(corr)
            if abs_corr > 0.75:
                interp = f"The variable '{var}' is strongly correlated with gender (correlation: {corr:.2f}). This may indicate proxy bias."
            elif 0.56 <= abs_corr <= 0.75:
                interp = f"The variable '{var}' is moderately correlated with gender (correlation: {corr:.2f}). Consider reviewing its potential influence."
            elif abs_corr < 0.45:
                interp = f"The variable '{var}' shows little or no correlation with gender (correlation: {corr:.2f}), indicating minimal bias."
            else:
                interp = f"The variable '{var}' has a neutral correlation with gender (correlation: {corr:.2f})."
            proxy_details.append(interp)

        report["🔗 Proxy Bias"] = {
            "explanation": "This checks if other variables are strongly correlated with gender, indicating indirect discrimination.",
            "result": proxy_result,
            "score": get_proxy_score(proxy_result),
            "interpretation": proxy_summary,
            "details": proxy_details
        }

        # --- Observer Bias Interpretation ---
        if allow_observer_analysis and "label" in df.columns:
            if observer_result > 0.75:
                observer_interp = "There is a high level of inconsistency between observers. This suggests strong observer bias that could distort results."
            elif 0.56 <= observer_result <= 0.75:
                observer_interp = "Moderate inconsistency detected in observer responses. While not extreme, this could still affect data quality."
            else:
                observer_interp = "Observer responses are consistent, suggesting minimal observer bias."

            report["👀 Observer Bias"] = {
                "explanation": "This identifies inconsistencies in how different people categorize data, which can introduce bias.",
                "result": round(observer_result, 3),
                "score": get_observer_score(observer_result),
                "interpretation": observer_interp
            }

        # Show report
        st.subheader("Bias Report")
        st.download_button(
            label="📥 Download Bias Report (.txt)",
            data=generate_text_report(report),
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
                        st.markdown("**Correlation with gender by variable:**")
                        st.table(proxy_df)

                for i, line in enumerate(data["details"]):
                    variable = proxy_df.iloc[i]["Variable"]
                    corr = abs(proxy_df.iloc[i]["Correlation"])
                    color = "green"
                if corr > 0.75:
                    color = "red"
                elif 0.56 <= corr <= 0.75:
                    color = "orange"

                with st.expander(f"{variable}"):
                    st.markdown(f"<span style='color:{color}'>{line}</span>", unsafe_allow_html=True)

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
