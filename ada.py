
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
        # Wrapped logic starts here
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

        allow_observer_analysis = (
            observer_column != "None"
            and observer_column in df.columns
            and "label" in df.columns
        )

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
                        lines.append(f"- {d[1]}")
                lines.append("")
            return "\n".join(lines)

        sampling_result = detect_sampling_bias(df)
        
        # Sidebar threshold controls for proxy bias
        st.sidebar.subheader("🔧 Proxy Bias Thresholds")
        strong_threshold = st.sidebar.slider("Strong correlation if >", 0.3, 1.0, 0.5, 0.05)
        moderate_threshold = st.sidebar.slider("Moderate correlation if >", 0.1, strong_threshold, 0.3, 0.05)

        proxy_result = detect_proxy_bias(df)
        observer_result = detect_observer_bias(df) if allow_observer_analysis else "N/A"

        report = {}

        # Sampling Bias
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

        # Proxy Bias
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
            if abs_corr > strong_threshold:
                interp = f"The variable '{var}' is strongly correlated with gender (correlation: {corr:.2f}). This may indicate proxy bias."
                color = "red"
            elif moderate_threshold <= abs_corr <= strong_threshold:
                interp = f"The variable '{var}' is moderately correlated with gender (correlation: {corr:.2f}). Consider reviewing its potential influence."
                color = "orange"
            else:
                interp = f"The variable '{var}' shows little or no correlation with gender (correlation: {corr:.2f}), indicating minimal bias."
                color = "green"
            proxy_details.append((var, interp, color))

        report["🔗 Proxy Bias"] = {
            "explanation": "This checks if other variables are strongly correlated with gender, indicating indirect discrimination.",
            "result": proxy_result,
            "score": get_proxy_score(proxy_result),
            "interpretation": proxy_summary,
            "details": proxy_details
        }

        if allow_observer_analysis:
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
                        
                        proxy_df = pd.DataFrame(data["result"].items(), columns=["Variable", "Correlation"]).round(2)
                        proxy_df["AbsCorrelation"] = proxy_df["Correlation"].abs()
                        proxy_df = proxy_df.sort_values(by="AbsCorrelation", ascending=False)

                        st.markdown("**Correlation with gender by variable:**")
                        for _, row in proxy_df.iterrows():
                            var = row["Variable"]
                            corr = row["Correlation"]
                            abs_corr = abs(corr)

                            if abs_corr > strong_threshold:
                                color = "red"
                            elif moderate_threshold <= abs_corr <= strong_threshold:
                                color = "orange"
                            else:
                                color = "green"

                            st.markdown(f"<span style='color:{color}'><strong>{var}</strong> – {corr:.2f}</span>", unsafe_allow_html=True)

                            
