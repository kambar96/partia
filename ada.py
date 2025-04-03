import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np

st.set_page_config(page_title="Partia", layout="wide")

# Custom Styling
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {background-color: #f8f9fa;}
        [data-testid="stAppViewContainer"] {background-color: #ffffff;}
        [data-testid="stHeader"] {display: none;}
        .block-container {padding-top: 2rem;}
    </style>
    """, unsafe_allow_html=True
)

# Sidebar
st.sidebar.image("https://raw.githubusercontent.com/kambar96/partia/main/Partia_landscape_image_template.png", use_container_width=True)
st.sidebar.title("Upload Data")
uploaded_file = st.sidebar.file_uploader("", type=["csv"])

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

# Function to draw barometer
def draw_barometer(score):
    fig, ax = plt.subplots(figsize=(7, 2))
    
    ax.barh([0], [10], color='lightgray', height=0.2)
    
    color = 'red' if score <= 3 else 'orange' if score <= 7 else 'green'
    
    ax.barh([0], [score], color=color, height=0.2)
    
    ax.set_xlim(0, 10)
    ax.set_yticks([])
    
    ax.text(score + 0.1, 0, f"Score: {score}", va='center', fontsize=12, color='black', fontweight='bold')
    
    ax.set_xticks(np.arange(0, 11, 1))
    ax.set_xticks(np.arange(1, 11, 1))
    ax.set_xticklabels([str(i) for i in range(1, 11)])
    
    return fig

# Streamlit UI
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    reference_distribution = {"male": 50, "female": 50}  # Example benchmark
    results = {
        "Sampling Bias": detect_sampling_bias(df),
        "Historical Bias": detect_historical_bias(df, "gender", reference_distribution),
        "Proxy Bias": detect_proxy_bias(df, "gender", [col for col in df.columns if col != "gender"]),
    }

    st.subheader("Bias Report")

    # Sampling Bias Section
    with st.expander("📊 Sampling Bias", expanded=True):
        st.metric(label="Male Percentage", value=f"{results['Sampling Bias'].get('male', 0):.2f}%")
        st.metric(label="Female Percentage", value=f"{results['Sampling Bias'].get('female', 0):.2f}%")
        st.pyplot(draw_barometer(7))

    # Historical Bias Section
    with st.expander("📜 Historical Bias", expanded=True):
        st.pyplot(draw_barometer(6))

    # Proxy Bias Section
    with st.expander("🔗 Proxy Bias", expanded=False):
        proxy_result = {k: round(v, 2) for k, v in results["Proxy Bias"].items()}
        st.table(pd.DataFrame(proxy_result.items(), columns=["Variable", "Correlation"]))
        st.pyplot(draw_barometer(5))
else:
    st.info("📂 Please upload a CSV file to analyze.")
