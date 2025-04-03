import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np

st.set_page_config(page_title="Partia - Powering Data Equality", layout="wide")

# Custom Styles
st.markdown(
    """
    <style>
    .css-18e3th9 {
        padding-top: 2rem;
    }
    .stApp {
        background-color: #F9F9F9;
    }
    .header {
        background-color: #5F16D0;
        padding: 15px;
        text-align: left;
        color: white;
        font-size: 20px;
        font-weight: bold;
    }
    .logo-container {
        text-align: center;
        margin-top: 50px;
    }
    .logo-text {
        font-size: 48px;
        font-weight: bold;
        color: black;
    }
    .logo-text span {
        color: #5F16D0;
    }
    .tagline {
        font-size: 20px;
        color: #666;
    }
    .upload-box {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 30px;
    }
    .stFileUploader {
        border: 2px solid #5F16D0 !important;
        border-radius: 10px;
    }
    .upload-btn {
        background-color: #5F16D0;
        color: white;
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown('<div class="header">partia</div>', unsafe_allow_html=True)

# Logo and Tagline
st.markdown(
    """
    <div class="logo-container">
        <div class="logo-text"><span>p</span>artia</div>
        <div class="tagline">Powering Data Equality</div>
    </div>
    """,
    unsafe_allow_html=True
)

# File Upload Box
uploaded_file = st.file_uploader("Browse files", type=["csv"], key="file_uploader")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
else:
    st.markdown('<div class="upload-btn">Upload your CSV</div>', unsafe_allow_html=True)
