import pandas as  pd
df = pd.read_csv("Airline Dataset Updated - v2.csv")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Airlines Dataset EDA", layout="wide")

st.title("✈️ Airlines Dataset Exploratory Data Analysis (EDA)")
st.write("Analyze trends, delays, and passenger statistics interactively")

# ---------------------------
# Load Dataset
# ---------------------------
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded successfully!")
else:
    st.info("Using default dataset...")
    file_path = "Airline Dataset Updated - v2.csv"   # <-- Change here if needed
    df = pd.read_csv(file_path)

# ---------------------------
# Dataset Overview
# ---------------------------
st.subheader("📌 Dataset Preview")
st.write(df.head())

col1, col2, col3 = st.columns(3)
col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Missing Values", df.isnull().sum().sum())

st.subheader("📊 Dataset Information")
st.write(df.describe())

st.subheader("🧹 Missing Values")
st.write(df.isnull().sum())

# ---------------------------
# Visualizations
# ---------------------------
st.header("📈 Visual Analysis")

# Gender Count
if "Gender" in df.columns:
    st.subheader("👨‍🦰 Gender Distribution")
    gender_count = df["Gender"].value_counts()

    fig, ax = plt.subplots()
    ax.bar(gender_count.index, gender_count.values)
    ax.set_xlabel("Gender")
    ax.set_ylabel("Count")
    ax.set_title("Gender Count")
    st.pyplot(fig)

# Age Distribution
if "Age" in df.columns:
    st.subheader("🎂 Age Distribution")
    fig, ax = plt.subplots()
    ax.hist(df["Age"], bins=20)
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    ax.set_title("Age Distribution of Passengers")
    st.pyplot(fig)

# Class Distribution
if "Class" in df.columns:
    st.subheader(" Seat Class Distribution")
    class_count = df["Class"].value_counts()

    fig, ax = plt.subplots()
    ax.pie(class_count.values, labels=class_count.index, autopct="%1.1f%%")
    ax.set_title("Passenger Class Share")
    st.pyplot(fig)

# Flight Day Count
if "Day of Week" in df.columns:
    st.subheader("📅 Flights per Day of Week")
    daily = df["Day of Week"].value_counts()

    fig, ax = plt.subplots()
    ax.bar(daily.index, daily.values)
    ax.set_xlabel("Day")
    ax.set_ylabel("Flights")
    st.pyplot(fig)

# Month-wise Flights
if "Month" in df.columns:
    st.subheader("📆 Flights per Month")
    df["Month"] = pd.Categorical(df["Month"],
                                 categories=["Jan","Feb","Mar","Apr","May","Jun",
                                             "Jul","Aug","Sep","Oct","Nov","Dec"],
                                 ordered=True)
    monthly = df["Month"].value_counts().sort_index()

    fig, ax = plt.subplots()
    ax.plot(monthly.index, monthly.values, marker="o")
    ax.set_xlabel("Month")
    ax.set_ylabel("Flights")
    st.pyplot(fig)

st.success("EDA Completed Successfully 🎯")
