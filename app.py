import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
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
    file_path = "Airline Dataset Updated - v2.csv"
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
st.write(df.describe(include='all'))

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
    ax.hist(df["Age"].dropna(), bins=20)
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

# ---------------------------
# Prediction UI + Logic (updated)
# ---------------------------
st.header("🧠 Flight Status Prediction")

# Expected model features
FEATURES = ["Gender", "Age", "Nationality", "Airport Continent", "Continents"]

# Try to load model and encoders
model = None
encoders = None
model_path = "model.pkl"
encoders_path = "encoders.pkl"

if os.path.exists(model_path) and os.path.exists(encoders_path):
    try:
        with open(model_path, "rb") as mf:
            model = pickle.load(mf)
        with open(encoders_path, "rb") as ef:
            encoders = pickle.load(ef)
        st.success("Model and encoders loaded ✅")
    except Exception as e:
        st.error(f"Error loading model/encoders: {e}")
else:
    st.warning("Model or encoders not found. Run `python train_model.py` to create them to enable predictions.")

# Mode: single or batch
mode = st.sidebar.radio("Prediction mode", ["Single", "Batch"])

# Common helpers
def opt_for(col, default_list):
    if col in df.columns:
        return sorted(df[col].dropna().unique().tolist())
    return default_list

gender_options = opt_for("Gender", ["Male", "Female"])
nationality_options = opt_for("Nationality", ["Unknown"])
airport_cont_options = opt_for("Airport Continent", ["Asia", "Europe", "North America", "South America", "Africa", "Oceania"])
continents_options = opt_for("Continents", ["Asia", "Europe", "North America", "South America", "Africa", "Oceania"])

# Safe encoding helper (handles unseen labels by extending classes_)
def safe_transform(le, val):
    try:
        if val in le.classes_:
            return int(le.transform([val])[0])
        else:
            le.classes_ = np.append(le.classes_, val)
            return int(le.transform([val])[0])
    except Exception as e:
        # fallback: return the raw value (may fail down the line)
        return val

# ---------------------------
# SINGLE prediction (with autofill)
# ---------------------------
if mode == "Single":
    st.subheader("Single-row prediction")

    # Fill from random row button (auto-populate inputs)
    if st.sidebar.button("Fill inputs from random row"):
        sample = df.sample(1).iloc[0]
        st.session_state.setdefault("gender", sample.get("Gender", gender_options[0]))
        st.session_state.setdefault("age", int(sample.get("Age", 30)))
        st.session_state.setdefault("nationality", sample.get("Nationality", nationality_options[0]))
        st.session_state.setdefault("airport_continent", sample.get("Airport Continent", airport_cont_options[0]))
        st.session_state.setdefault("continents", sample.get("Continents", continents_options[0]))
        # force rerun occurs automatically when session_state is set

    # Use session_state keys for persistence and autofill
    input_gender = st.sidebar.selectbox("Gender", gender_options, key="gender")
    input_age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=st.session_state.get("age", 30), key="age")
    input_nationality = st.sidebar.selectbox("Nationality", nationality_options, key="nationality")
    input_airport_continent = st.sidebar.selectbox("Airport Continent", airport_cont_options, key="airport_continent")
    input_continents = st.sidebar.selectbox("Continents", continents_options, key="continents")

    if st.sidebar.button("Predict Flight Status"):
        if model is None or encoders is None:
            st.error("Cannot predict: model or encoders missing. Train the model first.")
        else:
            input_row = {
                "Gender": input_gender,
                "Age": input_age,
                "Nationality": input_nationality,
                "Airport Continent": input_airport_continent,
                "Continents": input_continents
            }
            input_df = pd.DataFrame([input_row], columns=FEATURES)

            # Encode categorical columns
            for col in FEATURES:
                if col in encoders and input_df[col].dtype == object:
                    le = encoders[col]
                    input_df[col] = input_df[col].apply(lambda v: safe_transform(le, v))

            X_input = input_df[FEATURES].values
            try:
                pred = model.predict(X_input)[0]
                st.success(f"Predicted Flight Status: **{pred}** ✅")

                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_input)[0]
                    classes = model.classes_
                    prob_df = pd.DataFrame({"Class": classes, "Probability": proba})
                    prob_df = prob_df.sort_values("Probability", ascending=False).reset_index(drop=True)
                    st.subheader("Prediction probabilities")
                    st.write(prob_df)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ---------------------------
# BATCH prediction (CSV)
# ---------------------------
else:
    st.subheader("Batch prediction (upload CSV)")
    uploaded_batch = st.sidebar.file_uploader("Upload CSV for batch prediction", type=["csv"], key="batch_uploader")
    sample_rows = st.sidebar.number_input("Sample rows to generate (if no upload)", min_value=1, max_value=1000, value=5)

    # If user uploads a CSV, use that; else provide sample rows
    if uploaded_batch is not None:
        try:
            batch_df = pd.read_csv(uploaded_batch)
            st.write(f"Uploaded dataset shape: {batch_df.shape}")
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")
            batch_df = None
    else:
        st.info("No CSV uploaded — using random sample rows from the loaded dataset for demo.")
        batch_df = df.sample(min(sample_rows, len(df))).copy()

    if batch_df is not None:
        # Ensure expected columns exist, or attempt to map similarly-named columns
        missing = [c for c in FEATURES if c not in batch_df.columns]
        if missing:
            st.warning(f"Uploaded file is missing expected columns: {missing}. Rows without required fields may fail.")
        # Keep only the expected columns if present, else try to create them with defaults
        for c in FEATURES:
            if c not in batch_df.columns:
                # fill with first option or NaN
                if c == "Age":
                    batch_df[c] = 30
                else:
                    batch_df[c] = (gender_options[0] if c == "Gender" else nationality_options[0] if c == "Nationality" else airport_cont_options[0] if c == "Airport Continent" else continents_options[0])

        # Apply encoders column-wise
        encoded_df = batch_df[FEATURES].copy()
        for col in FEATURES:
            if col in encoders and encoded_df[col].dtype == object:
                le = encoders[col]
                # transform values and extend classes_ for unseen values
                def _safe_val(v):
                    try:
                        if v in le.classes_:
                            return int(le.transform([v])[0])
                        else:
                            le.classes_ = np.append(le.classes_, v)
                            return int(le.transform([v])[0])
                    except Exception:
                        return v
                encoded_df[col] = encoded_df[col].apply(_safe_val)

        X_batch = encoded_df[FEATURES].values

        if model is None or encoders is None:
            st.error("Cannot predict: model or encoders missing. Train the model first.")
        else:
            try:
                preds = model.predict(X_batch)
                result_df = batch_df.copy().reset_index(drop=True)
                result_df["Predicted Flight Status"] = preds

                st.subheader("Batch predictions")
                st.write(result_df.head(20))

                # Provide download
                csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")
