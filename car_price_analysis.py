import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import os
import io

# Set Streamlit page configuration
st.set_page_config(page_title="Car Price Prediction", layout="wide")

# Add custom CSS for styling
st.markdown(
    """
    <style>
    .reportview-container {
        background: linear-gradient(to right, #ff7e5f, #feb47b);
        font-family: 'Arial', sans-serif;
    }
    .sidebar .sidebar-content {
        background: #31314F;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("🚗 Car Price Predictor")
st.markdown("Welcome to the Car Price Prediction app! Upload your dataset and explore predictions with advanced analytics.")

# File uploader
uploaded_file = st.file_uploader("Upload your car dataset (CSV)", type="csv")

if uploaded_file:
    try:
        # Load and preprocess data
        df = pd.read_csv(uploaded_file)

        # Validate required columns
        required_columns = {'Year', 'Car_Name', 'Present_Price', 'Selling_Price', 'Fuel_Type', 'Selling_type', 'Transmission'}
        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            st.error(f"The dataset is missing required columns: {missing_columns}")
        else:
            # Tabs for better organization
            tab1, tab2, tab3, tab4 = st.tabs(["Dataset Info", "EDA", "Model Training", "Prediction"])

            with tab1:
                st.subheader("Dataset Information")
                buffer = io.StringIO()
                df.info(buf=buffer)
                s = buffer.getvalue()
                st.text(s)
                st.write("First few rows of the dataset:")
                st.write(df.head())

            # Preprocess data
            df['Car_Age'] = 2024 - df['Year']
            df = df.drop(columns=['Year', 'Car_Name'])
            df = pd.get_dummies(df, columns=['Fuel_Type', 'Selling_type', 'Transmission'], drop_first=True)

            with tab2:
                st.subheader("Exploratory Data Analysis (EDA)")
                col1, col2 = st.columns(2)

                with col1:
                    st.write("Null Values in the Dataset")
                    st.write(df.isnull().sum())
                with col2:
                    st.write("Data Distribution Overview")
                    st.write(df.describe())

                # Visualizations
                st.subheader("Selling Price vs. Present Price")
                fig = px.scatter(df, x='Present_Price', y='Selling_Price', color='Fuel_Type_Petrol')
                st.plotly_chart(fig)

                st.subheader("Distribution of Car Age")
                fig = px.histogram(df, x='Car_Age', nbins=10, title="Car Age Distribution")
                st.plotly_chart(fig)

            # Splitting data
            X = df.drop(columns=['Selling_Price'])
            y = df['Selling_Price']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            with tab3:
                st.subheader("Model Training")
                col1, col2 = st.columns(2)

                with col1:
                    n_estimators = st.slider("Number of Estimators (Trees)", 50, 200, 100)
                with col2:
                    max_depth = st.slider("Max Depth of Trees", 5, 20, 10)

                rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                rf_model.fit(X_train, y_train)

                y_pred = rf_model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                st.metric(label="Mean Squared Error (MSE)", value=f"{mse:.2f}")
                st.metric(label="R\u00b2 Score", value=f"{r2:.2f}")

                # Feature Importances
                feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf_model.feature_importances_}).sort_values(by='Importance', ascending=False)
                st.subheader("Feature Importances")
                st.write(feature_importances)

                fig = px.bar(feature_importances, x='Importance', y='Feature', orientation='h', title="Feature Importance")
                st.plotly_chart(fig)

            with tab4:
                st.subheader("Predict Selling Price")

                # User input
                col1, col2 = st.columns(2)
                with col1:
                    car_age = st.number_input("Car Age (Years)", min_value=0, max_value=50, value=5)
                    present_price = st.number_input("Present Price (in Lakhs)", min_value=0.0, value=5.0)
                with col2:
                    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
                    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
                    selling_type = st.selectbox("Selling Type", ["Dealer", "Individual"])
                    driven_kms = st.number_input("Driven Kilometers", min_value=0, value=10000)

                # Prepare user data
                user_data = pd.DataFrame({
                    'Present_Price': [present_price],
                    'Car_Age': [car_age],
                    'Fuel_Type_Diesel': [1 if fuel_type == "Diesel" else 0],
                    'Fuel_Type_Petrol': [1 if fuel_type == "Petrol" else 0],
                    'Selling_type_Individual': [1 if selling_type == "Individual" else 0],
                    'Transmission_Manual': [1 if transmission == "Manual" else 0],
                    'Driven_kms': [driven_kms]
                })

                missing_cols = set(X.columns) - set(user_data.columns)
                for col in missing_cols:
                    user_data[col] = 0

                user_data = user_data[X.columns]

                if st.button("Predict Selling Price"):
                    predicted_price = rf_model.predict(user_data)
                    st.success(f"Predicted Selling Price: \u20b9{predicted_price[0]:.2f} Lakhs")
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.write("Please upload a CSV file to proceed.")
