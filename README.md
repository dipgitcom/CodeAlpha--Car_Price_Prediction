# Car Price Prediction App 🚗

## Overview

This project is a car price prediction web application built using **Streamlit**. It allows users to upload a dataset containing car details and predict the selling price of a car based on various features.

## Features
- Upload your car dataset (CSV) to explore and predict car prices.
- Exploratory Data Analysis (EDA) with visualizations.
- Train a machine learning model (Random Forest Regressor).
- Make predictions for a car's selling price.

## Prerequisites

Make sure you have **Python 3.6+** and **pip** installed.

### Install Dependencies

You can install all the required dependencies by running:

```bash
pip install -r requirements.txt

Predictions are made on the test dataset. Mean Squared Error (MSE) and R² Score are calculated to assess the model’s prediction accuracy. Feature Importance:

A feature importance plot is generated to show which features contribute the most to predicting the selling price. Visualization:

Selling Price vs. Present Price: A scatter plot shows how the present price relates to the selling price. Car Age Distribution: A histogram visualizes the distribution of car ages in the dataset.
