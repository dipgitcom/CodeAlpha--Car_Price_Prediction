# CodeAlpha--Car_Price_Prediction

Car Price Prediction with Machine Learning Overview The Car Price Prediction project is a machine learning-based application designed to predict the selling price of used cars based on various features such as car make, model, year of manufacture, fuel type, transmission type, and more. By leveraging historical car sale data and applying a Random Forest Regressor model, this project forecasts car prices, providing valuable insights for buyers, sellers, and dealerships.

Features Data Upload: Users can upload their car dataset in CSV format. The dataset should include columns like car make, model, year of manufacture, present price, and selling price. Data Cleaning & Preprocessing: The dataset undergoes preprocessing to handle missing values and convert categorical variables into numerical ones using one-hot encoding. A new feature, Car Age, is engineered by subtracting the year of manufacture from the current year. Model Training: The dataset is split into training and test sets, and a Random Forest Regressor model is trained to predict the car's selling price. Prediction & Evaluation: The model's performance is evaluated using Mean Squared Error (MSE) and R² Score metrics. Feature Importance: Displays the relative importance of each feature in predicting the selling price. Visualization: Interactive visualizations to: Show the relationship between Present Price and Selling Price. Illustrate the distribution of Car Age. Display the Feature Importance plot for better insight into the model. Technologies Used Python: The core programming language for data manipulation, model training, and evaluation. Pandas: A powerful data manipulation library for handling and cleaning the dataset. NumPy: A library for numerical operations. Scikit-learn: For machine learning algorithms, model evaluation, and data splitting. Seaborn & Matplotlib: For data visualization. Streamlit: A framework used to create an interactive web application where users can upload datasets and visualize model predictions. Steps Involved Dataset Upload:

Upload a CSV file containing car data. Ensure that the dataset includes features like Year, Fuel_Type, Selling_Price, Present_Price, Owner, etc. Data Preprocessing:

Categorical Columns: Convert categorical features like Fuel_Type, Transmission, etc., into numerical format using one-hot encoding. Feature Engineering: Calculate Car Age and drop irrelevant columns like Car Name. Handling Missing Data: Check for and handle missing values if applicable. Model Training:

The dataset is split into training and test sets. A Random Forest Regressor model is trained using the training data to predict the car's Selling Price. Hyperparameters like the number of estimators are adjusted to optimize the model's performance. Model Evaluation:

Predictions are made on the test dataset. Mean Squared Error (MSE) and R² Score are calculated to assess the model’s prediction accuracy. Feature Importance:

A feature importance plot is generated to show which features contribute the most to predicting the selling price. Visualization:

Selling Price vs. Present Price: A scatter plot shows how the present price relates to the selling price. Car Age Distribution: A histogram visualizes the distribution of car ages in the dataset.
