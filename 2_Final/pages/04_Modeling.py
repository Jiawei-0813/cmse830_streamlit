#!/usr/bin/env python

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Modeling",
    layout="wide"
)

# Reload the dataset if `combined_data` is not in session_state
if "combined_data" not in st.session_state:
    try:
        # Load data from file
        combined = pd.read_csv("data/1_london_bike_weather_2023.csv")
        # Store it in session state
        st.session_state["combined_data"] = combined
        st.success("Combined dataset reloaded from file!")
    except FileNotFoundError:
        st.error("Combined dataset file not found. Please run the preprocessing step first.")
        st.stop()
else:
    # Retrieve the data from session state if already loaded
    combined = st.session_state["combined_data"]

# Main Navigation
page = st.selectbox(
    "Choose a Section:",
    ["Regression Models", "Ensemble Models"],
    help="Select a predictive analysis technique."
)

# Sidebar slider for train-test split size
test_size = st.sidebar.slider(
    "Test Set Size:",
    min_value=0.1, max_value=0.5, value=0.3, step=0.05,
    help="Adjust the proportion of the test set."
)

# General Utility Functions
target_var = "Bike Counts"

# Directly creating a DataFrame with aggregated bike counts per day
bike_counts = combined.groupby('Date').size().reset_index(name='Bike Counts')
combined = combined.merge(bike_counts, on='Date', how='left')
target_var_column = 'Bike Counts'

predictors = st.multiselect(
    "Predictors:",
    [
        "Real Temperature (°C)", "Feels Like Temperature (°C)", "Humidity (%)",
        "Wind Speed (m/s)", "Day of Week", "Day of Month", "Hour of Day",
        "Station Number (Start)", "Station Number (End)"
    ],
    default=["Real Temperature (°C)", "Humidity (%)"],
    help="Select features to use in the models."
)

# Map predictors to dataset columns
predictor_mapping = {
    "Real Temperature (°C)": "real_temperature_C",
    "Feels Like Temperature (°C)": "feels_like_temperature_C",
    "Humidity (%)": "humidity_percentage",
    "Wind Speed (m/s)": "wind_speed_10m",
    "Day of Week": "Day_of_Week",
    "Day of Month": "Day_of_Month",
    "Hour of Day": "Hour_of_Day",
    "Station Number (Start)": "Start station number",
    "Station Number (End)": "End station number"
}
predictors_mapped = [predictor_mapping[p] for p in predictors]

# Ensure predictors are selected
if not predictors_mapped:
    st.warning("Please select at least one predictor.")
else:
    X = combined[predictors_mapped]
    y = combined[target_var_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Regression Section
    if page == "Regression Models":
        st.header("Regression Analysis")

        # Model Selection
        regression_type = st.sidebar.radio(
            "Select Regression Model:",
            ["Linear", "Ridge", "Lasso", "Kernel Ridge"],
            help="Choose a regression model."
        )

        # Hyperparameter Selection
        alpha = 1.0  # Default value for alpha
        if regression_type in ["Ridge", "Lasso", "Kernel Ridge"]:
            alpha = st.sidebar.slider("Regularization Strength (Alpha):", 0.01, 10.0, 1.0, 0.1)

        kernel_type = "linear"  # Default kernel type for Kernel Ridge
        if regression_type == "Kernel Ridge":
            kernel_type = st.sidebar.selectbox(
                "Kernel Type:", 
                ["linear", "polynomial", "rbf"], 
                help="Choose a kernel for Kernel Ridge Regression."
            )

        # Model Initialization
        model_map = {
            "Linear": LinearRegression(),
            "Ridge": Ridge(alpha=alpha),
            "Lasso": Lasso(alpha=alpha),
            "Kernel Ridge": KernelRidge(alpha=alpha, kernel=kernel_type)
        }
        model = model_map[regression_type]
        model.fit(X_train, y_train)

        # Predictions
        y_train_pred, y_test_pred = model.predict(X_train), model.predict(X_test)

        # Metrics
        metrics = {
            "R² (Train)": r2_score(y_train, y_train_pred),
            "RMSE (Train)": np.sqrt(mean_squared_error(y_train, y_train_pred)),
            "R² (Test)": r2_score(y_test, y_test_pred),
            "RMSE (Test)": np.sqrt(mean_squared_error(y_test, y_test_pred))
        }

        # Display Metrics
        with st.expander("Model Performance Metrics", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(label="R² (Train)", value=f"{metrics['R² (Train)']:.2f}")
            with col2:
                st.metric(label="RMSE (Train)", value=f"{metrics['RMSE (Train)']:.2f}")
            with col3:
                st.metric(label="R² (Test)", value=f"{metrics['R² (Test)']:.2f}")
            with col4:
                st.metric(label="RMSE (Test)", value=f"{metrics['RMSE (Test)']:.2f}")

            # Coefficients (if applicable)
            if regression_type in ["Linear", "Ridge", "Lasso"] and hasattr(model, "coef_") and st.checkbox("Show Coefficients"):
                coef_df = pd.DataFrame({"Predictor": predictors, "Coefficient": model.coef_}).sort_values(by="Coefficient", key=abs, ascending=False)
                st.write("### Coefficients")
                st.dataframe(coef_df)

        # Comparison of Models
        with st.expander("Comparison of Regression Models"):
            models = {
                "Linear": LinearRegression(),
                "Ridge": Ridge(alpha=alpha),
                "Lasso": Lasso(alpha=alpha),
                "Kernel Ridge": KernelRidge(alpha=alpha, kernel=kernel_type)
            }
            results = []
            for name, mdl in models.items():
                start_time = time.time()
                mdl.fit(X_train, y_train)
                running_time = time.time() - start_time
                y_train_pred = mdl.predict(X_train)
                y_test_pred = mdl.predict(X_test)
                results.append({
                    "Model": name,
                    "R² (Train)": r2_score(y_train, y_train_pred),
                    "RMSE (Train)": np.sqrt(mean_squared_error(y_train, y_train_pred)),
                    "R² (Test)": r2_score(y_test, y_test_pred),
                    "RMSE (Test)": np.sqrt(mean_squared_error(y_test, y_test_pred)),
                    "Running Time (s)": running_time
                })

            comparison_df = pd.DataFrame(results)
            st.dataframe(comparison_df)

            # Best Model
            best_model = comparison_df.loc[comparison_df["R² (Test)"].idxmax()]
            st.write(f"### Best Model: {best_model['Model']}")
            st.write(f"**R² (Test):** {best_model['R² (Test)']:.2f}")
            st.write(f"**RMSE (Test):** {best_model['RMSE (Test)']:.2f}")
            st.write(f"**Running Time:** {best_model['Running Time (s)']:.2f} seconds")

    elif page == "Ensemble Models":
        st.header("Ensemble Models")
        # Model Selection
        st.sidebar.subheader("Choose Ensemble Model")
        ensemble_model_type = st.sidebar.radio("Model Type:", ["Gradient Boosting", "Random Forest"])

        # Hyperparameter Tuning
        st.sidebar.subheader("Hyperparameter Tuning")
        if ensemble_model_type == "Gradient Boosting":
            n_estimators = st.sidebar.slider("Number of Trees (n_estimators)", 50, 500, 100, 50)
            max_depth = st.sidebar.slider("Max Depth", 1, 10, 3)
            learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01)
            model = GradientBoostingRegressor(
                n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42
            )
        else:  # Random Forest
            n_estimators = st.sidebar.slider("Number of Trees (n_estimators)", 50, 500, 100, 50)
            max_depth = st.sidebar.slider("Max Depth", 1, 10, 3)
            model = RandomForestRegressor(
                n_estimators=n_estimators, max_depth=max_depth, random_state=42
            )
        
        # Model Training and Predictions
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Metrics
        r2_train = r2_score(y_train, y_train_pred)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        r2_test = r2_score(y_test, y_test_pred)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        # Display Metrics
        with st.expander("Model Performance Metrics", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(label="R² (Train)", value=f"{r2_train:.2f}")
            with col2:
                st.metric(label="RMSE (Train)", value=f"{rmse_train:.2f}")
            with col3:
                st.metric(label="R² (Test)", value=f"{r2_test:.2f}")
            with col4:
                st.metric(label="RMSE (Test)", value=f"{rmse_test:.2f}")
        
        # Feature Importance
        if st.checkbox("Show Feature Importance", value=True):
            if hasattr(model, "feature_importances_"):
                feature_importance = pd.DataFrame({
                    "Feature": predictors,
                    "Importance": model.feature_importances_
                }).sort_values(by="Importance", ascending=False)
                st.write("### Feature Importance")
                st.dataframe(feature_importance, use_container_width=True)

                # Key Insight
                st.write(f"**Key Insight:** The selected {ensemble_model_type} model achieved an R² of **{r2_test:.2f}** on the test set. "
                        f"Feature importance analysis highlights **{feature_importance.iloc[0]['Feature']}** as the most significant predictor.")
            else:
                st.warning(f"The selected model `{ensemble_model_type}` does not support feature importance analysis.")
        
        # Plot Actual vs Predicted
        if st.checkbox("Show Actual vs Predicted Plot"):
            st.subheader("Actual vs Predicted Values")
            fig = px.scatter(
                x=list(y_train) + list(y_test),
                y=list(y_train_pred) + list(y_test_pred),
                color=["Train"] * len(y_train) + ["Test"] * len(y_test),
                labels={"x": "Actual Values", "y": "Predicted Values", "color": "Dataset"},
                color_discrete_sequence=["blue", "orange"]
            )
            fig.add_trace(
                go.Scatter(
                    x=list(y_train) + list(y_test),
                    y=list(y_train) + list(y_test),
                    mode="lines",
                    line=dict(color="green", dash="dash"),
                    name="Ideal Fit (y=x)"
                )
            )
            st.plotly_chart(fig, use_container_width=True)

        # Model Summary Insights
        # st.write(f"**Key Insight:** The selected {ensemble_model_type} model achieved an R² of **{r2_test:.2f}** on the test set. Feature importance analysis highlights **{feature_importance.iloc[0]['Feature']}** as the most significant predictor.")
        # Time Series Analysis Section
        # elif page == "Time Series Analysis":
        #     st.header("Time Series Analysis")

        #     # Feature selection for modeling
        #     st.write("### Select Features for Time-Series Analysis")
        #     ts_features = st.multiselect(
        #         "Choose Features for Time-Series Analysis:",
        #         ['Hour_of_Day', 'Day_of_Week', 'Day_of_Month'],
        #         help="Select features to enhance the model."
        #     )

        #     # Prepare time-series data
        #     ts_data = combined[['Date'] + ts_features + [target_var_column]].copy()
        #     ts_data['Date'] = pd.to_datetime(ts_data['Date'], errors='coerce')  # Ensure datetime format
        #     ts_data = ts_data.groupby('Date').sum().reset_index()  # Aggregate the target variable by date
            
        #     # Display preview of time-series data
        #     st.write("### Time-Series Data Preview")
        #     st.dataframe(ts_data.head())
        #     st.line_chart(ts_data.set_index('Date')[target_var_column], use_container_width=True, title=f"{target_var} Overview")

        #     # Select forecasting model
        #     st.write("### Select Forecasting Model")
        #     forecasting_model = st.radio(
        #         "Choose a Model for Time-Series Forecasting:",
        #         ["ARIMA", "Prophet"],
        #         help="Select the model to predict future values."
        #     )

        #     if forecasting_model == "ARIMA":
        #         # ARIMA model setup
        #         st.write("### ARIMA Model Configuration")
        #         p = st.slider("ARIMA Order (p)", 0, 5, 1)
        #         d = st.slider("ARIMA Order (d)", 0, 2, 1)
        #         q = st.slider("ARIMA Order (q)", 0, 5, 1)

        #         seasonal = st.checkbox("Add Seasonal Component (SARIMA)?", value=False)
        #         if seasonal:
        #             P = st.slider("Seasonal Order (P)", 0, 5, 1)
        #             D = st.slider("Seasonal Order (D)", 0, 2, 1)
        #             Q = st.slider("Seasonal Order (Q)", 0, 5, 1)
        #             s = st.slider("Seasonal Period (s)", 1, 365, 7, help="Set the seasonal period, e.g., 7 for weekly data.")

        #         # Fit ARIMA/SARIMA model
        #         st.write("### Training ARIMA Model")
        #         try:
        #             if seasonal:
        #                 from statsmodels.tsa.statespace.sarimax import SARIMAX
        #                 model = SARIMAX(ts_data[target_var_column], order=(p, d, q), seasonal_order=(P, D, Q, s))
        #             else:
        #                 from statsmodels.tsa.arima.model import ARIMA
        #                 model = ARIMA(ts_data[target_var_column], order=(p, d, q))

        #             results = model.fit()

        #             # Forecasting
        #             steps = st.slider("Number of Steps to Forecast", 1, 365, 30)
        #             forecast = results.get_forecast(steps=steps)
        #             forecast_df = forecast.summary_frame()

        #             # Visualization
        #             st.write("### ARIMA Forecast Results")
        #             st.line_chart(forecast_df[['mean']], use_container_width=True)
        #             st.write(forecast_df)

        #         except Exception as e:
        #             st.error(f"Error training ARIMA model: {e}")

        #     elif forecasting_model == "Prophet":
        #         # Prophet model setup
        #         st.write("### Prophet Model Configuration")
        #         daily = st.checkbox("Enable Daily Seasonality?", value=True)
        #         weekly = st.checkbox("Enable Weekly Seasonality?", value=True)

        #         # Fit Prophet model
        #         st.write("### Training Prophet Model")
        #         try:
        #             from prophet import Prophet
        #             ts_data.rename(columns={"Date": "ds", target_var_column: "y"}, inplace=True)  # Rename for Prophet
        #             model = Prophet(
        #                 daily_seasonality=daily,
        #                 weekly_seasonality=weekly,
        #             )
        #             model.fit(ts_data)

        #             # Forecasting
        #             steps = st.slider("Number of Days to Forecast", 1, 365, 30)
        #             future = model.make_future_dataframe(periods=steps)
        #             forecast = model.predict(future)

        #             # Visualization
        #             st.write("### Prophet Forecast Results")
        #             fig = model.plot(forecast)
        #             st.plotly_chart(fig, use_container_width=True)

        #             # Seasonal components
        #             st.write("### Seasonal Components")
        #             fig_components = model.plot_components(forecast)
        #             st.pyplot(fig_components)

        #         except Exception as e:
        #             st.error(f"Error training Prophet model: {e}")

        #     else:
        #         st.error("The required columns for time-series analysis are missing. Please ensure the dataset contains 'Hour_of_Day', 'Day_of_Week', 'Day_of_Month', and 'Date'.")
