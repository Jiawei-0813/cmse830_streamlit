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


st.set_page_config(
    page_title="Modeling",
    layout="wide"
)

# Load the combined dataset
if "combined_data" not in st.session_state:
    combined_data = st.session_state["combined_data"]

# Sidebar Navigation
st.sidebar.title("Modeling Menu")
page = st.sidebar.radio(
    "Choose a Section:",
    ["Introduction", "Regression Models", "Ensemble Models", "Time Series Analysis"],
    help="Select a predictive analysis techniques."
)

# Sidebar slider for train-test split size
test_size = st.sidebar.slider(
    "Test Set Size:",
    min_value=0.1, max_value=0.5, value=0.3, step=0.05,
    help="Adjust the proportion of the test set."
)

# General Utility Functions


# Predictor selection (common for all sections)
st.sidebar.write("### Select Predictors and Target")
target_var = st.sidebar.radio(
    "Target Variable:",
    ["Total Duration (Minutes)", "Bike Counts"],
    help="Choose the variable to predict."
)

if target_var == "Bike Counts":
    # Directly creating a DataFrame with aggregated bike counts per day
    bike_counts = combined.groupby('Date').size().reset_index(name='Bike Counts')
    combined = combined.merge(bike_counts, on='Date', how='left')
    target_var_column = 'Bike Counts'

else:
    target_var_column = 'Total_duration_min'

predictors = st.multiselect(
    "Predictors:",
    [
        "Real Temperature (°C)", "Feels Like Temperature (°C)", "Humidity (%)",
        "Wind Speed (m/s)", "Day of Week", "Day of Month", "Hour of Day",
        "Time of Day", "Station Number (Start)", "Station Number (End)"
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
    "Time of Day": "Time_of_Day",
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
    if modeling_option == "Regression":
            st.header("Regression Analysis")
            regression_type = st.selectbox(
                "Regression Model:",
                ["Linear Regression", "Ridge Regression", "Lasso Regression", "Kernel Ridge Regression"],
                help="Choose a regression model."
            )

            # Hyperparameters
            if regression_type in ["Ridge Regression", "Lasso Regression", "Kernel Ridge Regression"]:
                alpha = st.slider("Regularization Strength (Alpha):", 0.01, 10.0, 1.0, 0.1)
            if regression_type == "Kernel Ridge Regression":
                kernel_type = st.selectbox(
                    "Kernel Type:",
                    ["linear", "polynomial", "rbf"],
                    help="Choose a kernel for Kernel Ridge Regression."
                )
            
            # Initialize and fit the model
            if regression_type == "Linear Regression":
                model = LinearRegression()
            elif regression_type == "Ridge Regression":
                model = Ridge(alpha=alpha)
            elif regression_type == "Lasso Regression":
                model = Lasso(alpha=alpha)
            else:  # Kernel Ridge Regression
                model = KernelRidge(alpha=alpha, kernel=kernel_type)

            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)       

            # Metrics
            r2_train = r2_score(y_train, y_train_pred)
            rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
            r2_test = r2_score(y_test, y_test_pred)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

            # Display Metrics
            st.subheader("Model Performance")
            cols = st.columns(4)
            cols[0].metric("R² (Train)", f"{r2_train:.2f}")
            cols[1].metric("RMSE (Train)", f"{rmse_train:.2f}")
            cols[2].metric("R² (Test)", f"{r2_test:.2f}")
            cols[3].metric("RMSE (Test)", f"{rmse_test:.2f}")

            # Plot Actual vs Predicted
            fig = px.scatter(
                x=list(y_train) + list(y_test),
                y=list(y_train_pred) + list(y_test_pred),
                color=["Train"] * len(y_train) + ["Test"] * len(y_test),
                labels={"x": "Actual Values", "y": "Predicted Values", "color": "Dataset"},
                title="Actual vs Predicted Values",
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

            # Coefficients
            if regression_type != "Kernel Ridge Regression" and st.checkbox("Show Coefficients (if applicable)", value=False):
                if hasattr(model, "coef_"):
                    coef_df = pd.DataFrame({
                        "Predictor": predictors,
                        "Coefficient": model.coef_
                    }).sort_values(by="Coefficient", key=abs, ascending=False)
                    st.write("### Coefficients")
                    st.dataframe(coef_df, use_container_width=True)
            
    # Ensemble Models Section
    elif modeling_option == "Ensemble Models":
        st.title("Ensemble Models: Gradient Boosting & Random Forest")
        # Model Selection
        st.subheader("Choose Ensemble Model")
        ensemble_model_type = st.radio("Model Type:", ["Gradient Boosting", "Random Forest"])

        # Hyperparameter Tuning
        st.subheader("Hyperparameter Tuning")
        if ensemble_model_type == "Gradient Boosting":
            n_estimators = st.slider("Number of Trees (n_estimators)", 50, 500, 100, 50)
            max_depth = st.slider("Max Depth", 1, 10, 3)
            learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01)
            model = GradientBoostingRegressor(
                n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42
            )
        else:  # Random Forest
            n_estimators = st.slider("Number of Trees (n_estimators)", 50, 500, 100, 50)
            max_depth = st.slider("Max Depth", 1, 10, 3)
            model = RandomForestRegressor(
                n_estimators=n_estimators, max_depth=max_depth, random_state=42
            )

        # Model Training and Predictions
        st.subheader("Model Training and Evaluation")
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Metrics
        r2_train = r2_score(y_train, y_train_pred)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        r2_test = r2_score(y_test, y_test_pred)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

        # Display Metrics
        st.write("### Model Performance")
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
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            "Feature": predictors,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        st.dataframe(feature_importance, use_container_width=True)

        # Plot Actual vs Predicted
        st.subheader("Actual vs Predicted Values")
        fig = px.scatter(
            x=list(y_train) + list(y_test),
            y=list(y_train_pred) + list(y_test_pred),
            color=["Train"] * len(y_train) + ["Test"] * len(y_test),
            labels={"x": "Actual Values", "y": "Predicted Values", "color": "Dataset"},
            title="Actual vs Predicted Values",
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
        st.write(f"**Key Insight:** The selected {ensemble_model_type} model achieved an R² of **{r2_test:.2f}** on the test set. Feature importance analysis highlights **{feature_importance.iloc[0]['Feature']}** as the most significant predictor.")

    # Time Series Analysis Section
    elif modeling_option == "Time Series Analysis":
        st.header("Time Series Analysis")

        # Feature selection for modeling
        st.write("### Select Features for Time-Series Analysis")
        ts_features = st.multiselect(
            "Choose Features for Time-Series Analysis:",
            ['Hour_of_Day', 'Day_of_Week', 'Day_of_Month'],
            help="Select features to enhance the model."
        )

        # Prepare time-series data
        ts_data = combined[['Date'] + ts_features + [target_var_column]].copy()
        ts_data['Date'] = pd.to_datetime(ts_data['Date'], errors='coerce')  # Ensure datetime format
        ts_data = ts_data.groupby('Date').sum().reset_index()  # Aggregate the target variable by date
        
        # Display preview of time-series data
        st.write("### Time-Series Data Preview")
        st.dataframe(ts_data.head())
        st.line_chart(ts_data.set_index('Date')[target_var_column], use_container_width=True, title=f"{target_var} Overview")

        # Select forecasting model
        st.write("### Select Forecasting Model")
        forecasting_model = st.radio(
            "Choose a Model for Time-Series Forecasting:",
            ["ARIMA", "Prophet"],
            help="Select the model to predict future values."
        )

        if forecasting_model == "ARIMA":
            # ARIMA model setup
            st.write("### ARIMA Model Configuration")
            p = st.slider("ARIMA Order (p)", 0, 5, 1)
            d = st.slider("ARIMA Order (d)", 0, 2, 1)
            q = st.slider("ARIMA Order (q)", 0, 5, 1)

            seasonal = st.checkbox("Add Seasonal Component (SARIMA)?", value=False)
            if seasonal:
                P = st.slider("Seasonal Order (P)", 0, 5, 1)
                D = st.slider("Seasonal Order (D)", 0, 2, 1)
                Q = st.slider("Seasonal Order (Q)", 0, 5, 1)
                s = st.slider("Seasonal Period (s)", 1, 365, 7, help="Set the seasonal period, e.g., 7 for weekly data.")

            # Fit ARIMA/SARIMA model
            st.write("### Training ARIMA Model")
            try:
                if seasonal:
                    from statsmodels.tsa.statespace.sarimax import SARIMAX
                    model = SARIMAX(ts_data[target_var_column], order=(p, d, q), seasonal_order=(P, D, Q, s))
                else:
                    from statsmodels.tsa.arima.model import ARIMA
                    model = ARIMA(ts_data[target_var_column], order=(p, d, q))

                results = model.fit()
                st.success("Model trained successfully!")

                # Forecasting
                steps = st.slider("Number of Steps to Forecast", 1, 365, 30)
                forecast = results.get_forecast(steps=steps)
                forecast_df = forecast.summary_frame()

                # Visualization
                st.write("### ARIMA Forecast Results")
                st.line_chart(forecast_df[['mean']], use_container_width=True)
                st.write(forecast_df)

            except Exception as e:
                st.error(f"Error training ARIMA model: {e}")

        elif forecasting_model == "Prophet":
            # Prophet model setup
            st.write("### Prophet Model Configuration")
            daily = st.checkbox("Enable Daily Seasonality?", value=True)
            weekly = st.checkbox("Enable Weekly Seasonality?", value=True)
            yearly = st.checkbox("Enable Yearly Seasonality?", value=True)

            # Fit Prophet model
            st.write("### Training Prophet Model")
            try:
                from prophet import Prophet
                ts_data.rename(columns={"Date": "ds", target_var_column: "y"}, inplace=True)  # Rename for Prophet
                model = Prophet(
                    daily_seasonality=daily,
                    weekly_seasonality=weekly,
                    yearly_seasonality=yearly
                )
                model.fit(ts_data)

                # Forecasting
                steps = st.slider("Number of Days to Forecast", 1, 365, 30)
                future = model.make_future_dataframe(periods=steps)
                forecast = model.predict(future)

                # Visualization
                st.write("### Prophet Forecast Results")
                fig = model.plot(forecast)
                st.plotly_chart(fig, use_container_width=True)

                # Seasonal components
                st.write("### Seasonal Components")
                fig_components = model.plot_components(forecast)
                st.pyplot(fig_components)

            except Exception as e:
                st.error(f"Error training Prophet model: {e}")

    else:
        st.error("The required columns for time-series analysis are missing. Please ensure the dataset contains 'Hour_of_Day', 'Day_of_Week', 'Day_of_Month', and 'Date'.")
