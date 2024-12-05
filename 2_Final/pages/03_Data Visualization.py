import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Data Visualization",
    layout="wide"
)

if "combined_data" in st.session_state:
    combined = st.session_state["combined_data"]

# Define groups for visualization
weather_cols = ['real_temperature_C', 'humidity_percentage', 'feels_like_temperature_C', 'wind_speed_10m']
time_cols = ['Day_of_Month', 'Day_of_Week', 'Hour_of_Day'] # numeric
numeric_cols = ['Total_duration_min', 'weather_code', 'real_temperature_C', 
            'feels_like_temperature_C', 'wind_speed_10m', 'humidity_percentage', 
            'Day_of_Month', 'Hour_of_Day']

st.header('Data Visualization')

with st.sidebar:
    eda_option = st.selectbox(
        "Pick one to explore:",
        [
            "What correlations exist between variables?",
            "How do bike-sharing trends change over time?",
            "What impact does weather have on bike-sharing?"
        ]
    )

    # Add custom CSS to change the selectbox format
    st.markdown(
        """
        <style>
        div[data-testid="stSelectbox"] {
            background-color: #f0f8ff;
            color: #00008b;
            font-size: 28px; 
            text-align: center;
            border-radius: 10px;
            padding: 10px;
            margin: 10px 0;
            font-family: 'Arial', sans-serif;
        }
        div[data-testid="stSelectbox"] select {
            background-color: #e6f2ff; 
            color: #00008b;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


    if eda_option == "What correlations exist between variables?":

        plt.figure(figsize=(12, 8))
        color_scales = 'Viridis'

        fig_corr = px.imshow(
            combined[numeric_cols].corr(),
            text_auto='.2f',  # 2 decimal places
            aspect="auto",
            color_continuous_scale=color_scales,
            labels=dict(color="Correlation"),
            title="Correlation Heatmap: Overall"
        )
        fig_corr.update_traces(hovertemplate='Corr(%{x}, %{y}) = %{z:.6f}<extra></extra>')  # Show 6 decimal places on hover
        st.plotly_chart(fig_corr)
        plt.title("Correlation Heatmap: Overall", fontsize=26)

        st.markdown("""
        **Observation:**
        - **Temperature** shows a strong positive correlation with bike-sharing patterns.
        - **Real Temperature** and **Humidity** exhibit a noticeable negative correlation, meaning that as humidity increases, temperature tends to decrease.
        - **Time variables** like **Hour of Day** and **Day of Month** show weaker correlations with weather variables and bike-sharing.
        
        Understanding the relationship between weather, especially temperature and humidity, is key to predicting bike-sharing trends.
        """)

    if eda_option == "How do bike-sharing trends change over time?":

        time_variable = st.selectbox(
            "Select one to visualize:",
            ["General Trends", "Time of Day Effect", 
                "Weekday vs Weekend Patterns", "Hourly Patterns by Day"]
        )

        if time_variable == "General Trends":
            selected_time = st.radio(
                "**Select a time variable for trends:**",
                time_cols[::-1],  # Reverse the order of time_cols
                horizontal=True
            )
            if selected_time:
                
                # Count the number of bike-sharing by the selected time variable
                counts = combined.groupby(selected_time).size().reset_index(name='Number of Bike-Sharing')
                
                # Create a line plot to show bike-sharing trends
                fig_time = px.line(
                    counts,
                    x=selected_time,
                    y='Number of Bike-Sharing',
                    title=f"Bike-Sharing by {selected_time}",
                    labels={selected_time: selected_time, 'Number of Bike-Sharing': 'Count'},
                    markers=True,
                    line_shape='linear'
                )
                fig_time.update_traces(mode='lines+markers', line=dict(color='green'), name='Bike-Sharing Count')
                fig_time.add_scatter(
                    x=counts[selected_time],
                    y=counts['Number of Bike-Sharing'],
                    mode='lines',
                    fill='tozeroy',
                    fillcolor='rgba(144, 238, 144, 0.2)',  # Light green fill color
                    line=dict(width=0),
                    name='Filled Area'
                )
                fig_time.update_layout(hovermode='x unified')

                # Calculate the total count of bike-sharing
                total_count = counts['Number of Bike-Sharing'].sum()
                # Calculate the percentage of the total for each count
                counts['Percentage'] = counts['Number of Bike-Sharing'] / total_count * 100

                # Mark the maximum y value
                max_y = counts['Number of Bike-Sharing'].max()
                max_x = counts.loc[counts['Number of Bike-Sharing'].idxmax(), selected_time] # Get the corresponding x value
                max_percentage = counts.loc[counts['Number of Bike-Sharing'].idxmax(), 'Percentage'] # Get the corresponding percentage

                if selected_time == 'Day_of_Week':
                    day_of_week_mapping = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday", 7: "Sunday"}
                    max_x_label = day_of_week_mapping[max_x]
                elif selected_time == 'Day_of_Month':
                    max_x_label = f'Aug {max_x}'
                else:
                    max_x_label = max_x

                # Add a marker for the maximum value
                fig_time.add_scatter(
                    x=[max_x],
                    y=[max_y],
                    mode='markers+text',
                    marker=dict(color='darkgreen', size=10),  # Dark green marker color
                    text=[f'{max_x_label}<br>{max_y:,} ({max_percentage:.2f}%)'],
                    textposition='top center',
                    name='Max Value'
                )

                # Update x-axis labels for Day_of_Week
                if selected_time == 'Day_of_Week':
                    fig_time.update_xaxes(
                        tickmode='array',
                        tickvals=[1, 2, 3, 4, 5, 6, 7],
                        ticktext=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    )
                # Format y-axis numbers as x,xxx
                fig_time.update_yaxes(tickformat=',')

                # Update the hover template to display the count and percentage
                fig_time.update_traces(hovertemplate='%{y:,} (%{customdata[0]:.2f}%)<extra></extra>', customdata=counts[['Percentage']].values)
                
                st.plotly_chart(fig_time)

        elif time_variable == "Time of Day Effect":
            # Group by day of month and time of day
            time_of_day_counts_by_day = combined.groupby(['Day_of_Month', 'Time_of_Day']).size().reset_index(name='Number of Bike-Sharing')

            # Create a line plot with time of day as hue and day of month as x-axis
            fig_time_of_day = px.line(
                time_of_day_counts_by_day,
                x='Day_of_Month',
                y='Number of Bike-Sharing',
                color='Time_of_Day',  # This will act as the hue
                title="Bike-Sharing by Day of Month and Time of Day",
                labels={"Day_of_Month": "Day of Month", "Number of Bike-Sharing": "Count"},
                markers=True,
                category_orders={"Time_of_Day": ["Morning", "Afternoon", "Evening", "Night"]}  # Set hue order
            )

            fig_time_of_day.update_layout(
                hovermode='x unified',  # Unify the hover information
                xaxis=dict(tickmode='linear')  # Ensure all x-axis labels are shown
            )
            st.plotly_chart(fig_time_of_day)

            st.write("""
            **Observation:** The plot shows how bike-sharing activity varies throughout the month, segmented by different times of the day.
            - Evening and Morning times see higher bike-sharing activity, suggesting that many bike users are likely commuters traveling to and from work.
            - Afternoon shows a moderate level of activity, likely due to leisure or errand trips.
            - Night has the lowest activity and fluctuates the least, indicating fewer bike-sharing trips during late hours and more consistent usage patterns at night.
            """)
        
        elif time_variable == "Weekday vs Weekend Patterns":
            # Create counts for weekends and add a new column for labeling
            weekend_counts = combined[combined['is_Weekend_Name'] == 'Weekend'].groupby('Hour_of_Day').size().reset_index(name='Number of Bike-Sharing')
            weekend_counts['is_Weekend_Name'] = 'Weekend'

            # Create counts for weekdays and add a new column for labeling
            weekday_counts = combined[combined['is_Weekend_Name'] == 'Weekday'].groupby('Hour_of_Day').size().reset_index(name='Number of Bike-Sharing')
            weekday_counts['is_Weekend_Name'] = 'Weekday'

            # Combine both weekend and weekday counts
            combined_counts = pd.concat([weekend_counts, weekday_counts])

            # Create line plot to show hourly changes for weekends and weekdays
            fig_hourly = px.line(
                combined_counts,
                x='Hour_of_Day',
                y='Number of Bike-Sharing',
                color='is_Weekend_Name', 
                title="Hourly Bike-Sharing: Weekday vs Weekend",
                labels={"Hour_of_Day": "Hour of Day", "Number of Bike-Sharing": "Count"},
                markers=True,
                color_discrete_map={'Weekend': 'green', 'Weekday': 'lightgreen'}  # Use green shades for the lines
            )

            # Update y-axis to show numbers in xx,xxx format
            fig_hourly.update_layout(
                yaxis=dict(
                    tickformat=','
                ),
                xaxis=dict(
                    range=[0, 24]  # Ensure x-axis starts from 0 and ends at 24
                )
            )
            fig_hourly.update_traces(mode='lines+markers', hovertemplate='N = %{y}')
            fig_hourly.update_layout(hovermode='x unified')
            fig_hourly.update_xaxes(tickmode='linear')  # Ensure all x-axis labels are shown
            st.plotly_chart(fig_hourly)

            st.write("""
                **Observation:** The plot illustrates the hourly bike-sharing patterns for weekdays and weekends.
                - **Weekdays** activity is generally higher than weekend activity, reflecting commuter use during the workweek.
                - On **weekdays**, there aretwo distinct peaks during the **morning** (commuter rush hour) and **early evening** (after-work period), indicating high bike-sharing activity during commute times.
                - **Weekends** show a more consistent distribution of bike-sharing throughout the day, with a slight increase in the afternoon.
                - Both weekends and weekdays tend to have low activity during late-night hours.
            """)

        elif time_variable == "Hourly Patterns by Day":            
            # Define the correct order for days of the week
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

            # Ensure 'Day_of_Week_Name' is treated as a categorical variable with this order
            combined['Day_of_Week_Name'] = pd.Categorical(combined['Day_of_Week_Name'], categories=day_order, ordered=True)

            # Group and unstack to create a 2D matrix
            avg_hour_day = combined.groupby(['Hour_of_Day', 'Day_of_Week_Name']).size().unstack(fill_value=0)

            # Reorder the columns to ensure the days of the week are in the correct order (optional, as pd.Categorical should handle this)
            avg_hour_day = avg_hour_day.reindex(columns=day_order)

            # Create a heatmap
            fig_avg_hour_day = px.imshow(
                avg_hour_day,
                aspect="auto",
                labels=dict(x="Day of Week", y="Hour of Day", color="Average Bike-Sharing"),
                title="Average Bike-Sharing by Hour and Day of Week",
                color_continuous_scale='YlGnBu',
                text_auto='.0f'  # Display annotations with 0 decimal places
            )

            # Ensure that x-axis follows the correct day order
            fig_avg_hour_day.update_layout(
                xaxis_title='Day of Week', 
                yaxis_title='Hour of Day',
                xaxis_categoryorder='array',  # Ensure days of the week are displayed in correct order
                xaxis_tickvals=[0, 1, 2, 3, 4, 5, 6],  # Tick values for the days
                xaxis_ticktext=day_order,  # Text for the tick values
                autosize=True,
                yaxis=dict(tickmode='linear')  # Ensure all y-axis labels are shown
            )

            # Increase the size of the heatmap
            fig_avg_hour_day.update_layout(
                height=800,  
                width=1200   
            )

            st.plotly_chart(fig_avg_hour_day)

            st.markdown("""
            **Observation:** The heatmap visualizes the average number of bike-sharing trips by the hour of the day across each day of the week, showing **clear peak times for bike-sharing** across the week.
            - **Weekdays**, especially **Tuesday and Thursdays**, show the highest activity, with a notable spike in the **late afternoon (4 PM - 7 PM)**. 
            - **Saturdays** consistently show the lowest activity.
            - These patterns highlight the influence of work schedules on bike-sharing trends, with significant usage concentrated during the weekday mornings and evenings.
                        """)

    if eda_option == "What impact does weather have on bike-sharing?":

        weather_option = st.selectbox ("Select one to visualize:", ["Weather Conditions", "Meteorological Factors"])

        if weather_option == "Weather Conditions":   

            st.subheader("Impact of Weather Conditions on Bike-Sharing")

            weather_counts = combined.groupby('Weather Description').size().reset_index(name='Number of Bike-Sharing')
            total_bike_sharing = weather_counts['Number of Bike-Sharing'].sum()
            weather_counts['Percentage'] = (weather_counts['Number of Bike-Sharing'] / total_bike_sharing) * 100

            # Create a bar plot for the number of bike-sharing trips by weather condition
            fig_weather_bar = px.bar(
                weather_counts,
                x='Weather Description',  
                y='Number of Bike-Sharing',
                color='Number of Bike-Sharing',
                color_continuous_scale='Viridis',
                title="Bike-Sharing Distribution by Weather Condition",
                labels={'Number of Bike-Sharing': 'Bike-Sharing Trip Count'}  # Update the color legend
            )

            # Format y-axis numbers as x,xxx
            fig_weather_bar.update_layout(yaxis=dict(tickformat=','))

            # Annotate bars with counts and percentages
            fig_weather_bar.update_traces(
                texttemplate='%{y:,} (%{customdata[0]:.2f}%)',  # Format the text with commas and percentages
                textposition='outside',  # Position the text outside the bars
                customdata=weather_counts[['Percentage']].values  # Add percentage data for annotation
            )

            # Hide the legend and y-axis
            fig_weather_bar.update_layout(showlegend=False, yaxis=dict(visible=False))

            # Adjust the height of the plot
            fig_weather_bar.update_layout(height=500)

            st.plotly_chart(fig_weather_bar)

            st.markdown("""
            **Observation:**
            - **Partly Cloudy** and **Clear Sky** weather conditions have the highest number of bike-sharing trips.
            - Wet weather conditions, such as **Heavy Drizzle** and **Moderate Rain**, see the lowest number of bike-sharing trips.
                        """)
            
            st.divider()

            # Add radio buttons for heatmap selection
            weather_condition_option = st.radio(
                "Select a heatmap to display:",
                ["Hourly Bike-Sharing by Weather Condition", "Bike-Sharing by Day of Week and Weather Condition"]
            )

            if weather_condition_option == "Bike-Sharing Trips per Hour by Weather Condition":
                # Group and unstack to create a 2D matrix for average bike-sharing by hour and weather condition
                avg_hour_weather = combined.groupby(['Hour_of_Day', 'Weather Description']).size().unstack(fill_value=0)

                fig_avg_hour_weather = px.imshow(
                    avg_hour_weather,
                    aspect="auto",
                    labels=dict(y="Hour of Day", x="Weather Condition", color="Bike-Sharing Trip Count"),
                    title="Bike-Sharing Trips per Hour by Weather Condition",
                    color_continuous_scale='Blues',  
                    text_auto='.0f'  # Display annotations with 0 decimal places
                )
                fig_avg_hour_weather.update_layout(
                    height=800,  
                    width=1200,
                    xaxis_title=None, # Remove x-axis label
                    yaxis_title='Hour of Day',  
                    yaxis=dict(tickmode='linear', tick0=0, dtick=1)  # Ensure all y-axis labels are shown from 0 to 23
                )

                st.plotly_chart(fig_avg_hour_weather)

                st.markdown("""
                            **Observation:** This heatmap visualizes the average number of bike-sharing trips across different hours of the day, broken down by various weather conditions.
                            - **Clear Sky** and **Partly Cloudy** conditions consistently show high bike-sharing activity, particularly during the evening hours (5 PM - 7 PM).
                            - **Wet Weather**, such as Drizzle, Heavy Drizzle, and Moderate Rain conditions, exhibit significantly lower bike-sharing activity throughout the day.
                            - The highest activity occurs around 6 PM, especially under **Clear Sky** weather conditions.
                            """)

            if weather_condition_option == "Bike-Sharing by Day of Week and Weather Condition":

                # Group and unstack to create a 2D matrix for average bike-sharing by day and weather condition
                avg_day_weather = combined.groupby(['Weather Description', 'Day_of_Week']).size().unstack(fill_value=0)

                # Define the correct order for days of the week
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

                # Ensure 'Day_of_Week' is treated as a categorical variable with this order
                avg_day_weather.columns = pd.Categorical(avg_day_weather.columns, categories=range(1, 8), ordered=True)
                avg_day_weather.columns = [day_order[day - 1] for day in avg_day_weather.columns]

                # Heatmap for average bike-sharing by day and weather condition
                fig_avg_day_weather = px.imshow(
                    avg_day_weather,
                    aspect="auto",
                    labels=dict(y="Weather Condition", x="Day of Week", color="Bike-Sharing Trip Count"),
                    title="Bike-Sharing by Day of Week and Weather Condition",
                    color_continuous_scale='Blues',
                    text_auto='.0f'
                )
                fig_avg_day_weather.update_layout(
                    xaxis_title=None, 
                    yaxis_title=None
                )
                st.plotly_chart(fig_avg_day_weather)

                st.markdown("""
                                **Observation:** The heatmap visualizes the average number of bike-sharing trips on each day of the week, categorized by various weather conditions.
                            - **Wednesday** and **Thursday** show the highest bike-sharing activity, especially under **Partly Cloudy** and **Clear Sky** conditions.
                            - **Overcast** and **Cloudy** conditions still see moderate activity, particularly in the middle of the week.
                            - **Heavy Drizzle** and **Moderate Rain** conditions result in minimal bike-sharing, with nearly no trips observed on certain days.
                            """)

        if weather_option == "Meteorological Factors":

            st.subheader("Impact of Key Meterological Factors on Bike-Sharing")

            # Violin Plot - Distribution of Bike-Sharing by Weather Variables
            # Reshape the weather columns into long format
            weather_cols = ['real_temperature_C', 'humidity_percentage', 'feels_like_temperature_C', 'wind_speed_10m']

            # Convert the wide data to long format
            long_format = pd.melt(combined, id_vars=[], value_vars=weather_cols, var_name='Weather_Variable', value_name='Value')

            # Add meaningful labels for each weather variable
            weather_variable_labels = {
                'real_temperature_C': 'Temperature (°C)',
                'humidity_percentage': 'Humidity (%)',
                'feels_like_temperature_C': 'Feels Like (°C)',
                'wind_speed_10m': 'Wind Speed (m/s)'
            }

            # Map the weather variables to the labels
            long_format['Weather_Variable'] = long_format['Weather_Variable'].map(weather_variable_labels)

            # Violin plot for weather variables
            fig_weather_facet = px.violin(
                long_format,
                y='Value',
                x='Weather_Variable',
                box=True,  # Add box plot inside the violin
                color='Weather_Variable',
                color_discrete_sequence=px.colors.qualitative.Vivid
            )
            # Update layout
            fig_weather_facet.update_layout(
                title="Bike-Sharing Distribution by Temperature, Humidity, Feels Like Temperature, and Wind Speed",
                xaxis_title=None,  # Remove x-axis title
                yaxis_title="Value (Normalized)",  
                yaxis_range=[0, 1],  # Normalize the y-axis range
                showlegend=False  # Hide the legend
            )

            # Update hover template to show 4 decimal places
            fig_weather_facet.update_traces(hovertemplate='%{y:.4f}')

            st.plotly_chart(fig_weather_facet)

            st.markdown("""
                        **Observation:** The violin plot shows the distribution of bike-sharing activity across different weather variables.
                        - Bike-sharing tends to occur more frequently in **moderate weather conditions**,
                        including mild temperatures, moderate humidity, and calm winds. 
                        - Extreme conditions, such as high humidity or strong winds, see less activity.
                        """)

            st.divider()

            # Linear Regression - Bike-Sharing vs. Weather Variables
            st.subheader("Impact of Weather on Bike-Sharing: Linear Regression Analysis")

            # Prepare the dataset for regression
            bike_weather_data = combined[['real_temperature_C', 'humidity_percentage', 'feels_like_temperature_C', 'wind_speed_10m']]

            # Add a column for Bike-Sharing counts (each row = 1 bike-sharing event)
            bike_weather_data['Bike-Sharing Count'] = 1

            # Group by weather variables
            bike_weather_grouped = bike_weather_data.groupby(['real_temperature_C', 'humidity_percentage', 'feels_like_temperature_C', 'wind_speed_10m']).size().reset_index(name='Bike-Sharing Count')

            # Select variable for regression
            selected_variable = st.selectbox(
                "Select a variable for linear regression:",
                options=['real_temperature_C', 'humidity_percentage', 'feels_like_temperature_C', 'wind_speed_10m']
            )

            if selected_variable:
                # Prepare data for regression
                X = bike_weather_grouped[[selected_variable]]  
                y = bike_weather_grouped['Bike-Sharing Count']  # Dependent variable: Bike-Sharing Count

                # Split into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                # Create and fit the linear regression model
                model = LinearRegression()
                model.fit(X_train, y_train)

                # Predict on the test set
                y_pred = model.predict(X_test)

                # Evaluate the model (R-squared, RMSE)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                # Get the coefficients for the regression equation
                slope = model.coef_[0]
                intercept = model.intercept_

                # Plot the regression line and the data using Plotly
                fig_regression = px.scatter(
                    x=X_test[selected_variable], 
                    y=y_test, 
                    labels={'x': weather_variable_labels[selected_variable], 'y': 'Count'},
                    title=f'Bike-Sharing vs. {weather_variable_labels[selected_variable]} (Linear Regression)',
                    color_discrete_sequence=['#1f77b4']  # Better color for scatter points
                )

                # Add the regression line
                fig_regression.add_trace(
                    go.Scatter(
                        x=X_test[selected_variable], 
                        y=y_pred, 
                        mode='lines',
                        line=dict(color='#ff7f0e'),  # Better color for regression line
                        name='Regression Line'
                    )
                )

                # Update layout
                fig_regression.update_layout(
                    xaxis_title=weather_variable_labels[selected_variable],
                    yaxis_title='Count',
                    showlegend=False
                )

                # Add R-squared and RMSE to the plot in separate locations
                fig_regression.add_annotation(
                    xref="paper", yref="paper",
                    x=0.05, y=0.95,
                    text=f"R-squared: {r2:.2f}<br>RMSE: {rmse:.2f}",
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    align="left",
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=4,
                    bgcolor="#ffebcd",  # Changed background color for annotation
                    opacity=0.8
                )

                # Add annotation for the regression equation 
                fig_regression.add_annotation(
                    x=X_test[selected_variable].mean() + (X_test[selected_variable].max() - X_test[selected_variable].min()) * 0.3,  # Adjust x position to the right
                    y=y_pred.mean() + (y_pred.max() - y_pred.min()) * 0.1,  # Adjust y position to be above the line
                    text=f'y = {slope:.2f}x + {intercept:.2f}',
                    showarrow=False,
                    font=dict(size=12, color="#ff7f0e"),  # Better color for annotation text
                    align="center",
                    bordercolor="#ff7f0e",
                    borderwidth=1,
                    borderpad=4,
                    bgcolor="#fff3e0",  # Better background color for annotation
                    opacity=0.8
                )

                st.plotly_chart(fig_regression)

            with st.popover("Conclusion"):
                st.markdown("""
                            **Observation:** The linear regression analysis shows the relationship between bike-sharing and a selected meteorological factor.
                            - Temperature and Feels Like Temperature have a positive, though weak, relationship with bike-sharing, indicating that higher temperatures lead to more bike-sharing trips.
                            - Humidity appears to have a negative relationship with bike-sharing, with fewer trips at higher humidity levels.
                            - Wind speed seems to have little to no noticeable effect on the number of trips.
                            """)
                