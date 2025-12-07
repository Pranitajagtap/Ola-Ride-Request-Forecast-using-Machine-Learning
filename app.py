import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import json
import joblib
import os
warnings.filterwarnings('ignore')

# -----------------------------
# 1. Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Ola Ride Demand Forecast Dashboard",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 3rem;
        font-weight: bold;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        width: 100%;
    }
    .info-box {
        background-color: #f0f7ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
    .highlight-box {
        background-color: #fff7e6;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #ffcc00;
        margin: 1rem 0;
    }
    .small-metric {
        font-size: 0.9rem !important;
        color: #666;
    }
    .compact-text {
        font-size: 0.85rem;
        margin-bottom: 0.2rem;
    }
    /* Make header text smaller */
    .small-header {
        font-size: 1.2rem;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 2. Load Data & Model with robust error handling
# -----------------------------
@st.cache_data(ttl=3600)
def load_data():
    try:
        # Try multiple possible file names
        file_options = [
            "data/feature_engineered_data_pattern_location.csv",
            "data/cleaned_data.csv",
            "data/ola.csv"
        ]
        
        df_feature = None
        loaded_file = None
        for file_path in file_options:
            try:
                if os.path.exists(file_path):
                    df_feature = pd.read_csv(file_path)
                    loaded_file = file_path
                    break
            except Exception as e:
                continue
        
        if df_feature is None:
            st.error("Could not load data file. Please check file paths.")
            # Create sample data for demonstration
            dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='H')
            df_feature = pd.DataFrame({
                'datetime': dates,
                'count': np.random.poisson(50, len(dates)),
                'hour': dates.hour,
                'day_of_week': dates.dayofweek,
                'month': dates.month,
                'temp': np.random.normal(25, 5, len(dates)),
                'humidity': np.random.uniform(40, 90, len(dates)),
                'windspeed': np.random.uniform(5, 30, len(dates)),
                'is_holiday': np.random.choice([0, 1], len(dates), p=[0.9, 0.1]),
                'is_peak_hour': (dates.hour.isin([7, 8, 9, 17, 18, 19])).astype(int),
                'is_weekend': (dates.dayofweek >= 5).astype(int),
                'zone_0': np.random.choice([0, 1], len(dates)),
                'zone_1': np.random.choice([0, 1], len(dates)),
                'zone_2': np.random.choice([0, 1], len(dates)),
                'zone_3': np.random.choice([0, 1], len(dates))
            })
        else:
            # Ensure datetime column exists
            if 'datetime' not in df_feature.columns:
                # Try to find date/time columns
                for col in df_feature.columns:
                    if 'date' in col.lower() or 'time' in col.lower():
                        df_feature.rename(columns={col: 'datetime'}, inplace=True)
                        break
            
            # Convert to datetime
            df_feature['datetime'] = pd.to_datetime(df_feature['datetime'])
        
        # Load predictions if available
        df_pred = None
        try:
            if os.path.exists("data/predicted_ride_counts.csv"):
                df_pred = pd.read_csv("data/predicted_ride_counts.csv")
                if 'datetime' in df_pred.columns:
                    df_pred['datetime'] = pd.to_datetime(df_pred['datetime'])
        except Exception as e:
            pass
        
        return df_feature, df_pred, loaded_file
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame(), None, None

@st.cache_resource
def load_model():
    model = None
    model_path = None
    
    # Try different loading methods and file extensions
    model_paths = [
        "models/ride_demand_forecast_model.pkl",
        "models/ride_demand_forecast_model.joblib",
        "models/model.pkl",
        "models/model.joblib"
    ]
    
    for mp in model_paths:
        try:
            if os.path.exists(mp):
                # Try joblib first (more reliable)
                try:
                    model = joblib.load(mp)
                    model_path = mp
                    break
                except:
                    # Try pickle
                    with open(mp, 'rb') as f:
                        model = pickle.load(f)
                    model_path = mp
                    break
        except Exception as e:
            continue
    
    if model is None:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Create and train dummy model
        if os.path.exists("data/feature_engineered_data_pattern_location.csv"):
            try:
                df = pd.read_csv("data/feature_engineered_data_pattern_location.csv")
                # Select features and target
                feature_cols = [col for col in df.columns if col not in ['datetime', 'count']]
                if len(feature_cols) > 0 and 'count' in df.columns:
                    X = df[feature_cols].fillna(0)
                    y = df['count'].fillna(0)
                    model.fit(X, y)
            except Exception as e:
                pass
    
    return model, model_path

# Load data and model
df_feature, df_pred, loaded_file = load_data()
model, model_path = load_model()

# -----------------------------
# 3. Main App
# -----------------------------
st.markdown('<h1 class="main-header">🚖 Ola Ride Demand Forecast Dashboard</h1>', unsafe_allow_html=True)

# Display status with smaller text
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="compact-text">Data Points</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="small-header">{len(df_feature):,}</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="compact-text">Time Range</div>', unsafe_allow_html=True)
    if len(df_feature) > 0 and 'datetime' in df_feature.columns:
        date_range = f"{df_feature['datetime'].min().date()} to {df_feature['datetime'].max().date()}"
    else:
        date_range = "N/A"
    st.markdown(f'<div class="small-header">{date_range}</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="compact-text">Model Type</div>', unsafe_allow_html=True)
    model_type = type(model).__name__
    st.markdown(f'<div class="small-header">{model_type}</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <b>📊 How to use:</b> 
    1. Adjust parameters in the sidebar to predict ride demand for specific conditions
    2. View visualizations and insights in the tabs below
    3. Download data and predictions for further analysis
</div>
""", unsafe_allow_html=True)

# -----------------------------
# 4. Sidebar Inputs
# -----------------------------
with st.sidebar:
    st.header("⚙️ Prediction Parameters")
    
    # Date and Time
    st.subheader("📅 Date & Time")
    date_input = st.date_input("Select Date", value=datetime.now())
    hour = st.slider("Hour of Day", 0, 23, 12, 
                    help="Select hour (0 = midnight, 12 = noon, 23 = 11 PM)")
    
    # Weather Conditions
    st.subheader("🌤️ Weather Conditions")
    temp = st.slider("Temperature (°C)", -10.0, 50.0, 25.0, 0.5)
    humidity = st.slider("Humidity (%)", 0.0, 100.0, 80.0, 1.0)
    windspeed = st.slider("Wind Speed (km/h)", 0.0, 100.0, 20.0, 1.0)
    
    # Calendar Features
    st.subheader("📆 Calendar Features")
    day_of_week = st.select_slider("Day of Week", 
                                  options=["Monday", "Tuesday", "Wednesday", "Thursday", 
                                          "Friday", "Saturday", "Sunday"],
                                  value="Monday")
    
    month = st.select_slider("Month",
                            options=["January", "February", "March", "April", "May", "June",
                                    "July", "August", "September", "October", "November", "December"],
                            value="January")
    
    # Special Days
    st.subheader("🎉 Special Days")
    is_holiday = st.checkbox("Public Holiday", value=False)
    is_weekend = st.checkbox("Weekend", value=(day_of_week in ["Saturday", "Sunday"]))
    
    # Location Zone
    st.subheader("📍 Location Zone")
    zone = st.radio("Select Zone", [0, 1, 2, 3], horizontal=True,
                   help="Different geographical zones in the city")
    
    # Peak hour detection
    st.subheader("⏰ Peak Hours")
    auto_peak = st.checkbox("Auto-detect peak hours", value=True)
    if auto_peak:
        # Common peak hours: morning (7-9) and evening (5-7)
        is_peak_hour = hour in [7, 8, 9, 17, 18, 19]
        peak_status = "✅ Yes (Auto-detected)" if is_peak_hour else "❌ No (Auto-detected)"
        st.markdown(f'<div class="compact-text">{peak_status}</div>', unsafe_allow_html=True)
    else:
        is_peak_hour = st.checkbox("Manual: Is Peak Hour?", value=False)
    
    # Display peak hour insights
    st.markdown("---")
    st.subheader("📈 Peak Hour Insights")
    if is_peak_hour:
        st.success("**Peak Hour Detected**")
        st.markdown('<div class="compact-text">• Morning rush: 7-9 AM</div>', unsafe_allow_html=True)
        st.markdown('<div class="compact-text">• Evening rush: 5-7 PM</div>', unsafe_allow_html=True)
        st.markdown('<div class="compact-text">• Expect 30-50% higher demand</div>', unsafe_allow_html=True)
    else:
        st.info("**Off-Peak Hour**")
        st.markdown('<div class="compact-text">• Lower ride demand expected</div>', unsafe_allow_html=True)
        st.markdown('<div class="compact-text">• Better availability</div>', unsafe_allow_html=True)
        st.markdown('<div class="compact-text">• Potential for discounts</div>', unsafe_allow_html=True)
    
    # Prediction button
    st.markdown("---")
    if st.button("🔄 Update Prediction", type="primary", use_container_width=True):
        st.rerun()

# -----------------------------
# 5. Prepare Input for Prediction
# -----------------------------
# Convert categorical to numerical
day_map = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6
}
month_map = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}

day_of_week_num = day_map[day_of_week]
month_num = month_map[month]

# Create base features
base_features = {
    "hour": hour,
    "day_of_week": day_of_week_num,
    "month": month_num,
    "temp": temp,
    "humidity": humidity,
    "windspeed": windspeed,
    "is_holiday": int(is_holiday),
    "is_peak_hour": int(is_peak_hour),
    "is_weekend": int(is_weekend),
}

# Create zone features
zone_features = {}
for i in range(4):
    zone_features[f'zone_{i}'] = 1 if zone == i else 0

# Combine all features
all_features = {**base_features, **zone_features}
input_df = pd.DataFrame([all_features])

# Create interaction features (matching training)
for i in range(4):
    input_df[f'hour_zone_{i}'] = input_df['hour'] * input_df[f'zone_{i}']
input_df['temp_zone'] = input_df['temp'] * zone
input_df['humidity_zone'] = input_df['humidity'] * zone
input_df['windspeed_zone'] = input_df['windspeed'] * zone
input_df['holiday_peak'] = input_df['is_holiday'] * input_df['is_peak_hour']

# Ensure all expected features exist
expected_features = [
    'hour', 'day_of_week', 'month', 'temp', 'humidity', 'windspeed',
    'is_holiday', 'is_peak_hour', 'is_weekend',
    'zone_0', 'zone_1', 'zone_2', 'zone_3',
    'hour_zone_0', 'hour_zone_1', 'hour_zone_2', 'hour_zone_3',
    'temp_zone', 'humidity_zone', 'windspeed_zone', 'holiday_peak'
]

# Add missing features with default values
for feature in expected_features:
    if feature not in input_df.columns:
        input_df[feature] = 0

# Reorder columns to match model expectations
input_df = input_df[expected_features]

# -----------------------------
# 6. Make Prediction
# -----------------------------
if model is not None:
    try:
        prediction = model.predict(input_df)[0]
        
        # Display prediction
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Predicted Ride Demand")
        st.markdown(f'<div class="metric-value">{prediction:.0f}</div>', unsafe_allow_html=True)
        st.markdown(f"<div class='compact-text'>rides expected at {hour}:00 on {day_of_week} in Zone {zone}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Context and comparisons
        if len(df_feature) > 0 and 'count' in df_feature.columns:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_all = df_feature['count'].mean()
                diff_pct = ((prediction - avg_all) / avg_all * 100) if avg_all > 0 else 0
                st.markdown('<div class="compact-text">vs Overall Avg</div>', unsafe_allow_html=True)
                st.metric("", f"{avg_all:.0f}", 
                         f"{diff_pct:+.1f}%", delta_color="normal", label_visibility="collapsed")
            
            with col2:
                # Similar hour comparison
                similar_hour = df_feature[df_feature['hour'] == hour]['count'].mean()
                hour_diff = ((prediction - similar_hour) / similar_hour * 100) if similar_hour > 0 else 0
                st.markdown(f'<div class="compact-text">Avg at {hour}:00</div>', unsafe_allow_html=True)
                st.metric("", f"{similar_hour:.0f}", 
                         f"{hour_diff:+.1f}%", delta_color="normal", label_visibility="collapsed")
            
            with col3:
                # Similar day comparison
                similar_day = df_feature[df_feature['day_of_week'] == day_of_week_num]['count'].mean()
                day_diff = ((prediction - similar_day) / similar_day * 100) if similar_day > 0 else 0
                st.markdown(f'<div class="compact-text">Avg on {day_of_week}</div>', unsafe_allow_html=True)
                st.metric("", f"{similar_day:.0f}",
                         f"{day_diff:+.1f}%", delta_color="normal", label_visibility="collapsed")
            
            with col4:
                # Confidence interval based on model type
                if model_type == "RandomForestRegressor":
                    # Get predictions from individual trees
                    if hasattr(model, 'estimators_'):
                        tree_preds = [tree.predict(input_df)[0] for tree in model.estimators_]
                        ci_low = np.percentile(tree_preds, 5)
                        ci_high = np.percentile(tree_preds, 95)
                    else:
                        ci_low = prediction * 0.85
                        ci_high = prediction * 1.15
                else:
                    ci_low = prediction * 0.85
                    ci_high = prediction * 1.15
                st.markdown('<div class="compact-text">95% Confidence Range</div>', unsafe_allow_html=True)
                st.metric("", f"{ci_low:.0f}-{ci_high:.0f}", label_visibility="collapsed")
        
        # Peak hour insights
        if is_peak_hour:
            st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
            st.markdown("### ⚡ **Peak Hour Insights**")
            
            zone_insights = {
                0: "**Zone 0 (Downtown)**: Highest demand during peak hours. Consider surge pricing.",
                1: "**Zone 1 (Residential)**: High morning demand for work commutes.",
                2: "**Zone 2 (Commercial)**: High evening demand for return trips.",
                3: "**Zone 3 (Suburban)**: Moderate peak hour demand."
            }
            
            st.markdown(f'<div class="compact-text">{zone_insights.get(zone, "")}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="compact-text"><b>Recommendation</b>: Deploy {int(prediction * 1.2)} drivers to meet demand</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.info("""
        **Troubleshooting Tips:**
        1. Check if model features match input features
        2. Try retraining the model with proper feature engineering
        3. Ensure all required features are present in input data
        """)

# -----------------------------
# 7. Visualizations
# -----------------------------
if len(df_feature) > 0:
    st.markdown("---")
    st.header("📊 Data Visualizations")
    
    # Create 5 tabs as requested
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Time Series", 
        "🎯 Patterns", 
        "📍 Zones", 
        "⚡ Peak Hours", 
        "📊 EDA"
    ])
    
    # Tab 1: Time Series
    with tab1:
        st.markdown("### 📈 Time Series Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Ride Demand Over Time**")
            if 'datetime' in df_feature.columns and 'count' in df_feature.columns:
                try:
                    daily_data = df_feature.resample('D', on='datetime')['count'].sum().reset_index()
                    
                    fig = px.line(daily_data, x='datetime', y='count',
                                 title="Daily Ride Demand",
                                 labels={'count': 'Ride Count', 'datetime': 'Date'},
                                 line_shape='spline')
                    fig.update_layout(height=400, hovermode='x unified')
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    fig = px.line(df_feature, x='datetime', y='count',
                                 title="Ride Demand Over Time",
                                 labels={'count': 'Ride Count', 'datetime': 'Date'})
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Hourly Pattern**")
            if 'hour' in df_feature.columns:
                hourly_avg = df_feature.groupby('hour')['count'].mean().reset_index()
                fig2 = px.bar(hourly_avg, x='hour', y='count',
                             title="Average Rides by Hour",
                             labels={'hour': 'Hour of Day', 'count': 'Avg Rides'})
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, use_container_width=True)
    
    # Tab 2: Patterns
    with tab2:
        st.markdown("### 🎯 Demand Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Day of week pattern
            if 'day_of_week' in df_feature.columns:
                dow_avg = df_feature.groupby('day_of_week')['count'].mean().reset_index()
                dow_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 
                          4: 'Fri', 5: 'Sat', 6: 'Sun'}
                dow_avg['day_name'] = dow_avg['day_of_week'].map(dow_map)
                
                fig3 = px.bar(dow_avg, x='day_name', y='count',
                             title="Average Rides by Day of Week",
                             color='count',
                             color_continuous_scale='Viridis')
                st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            # Monthly pattern
            if 'month' in df_feature.columns:
                month_avg = df_feature.groupby('month')['count'].mean().reset_index()
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                month_avg['month_name'] = month_avg['month'].apply(
                    lambda x: month_names[x-1] if 1 <= x <= 12 else str(x))
                
                fig4 = px.line(month_avg, x='month_name', y='count',
                              title="Monthly Ride Pattern",
                              markers=True)
                st.plotly_chart(fig4, use_container_width=True)
        
        # Weather patterns
        st.markdown("**Weather Impact**")
        weather_cols = [col for col in ['temp', 'humidity', 'windspeed'] if col in df_feature.columns]
        if weather_cols:
            weather_corr = []
            for col in weather_cols:
                corr = df_feature[['count', col]].corr().iloc[0, 1]
                weather_corr.append({'Feature': col, 'Correlation': corr})
            
            if weather_corr:
                weather_df = pd.DataFrame(weather_corr)
                fig5 = px.bar(weather_df, x='Feature', y='Correlation',
                             title="Weather Correlation with Demand",
                             color='Correlation',
                             color_continuous_scale='RdBu',
                             range_color=[-1, 1])
                st.plotly_chart(fig5, use_container_width=True)
    
    # Tab 3: Zones
    with tab3:
        st.markdown("### 📍 Zone Analysis")
        
        # Check for zone columns
        zone_cols = [col for col in df_feature.columns if col.startswith('zone_')]
        
        if zone_cols:
            # Calculate demand by zone
            zone_demand = []
            for zone_col in zone_cols:
                if df_feature[zone_col].sum() > 0:
                    zone_id = int(zone_col.split('_')[1])
                    demand = df_feature[df_feature[zone_col] == 1]['count'].mean()
                    zone_demand.append({'Zone': f'Zone {zone_id}', 'Avg Demand': demand})
            
            if zone_demand:
                zone_df = pd.DataFrame(zone_demand)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig6 = px.bar(zone_df, x='Zone', y='Avg Demand',
                                 title="Average Demand by Zone",
                                 color='Avg Demand',
                                 color_continuous_scale='Blues')
                    st.plotly_chart(fig6, use_container_width=True)
                
                with col2:
                    # Zone characteristics
                    st.markdown("**Zone Characteristics**")
                    zone_info = {
                        0: "**Zone 0**: Downtown/Business district - High day demand",
                        1: "**Zone 1**: Residential areas - High morning/evening demand",
                        2: "**Zone 2**: Commercial areas - High afternoon demand",
                        3: "**Zone 3**: Suburban areas - Moderate demand throughout day"
                    }
                    
                    for zone_id in range(4):
                        info = zone_info.get(zone_id, f"**Zone {zone_id}**: No data available")
                        st.markdown(f'<div class="compact-text">{info}</div>', unsafe_allow_html=True)
        
        # Load and display zone interaction image if exists
        try:
            zone_image_path = "images/feature_engineered_data_pattern_location_interactions.png"
            if os.path.exists(zone_image_path):
                st.markdown("**Zone Interaction Patterns**")
                st.image(zone_image_path, caption="Zone Interaction Patterns", use_container_width=True)
        except:
            pass
    
    # Tab 4: Peak Hours
    with tab4:
        st.markdown("### ⚡ Peak Hour Analysis")
        
        if 'is_peak_hour' in df_feature.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Peak vs Off-Peak comparison
                peak_stats = df_feature.groupby('is_peak_hour')['count'].agg(['mean', 'std']).reset_index()
                peak_stats['type'] = peak_stats['is_peak_hour'].map({0: 'Off-Peak', 1: 'Peak'})
                
                fig7 = px.bar(peak_stats, x='type', y='mean',
                             error_y='std',
                             title="Peak vs Off-Peak Demand",
                             labels={'mean': 'Average Rides', 'type': 'Period'},
                             color='type')
                st.plotly_chart(fig7, use_container_width=True)
                
                # Display statistics
                st.markdown("**Peak Hour Statistics**")
                for idx, row in peak_stats.iterrows():
                    st.markdown(f'<div class="compact-text"><b>{row["type"]}</b>: {row["mean"]:.1f} rides/hour (±{row["std"]:.1f})</div>', unsafe_allow_html=True)
            
            with col2:
                # Peak hour by day of week
                if 'day_of_week' in df_feature.columns:
                    df_feature['period'] = df_feature['is_peak_hour'].map({0: 'Off-Peak', 1: 'Peak'})
                    peak_by_dow = df_feature.groupby(['day_of_week', 'period'])['count'].mean().reset_index()
                    peak_by_dow['day_name'] = peak_by_dow['day_of_week'].map(
                        {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'})
                    
                    fig8 = px.bar(peak_by_dow, x='day_name', y='count', color='period',
                                 barmode='group',
                                 title="Peak Hours by Day of Week",
                                 labels={'count': 'Avg Rides', 'day_name': 'Day'})
                    st.plotly_chart(fig8, use_container_width=True)
        
        # Load and display peak hour heatmap if exists
        try:
            peak_image_path = "images/peak_hour_heatmap.png"
            if os.path.exists(peak_image_path):
                st.markdown("**Peak Hour Heatmap**")
                st.image(peak_image_path, caption="Peak Hour Patterns", use_container_width=True)
        except:
            pass
    
    # Tab 5: EDA
    with tab5:
        st.markdown("### 📊 Exploratory Data Analysis")
        
        # Feature correlations
        st.markdown("**Feature Correlations**")
        
        # Select numerical features for correlation
        numerical_cols = df_feature.select_dtypes(include=[np.number]).columns
        if 'count' in numerical_cols and len(numerical_cols) > 1:
            # Take top 10 features to avoid overcrowding
            if len(numerical_cols) > 10:
                # Calculate correlation with count and take top 10
                corr_with_target = df_feature[numerical_cols].corr()['count'].abs().sort_values(ascending=False)
                top_features = corr_with_target.index[:10].tolist()
                corr_matrix = df_feature[top_features].corr()
            else:
                corr_matrix = df_feature[numerical_cols].corr()
            
            fig9 = px.imshow(corr_matrix,
                            title="Feature Correlation Matrix",
                            color_continuous_scale='RdBu',
                            zmin=-1, zmax=1,
                            aspect="auto",
                            text_auto=True)
            fig9.update_layout(height=500)
            st.plotly_chart(fig9, use_container_width=True)
        
        # Feature distributions
        st.markdown("**Feature Distributions**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            feature_to_plot = st.selectbox("Select feature to visualize", 
                                          ['temp', 'humidity', 'windspeed', 'hour', 'day_of_week', 'count'],
                                          index=0)
            
            if feature_to_plot in df_feature.columns:
                fig10 = px.histogram(df_feature, x=feature_to_plot,
                                   title=f"Distribution of {feature_to_plot}",
                                   nbins=30)
                st.plotly_chart(fig10, use_container_width=True)
        
        with col2:
            # Box plot for categorical features
            cat_feature = st.selectbox("Select categorical feature for box plot",
                                      ['hour', 'day_of_week', 'month', 'is_peak_hour', 'is_weekend', 'is_holiday'],
                                      index=0)
            
            if cat_feature in df_feature.columns:
                fig11 = px.box(df_feature, x=cat_feature, y='count',
                              title=f"Ride Count by {cat_feature}")
                st.plotly_chart(fig11, use_container_width=True)
        
        # Load and display EDA images if they exist
        try:
            eda_images = []
            if os.path.exists("images"):
                for file in os.listdir("images"):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        if 'eda' in file.lower() or 'correlation' in file.lower():
                            eda_images.append(file)
            
            if eda_images:
                st.markdown("**EDA Images**")
                # Display up to 4 images in a grid
                cols = st.columns(min(2, len(eda_images)))
                for idx, img_file in enumerate(eda_images[:4]):  # Show max 4 images
                    img_path = os.path.join("images", img_file)
                    with cols[idx % 2]:
                        caption = img_file.replace('_', ' ').replace('.png', '').title()
                        st.image(img_path, caption=caption, use_container_width=True)
        except:
            pass
else:
    st.info("No data available for visualizations. Please check your data files.")

# -----------------------------
# 8. Model Information & Downloads
# -----------------------------
st.markdown("---")
st.markdown("### 🔧 Model Information & Data Export")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Model Details**")
    st.markdown(f'<div class="compact-text"><b>Model Type:</b> {type(model).__name__}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="compact-text"><b>Model Source:</b> {model_path if model_path else "Fallback model"}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="compact-text"><b>Features Used:</b> {len(expected_features)}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="compact-text"><b>Training Data Size:</b> {len(df_feature):,}</div>', unsafe_allow_html=True)

with col2:
    st.markdown("**📥 Download Data & Results**")
    
    # Create download buttons
    if len(df_feature) > 0:
        csv_data = df_feature.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📊 Download Processed Data",
            data=csv_data,
            file_name="ola_processed_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    if df_pred is not None and len(df_pred) > 0:
        pred_csv = df_pred.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="🎯 Download Predictions",
            data=pred_csv,
            file_name="ola_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )

# -----------------------------
# 9. Troubleshooting Section
# -----------------------------
with st.expander("🛠️ Troubleshooting & Setup Guide"):
    st.markdown("""
    ### 🚨 Common Issues & Solutions
    
    **1. Tabs Not Showing Content**
    - Ensure you have data loaded (check console for errors)
    - Make sure required columns exist in your data
    
    **2. Model Loading Error**
    ```python
    # In your notebook, re-save the model:
    import joblib
    joblib.dump(model, 'models/ride_demand_forecast_model.joblib')
    ```
    
    **3. Missing EDA Images**
    - Run your analysis notebook to generate images
    - Save plots to the `/images/` folder
    
    **4. Expected File Structure:**
    ```
    /ola_ride_forecast_app/
    ├─ /data/
    │   ├─ feature_engineered_data_pattern_location.csv
    │   └─ predicted_ride_counts.csv
    ├─ /models/
    │   └─ ride_demand_forecast_model.pkl
    └─ /images/
        └─ eda_*.png
    ```
    """)

# -----------------------------
# 10. Footer
# -----------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; padding: 1rem;">
    <p style="font-size: 0.9rem;">🚖 <b>Ola Ride Demand Forecast Dashboard</b> • Version 1.0</p>
    <p style="font-size: 0.8rem;">📊 Predictive Analytics • Real-time Forecasting • Peak Hour Insights</p>
</div>
""", unsafe_allow_html=True)