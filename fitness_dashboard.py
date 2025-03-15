# Contents of fitness_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime, timedelta
import folium
from streamlit_folium import folium_static
# import requests

# Page configuration
st.set_page_config(
    page_title="Fitness Journey Planner",
    page_icon="🏋️",
    layout="wide"
)


# Load data and model
@st.cache_data
def load_data():
    df = pd.read_csv('gym_members_exercise_tracking.csv')
    return df


@st.cache_resource
def load_model():
    with open('fitness_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


df = load_data()
model = load_model()

# Dashboard title
st.title("Fitness Journey Planner Pro 🏋️‍♀️")

# Create sidebar for user profile and filters
st.sidebar.header("User Profile")

# User profile input
name = st.sidebar.text_input("Name", "John Doe")
age = st.sidebar.slider("Age", 18, 80, 30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
weight = st.sidebar.slider("Weight (kg)", 40, 150, 75)
height = st.sidebar.slider("Height (m)", 1.4, 2.1, 1.75, 0.01)
workout_freq = st.sidebar.slider("Workout Frequency (days/week)", 1, 7, 3)
workout_type = st.sidebar.selectbox("Preferred Workout Type",
                                    df['Workout_Type'].unique())
water_intake = st.sidebar.slider("Daily Water Intake (liters)", 0.5, 5.0, 2.0, 0.1)

# Calculate BMI and fat percentage (estimated)
bmi = weight / (height ** 2)
fat_percentage = st.sidebar.slider("Fat Percentage", 5.0, 40.0, 20.0, 0.1)

# Create tabs for different dashboard sections
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Workout Analysis", "Fitness Insights", "Recommendations", "Find Trainers"])

# Tab 1: Overview
with tab1:
    # User summary metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Profile Summary")
        st.write(f"**Name:** {name}")
        st.write(f"**Age:** {age}")
        st.write(f"**Gender:** {gender}")
        st.write(f"**BMI:** {bmi:.1f}")

        # BMI category
        if bmi < 18.5:
            bmi_category = "Underweight"
        elif bmi < 25:
            bmi_category = "Normal weight"
        elif bmi < 30:
            bmi_category = "Overweight"
        else:
            bmi_category = "Obese"

        st.write(f"**BMI Category:** {bmi_category}")

    with col2:
        st.subheader("Fitness Stats")
        st.write(f"**Weight:** {weight} kg")
        st.write(f"**Height:** {height} m")
        st.write(f"**Fat Percentage:** {fat_percentage}%")
        st.write(f"**Workout Frequency:** {workout_freq} days/week")
        st.write(f"**Preferred Workout:** {workout_type}")

    with col3:
        # Predict experience level based on input
        # Create a feature vector for prediction
        input_data = {
            'Age': age,
            'Weight (kg)': weight,
            'Height (m)': height,
            'Avg_BPM': 130,  # Default value
            'Session_Duration (hours)': 1.0,  # Default value
            'Calories_Burned': 400,  # Default value
            'Workout_Frequency (days/week)': workout_freq,
            'BMI': bmi,
            'Fat_Percentage': fat_percentage
        }

        # Add dummies for gender and workout type
        for gender_type in df['Gender'].unique():
            input_data[f'Gender_{gender_type}'] = 1 if gender == gender_type else 0

        for wt in df['Workout_Type'].unique():
            input_data[f'Workout_Type_{wt}'] = 1 if workout_type == wt else 0

        # Create dataframe for prediction
        input_df = pd.DataFrame([input_data])

        # Fill any missing columns that the model might expect
        missing_cols = set(model.feature_names_in_) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0

        # Ensure columns are in the right order
        input_df = input_df[model.feature_names_in_]

        # Make prediction
        experience_level = model.predict(input_df)[0]
        experience_probs = model.predict_proba(input_df)[0]

        # Map experience level to text
        experience_text = {1: "Beginner", 2: "Intermediate", 3: "Advanced"}

        st.subheader("Experience Assessment")
        st.write(f"**Predicted Level:** {experience_text[experience_level]}")
        st.write(f"**Confidence:** {experience_probs[experience_level - 1] * 100:.1f}%")

        # Experience level meter
        st.progress((experience_level - 1) / 2)

        # Next level tips
        if experience_level < 3:
            st.write(f"**To reach {experience_text[experience_level + 1]}:**")
            if experience_level == 1:
                st.write("- Increase workout frequency")
                st.write("- Try more varied workout types")
            else:
                st.write("- Increase workout intensity")
                st.write("- Focus on specific muscle groups")

    # Recent activity simulation
    st.subheader("Recent Activity")

    # Generate simulated recent activities
    today = datetime.now()
    activities = []

    for i in range(5):
        date = today - timedelta(days=i)
        if i == 0:
            status = "Completed"
        elif i == 1:
            status = "Completed"
        elif i == 2:
            status = "Missed"
        else:
            status = "Completed"

        if status == "Completed":
            workout = np.random.choice(df['Workout_Type'].unique())
            duration = round(np.random.uniform(0.5, 1.5), 1)
            calories = int(np.random.uniform(200, 500))
        else:
            workout = "-"
            duration = np.nan
            calories = np.nan

        activities.append({
            "Date": date.strftime("%Y-%m-%d"),
            "Workout": workout,
            "Duration (hours)": duration,
            "Calories": calories,
            "Status": status
        })

    # Display activities as a table
    activities_df = pd.DataFrame(activities)
    st.dataframe(activities_df, use_container_width=True)

# Tab 2: Workout Analysis
with tab2:
    st.subheader("Workout Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Workout type distribution
        st.subheader("Workout Types by Experience Level")
        workout_exp = pd.crosstab(df['Workout_Type'], df['Experience_Level'])
        st.bar_chart(workout_exp)

        # Calories burned by workout
        st.subheader("Calories Burned by Workout Type")
        calories_by_workout = df.groupby('Workout_Type')['Calories_Burned'].mean().sort_values(ascending=False)
        st.bar_chart(calories_by_workout)

    with col2:
        # Heart rate by workout type
        st.subheader("Heart Rate Metrics by Workout Type")
        bpm_metrics = df.groupby('Workout_Type')[['Max_BPM', 'Avg_BPM', 'Resting_BPM']].mean()
        st.line_chart(bpm_metrics)

        # Duration by workout type
        st.subheader("Session Duration by Workout Type")
        duration_by_workout = df.groupby('Workout_Type')['Session_Duration (hours)'].mean().sort_values(ascending=False)
        st.bar_chart(duration_by_workout)

# Tab 3: Fitness Insights
with tab3:
    st.subheader("Fitness Insights")

    col1, col2 = st.columns(2)

    with col1:
        # BMI distribution
        st.subheader("BMI Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['BMI'], bins=20, kde=True, ax=ax)
        ax.axvline(bmi, color='red', linestyle='--', label=f'Your BMI: {bmi:.1f}')
        ax.set_xlabel('BMI')
        ax.set_ylabel('Count')
        ax.legend()
        st.pyplot(fig)

        # Workout frequency vs. experience
        st.subheader("Workout Frequency vs. Experience Level")
        freq_by_exp = df.groupby('Experience_Level')['Workout_Frequency (days/week)'].mean()
        st.bar_chart(freq_by_exp)

    with col2:
        # Correlation heatmap
        st.subheader("Key Metric Correlations")
        corr_cols = ['Age', 'Weight (kg)', 'Calories_Burned', 'Session_Duration (hours)',
                     'Workout_Frequency (days/week)', 'Experience_Level']
        corr = df[corr_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        # Age vs. workout metrics
        st.subheader("Age vs. Fitness Metrics")
        bins = [15, 25, 35, 45, 55, 65, 75]
        labels = [f"{bins[i]}-{bins[i + 1]}" for i in range(len(bins) - 1)]
        df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)
        age_metrics = df.groupby('Age_Group', observed=True)[['Calories_Burned', 'Session_Duration (hours)']].mean()

        # Convert to a format Streamlit can handle
        age_metrics = age_metrics.reset_index()

        # Plot using matplotlib instead
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(age_metrics['Age_Group'], age_metrics['Calories_Burned'], marker='o', label='Calories Burned')
        ax.plot(age_metrics['Age_Group'], age_metrics['Session_Duration (hours)'] * 100, marker='s',
                label='Session Duration (min)')
        ax.set_xlabel('Age Group')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# Tab 4: Recommendations
with tab4:
    st.subheader("Personalized Recommendations")

    # Recommended workout plan based on experience level
    st.subheader(f"Weekly Workout Plan for {experience_text[experience_level]} Level")

    # Generate personalized workout recommendations
    if experience_level == 1:  # Beginner
        workouts = [
            {"Day": "Monday", "Type": "Cardio", "Description": "30 min light cardio (walking, cycling)",
             "Intensity": "Low"},
            {"Day": "Tuesday", "Type": "Rest", "Description": "Active recovery or light stretching",
             "Intensity": "Very Low"},
            {"Day": "Wednesday", "Type": "Strength", "Description": "Full body workout with light weights",
             "Intensity": "Moderate"},
            {"Day": "Thursday", "Type": "Rest", "Description": "Rest day", "Intensity": "None"},
            {"Day": "Friday", "Type": "Cardio", "Description": "30 min cardio (walking, cycling)",
             "Intensity": "Low-Moderate"},
            {"Day": "Saturday", "Type": "Yoga", "Description": "Beginner yoga or stretching routine",
             "Intensity": "Low"},
            {"Day": "Sunday", "Type": "Rest", "Description": "Rest day", "Intensity": "None"}
        ]
    elif experience_level == 2:  # Intermediate
        workouts = [
            {"Day": "Monday", "Type": "Strength", "Description": "Upper body strength training",
             "Intensity": "Moderate-High"},
            {"Day": "Tuesday", "Type": "Cardio", "Description": "45 min interval training", "Intensity": "Moderate"},
            {"Day": "Wednesday", "Type": "Yoga", "Description": "Intermediate yoga flow", "Intensity": "Moderate"},
            {"Day": "Thursday", "Type": "Strength", "Description": "Lower body strength training",
             "Intensity": "Moderate-High"},
            {"Day": "Friday", "Type": "HIIT", "Description": "30 min HIIT workout", "Intensity": "High"},
            {"Day": "Saturday", "Type": "Active Recovery", "Description": "Light cardio and stretching",
             "Intensity": "Low"},
            {"Day": "Sunday", "Type": "Rest", "Description": "Rest day", "Intensity": "None"}
        ]
    else:  # Advanced
        workouts = [
            {"Day": "Monday", "Type": "Strength", "Description": "Heavy upper body + core", "Intensity": "High"},
            {"Day": "Tuesday", "Type": "HIIT", "Description": "45 min high-intensity intervals",
             "Intensity": "Very High"},
            {"Day": "Wednesday", "Type": "Strength", "Description": "Heavy lower body", "Intensity": "High"},
            {"Day": "Thursday", "Type": "Cardio", "Description": "60 min endurance training",
             "Intensity": "Moderate-High"},
            {"Day": "Friday", "Type": "Strength", "Description": "Full body circuit training", "Intensity": "High"},
            {"Day": "Saturday", "Type": "Sport/Activity", "Description": "60-90 min sport or outdoor activity",
             "Intensity": "Varies"},
            {"Day": "Sunday", "Type": "Active Recovery", "Description": "Yoga or mobility work", "Intensity": "Low"}
        ]

    # Display workout plan
    workout_df = pd.DataFrame(workouts)
    st.dataframe(workout_df, use_container_width=True)

    # Nutrition recommendations based on BMI and activity level
    st.subheader("Nutrition Recommendations")

    col1, col2 = st.columns(2)

    with col1:
        # Calorie recommendations
        st.subheader("Daily Calorie Target")

        # Basic BMR calculation (Mifflin-St Jeor Equation)
        if gender == "Male":
            bmr = 10 * weight + 6.25 * (height * 100) - 5 * age + 5
        else:
            bmr = 10 * weight + 6.25 * (height * 100) - 5 * age - 161

        # Activity multiplier
        activity_multipliers = {
            1: 1.2,  # Sedentary
            2: 1.375,  # Light exercise
            3: 1.465,  # Moderate exercise
            4: 1.55,  # Heavy exercise
            5: 1.725,  # Very heavy exercise
            6: 1.9,  # Extra heavy exercise
            7: 1.9  # Extra heavy exercise
        }

        activity_multiplier = activity_multipliers.get(workout_freq, 1.375)
        maintenance_calories = int(bmr * activity_multiplier)

        # Goal-based calories
        if bmi > 25:  # Weight loss
            calorie_target = int(maintenance_calories * 0.85)
            goal = "Weight Loss"
        elif bmi < 18.5:  # Weight gain
            calorie_target = int(maintenance_calories * 1.15)
            goal = "Weight Gain"
        else:  # Maintenance
            calorie_target = maintenance_calories
            goal = "Maintenance"

        st.metric("Daily Calorie Target", f"{calorie_target} kcal", f"Goal: {goal}")

        # Macronutrient breakdown
        st.subheader("Macronutrient Breakdown")

        if workout_type in ["Strength", "HIIT"]:
            protein_pct = 0.30
            carb_pct = 0.45
            fat_pct = 0.25
        elif workout_type in ["Cardio", "Endurance"]:
            protein_pct = 0.25
            carb_pct = 0.55
            fat_pct = 0.20
        else:
            protein_pct = 0.25
            carb_pct = 0.50
            fat_pct = 0.25

        protein_g = int((calorie_target * protein_pct) / 4)
        carb_g = int((calorie_target * carb_pct) / 4)
        fat_g = int((calorie_target * fat_pct) / 9)

        # Display macros
        cols = st.columns(3)
        cols[0].metric("Protein", f"{protein_g}g", f"{int(protein_pct * 100)}%")
        cols[1].metric("Carbs", f"{carb_g}g", f"{int(carb_pct * 100)}%")
        cols[2].metric("Fat", f"{fat_g}g", f"{int(fat_pct * 100)}%")

    with col2:
        # Hydration recommendation
        st.subheader("Hydration Target")

        # Base recommendation on weight and activity level
        base_water = weight * 0.033  # 33ml per kg of body weight
        activity_water = workout_freq * 0.5  # Additional 0.5L per workout day

        total_water = round(base_water + activity_water, 1)

        st.metric("Daily Water Intake Target", f"{total_water}L",
                  f"{round(total_water - water_intake, 1)}L from current")

        # Water intake tips
        st.markdown("""
        **Hydration Tips:**
        - Drink 500ml of water upon waking up
        - Drink 500ml 30 minutes before each workout
        - Drink 250ml every 15-20 minutes during exercise
        - Drink 500ml within 30 minutes after workout
        """)

        # Meal timing recommendation
        st.subheader("Meal Timing")

        if workout_type in ["Strength", "HIIT"]:
            st.markdown("""
            **Strength/HIIT Training Meal Plan:**
            - **Pre-workout (1-2 hours before):** Protein + complex carbs
            - **Post-workout (within 30 min):** Protein + simple carbs
            - **Regular meals:** Balanced protein, carbs, and healthy fats
            """)
        else:
            st.markdown("""
            **Cardio/Endurance Meal Plan:**
            - **Pre-workout (1-2 hours before):** Complex carbs + moderate protein
            - **During (if >60 min):** Simple carbs, electrolytes
            - **Post-workout (within 30 min):** Carbs + protein (3:1 ratio)
            - **Regular meals:** Higher carb ratio for energy
            """)

# Inside the new "Find Trainers" tab
with tab5:
    st.subheader("Find Personal Trainers Near You")

    # User location input
    location_col1, location_col2 = st.columns(2)
    with location_col1:
        city = st.text_input("City", "San Francisco")
        state = st.text_input("State", "CA")

    with location_col2:
        search_radius = st.slider("Search Radius (miles)", 1, 25, 5)
        st.write("Enter your location to find certified personal trainers nearby")

    if st.button("Find Trainers"):
        with st.spinner("Searching for trainers in your area..."):
            # Geocode the address to get coordinates
            # Note: In a production app, you'd use an actual API key
            try:
                # Simulate geocoding (in a real app you'd call an API)
                # Here we're just using mock data for San Francisco
                lat, lon = 37.7749, -122.4194

                # Create map centered on user location
                m = folium.Map(location=[lat, lon], zoom_start=13)

                # Add a marker for user location
                folium.Marker(
                    [lat, lon],
                    popup="Your Location",
                    icon=folium.Icon(color="blue", icon="home")
                ).add_to(m)

                # Simulate trainer locations nearby
                # In a real app, you'd query a service like Google Places API
                trainer_locations = [
                    {"name": "FitLife Personal Training", "lat": lat + 0.01, "lon": lon + 0.01, "rating": 4.8,
                     "exp_level": 3},
                    {"name": "CoreStrength Fitness", "lat": lat - 0.01, "lon": lon - 0.02, "rating": 4.6,
                     "exp_level": 2},
                    {"name": "Elite Training Studio", "lat": lat + 0.02, "lon": lon - 0.01, "rating": 4.9,
                     "exp_level": 3},
                    {"name": "Beginner Friendly Fitness", "lat": lat - 0.02, "lon": lon + 0.015, "rating": 4.7,
                     "exp_level": 1},
                    {"name": "Total Body Transformation", "lat": lat + 0.018, "lon": lon + 0.025, "rating": 4.5,
                     "exp_level": 2},
                ]

                # Add trainer markers
                for trainer in trainer_locations:
                    # Color based on trainer's experience level
                    colors = {1: "green", 2: "orange", 3: "red"}
                    exp_text = {1: "Beginner-focused", 2: "Intermediate", 3: "Advanced"}

                    folium.Marker(
                        [trainer["lat"], trainer["lon"]],
                        popup=f"<b>{trainer['name']}</b><br>Rating: {trainer['rating']}⭐<br>Specialty: {exp_text[trainer['exp_level']]}",
                        icon=folium.Icon(color=colors[trainer["exp_level"]], icon="user")
                    ).add_to(m)

                # Display map
                folium_static(m)

                # Display trainer list with recommendations
                st.subheader("Recommended Trainers")

                # Filter trainers based on user's experience level
                recommended_trainers = [t for t in trainer_locations if t["exp_level"] <= experience_level + 1]

                # Sort by match with user's level
                recommended_trainers.sort(key=lambda x: abs(x["exp_level"] - experience_level))

                for i, trainer in enumerate(recommended_trainers[:3]):
                    with st.container():
                        st.markdown(f"#### {i + 1}. {trainer['name']} - {trainer['rating']}⭐")
                        st.markdown(f"Specialty: {exp_text[trainer['exp_level']]} training")
                        if trainer["exp_level"] == experience_level:
                            st.markdown("✅ **Perfect match for your experience level**")
                        st.markdown("---")

            except Exception as e:
                st.error(f"Error finding trainers. Please check your location information.")

    # Add tips section
    st.subheader("Tips for Working with a Trainer")

    # Customize tips based on user's experience level
    if experience_level == 1:
        st.markdown("""
        **For Beginners:**
        - Look for trainers who specialize in foundational movements
        - Ask about their experience with newcomers to fitness
        - Ensure they focus on proper form before intensity
        - Discuss any health concerns or limitations upfront
        """)
    elif experience_level == 2:
        st.markdown("""
        **For Intermediate Athletes:**
        - Find trainers who can help break through plateaus
        - Ask about their approach to periodization
        - Look for someone who can introduce more advanced techniques
        - Discuss your specific goals (strength, endurance, etc.)
        """)
    else:
        st.markdown("""
        **For Advanced Athletes:**
        - Seek trainers with specialized certifications
        - Look for someone with experience training athletes
        - Ask about their approach to optimizing performance
        - Consider trainers who use data-driven methods
        """)

# Footer with automations info
st.markdown("---")
st.subheader("Automated Features")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Weekly Workout Reminder**
    - Sent every Monday at 9:00 AM
    - Personalized workout plan for the week
    - Adapts based on your progress and experience level
    """)

with col2:
    st.markdown("""
    **Inactivity Alert**
    - Checks daily for workout gaps
    - Sends motivational message after 3 days of inactivity
    - Includes quick workout ideas to get back on track
    """)

# Show a simulate automation section
if st.button("Simulate Weekly Workout Reminder"):
    st.success("📩 Notification sent! Weekly workout plan has been delivered.")
    st.info(
        f"Your personalized {experience_text[experience_level]}-level workout plan for this week includes {workout_freq} sessions with focus on {workout_type} training.")


