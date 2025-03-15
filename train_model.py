import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

print("Loading data...")
# Load the dataset
df = pd.read_csv('gym_members_exercise_tracking.csv')

print("Preparing features...")
# Feature selection for predictive model
features = ['Age', 'Weight (kg)', 'Height (m)', 'Avg_BPM', 'Session_Duration (hours)',
            'Calories_Burned', 'Workout_Frequency (days/week)', 'BMI', 'Fat_Percentage']
X = df[features]
y = df['Experience_Level']

# Handling categorical features
X_with_dummies = pd.get_dummies(df[features + ['Workout_Type', 'Gender']])

print("Splitting data...")
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_with_dummies, y, test_size=0.25, random_state=42)

print("Training model...")
# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Model Accuracy:", model.score(X_test, y_test))

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X_with_dummies.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("Top 10 important features:")
print(feature_importance.head(10))

print("Saving model...")
# Save the model
with open('fitness_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved to fitness_model.pkl")