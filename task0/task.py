import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import warnings

# Load the data
data = {
    'Hours': [2.5, 5.1, 3.2, 8.5, 3.5, 1.5, 9.2, 5.5, 8.3, 2.7, 7.7, 5.9, 4.5, 3.3, 1.1, 8.9, 2.5, 1.9, 6.1, 7.4, 2.7, 4.8, 3.8, 6.9, 7.8],
    'Scores': [21, 47, 27, 75, 30, 20, 88, 60, 81, 25, 85, 62, 41, 42, 17, 95, 30, 24, 67, 69, 30, 54, 35, 76, 86]
}
df = pd.DataFrame(data)

# Split the data into features (X) and target variable (y)
X = df[['Hours']]
y = df['Scores']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Suppress the warning
warnings.filterwarnings('ignore', category=UserWarning)

# Create and fit the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make a prediction for a new input (9.25 hours/day)
new_input = pd.DataFrame([[9.25]], columns=['Hours'])
predicted_score = model.predict(new_input)
print(f"Predicted score for 9.25 hours/day: {predicted_score[0]:.2f}")

# Calculate the predicted percentage
predicted_percentage = (predicted_score[0] / 100) * 100
print(f"Predicted percentage for 9.25 hours/day: {predicted_percentage:.2f}%")