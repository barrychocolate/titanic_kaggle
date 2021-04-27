import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# load training data
train_data = pd.read_csv("train.csv")
train_data.head()

# Load test data
test_data = pd.read_csv("test.csv")
test_data.head()

# Percentage of females that survived
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)

# Percentage of males that survived
men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)

# Random forest

# Define the model
y = train_data["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

# Fit the model
model.fit(X, y)

# Make predictions
predictions = model.predict(X_test)

# Validate our model

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
