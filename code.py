import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

# Load the dataset
data = pd.read_csv("Insurance claims data.csv")

# Plot the distribution of the target variable 'claim_status'
plt.figure(figsize=(8, 5))
sns.countplot(x='claim_status', data=data)
plt.title('Distribution of Claim Status')
plt.show()

# Resample the minority class to handle imbalance
majority = data[data.claim_status == 0]
minority = data[data.claim_status == 1]

minority_oversampled = resample(minority,
                                replace=True,
                                n_samples=len(majority),
                                random_state=42)

oversampled_data = pd.concat([majority, minority_oversampled])

# Drop the 'policy_id' column
oversampled_data = oversampled_data.drop('policy_id', axis=1)

# Encode categorical variables
le = LabelEncoder()
X_oversampled = oversampled_data.drop('claim_status', axis=1)
y_oversampled = oversampled_data['claim_status']

X_oversampled_encoded = X_oversampled.apply(lambda col: le.fit_transform(col) if col.dtype == 'object' else col)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_oversampled_encoded, y_oversampled, test_size=0.3, random_state=42)

# Random forest model training
rf_model_oversampled = RandomForestClassifier(random_state=42)
rf_model_oversampled.fit(X_train, y_train)

# Prediction
y_pred = rf_model_oversampled.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = rf_model_oversampled.feature_importances_
features_df = pd.DataFrame({'Feature': X_oversampled.columns, 'Importance': feature_importance})
features_df = features_df.sort_values(by='Importance', ascending=False)
print(features_df.head(10))

# Predicting on the original data (without oversampling)
original_encoded = data.drop('policy_id', axis=1).apply(
    lambda col: le.fit_transform(col) if col.dtype == 'object' else col)

# Predict using the trained oversampled model
original_encoded_predictions = rf_model_oversampled.predict(original_encoded.drop('claim_status', axis=1))

# Comparison of predictions
comparison_df = pd.DataFrame({
    'Actual': original_encoded['claim_status'],
    'Predicted': original_encoded_predictions
})
print(comparison_df.head(10))

# Classification pie chart
correctly_classified = (comparison_df['Actual'] == comparison_df['Predicted']).sum()
incorrectly_classified = (comparison_df['Actual'] != comparison_df['Predicted']).sum()
classification_counts = [correctly_classified, incorrectly_classified]
labels = ['Correctly Classified', 'Misclassified']

plt.figure(figsize=(8, 8))
plt.pie(classification_counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#4CAF50', '#FF5733'])
plt.title('Classification Accuracy')
plt.show()
