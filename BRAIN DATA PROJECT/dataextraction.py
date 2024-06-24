import pandas as pd
val = 2
eeg_data = pd.read_csv(f'BCICIV_2a_{val}.csv')
print((eeg_data))

import pandas as pd
from sklearn.model_selection import train_test_split

# Load the BCIIV2a data from CSV
data = pd.read_csv("BCI_ kompetisisi_IV_2a.csv")  # Replace with your actual filename

# Preprocess the data (replace with your specific preprocessing steps)
# - Filter noise
# - Extract features (e.g., time-domain, frequency-domain, CSP)
# Preprocessed data will be a numpy array (X) and labels as another array (y)

# Define few-shot learning approach (replace with your chosen algorithm)
# This example uses a placeholder function for demonstration
def few_shot_learner(X_train, y_train, X_test, y_test):
  # Implement your specific few-shot learning model training and testing logic here
  # This function should take training data (X_train, y_train) and testing data (X_test, y_test)
  # and return the predicted labels for the testing data
  predicted_labels = None  # Replace with actual predictions from your model
  return predicted_labels

# Split data into training and testing sets (modify for few-shot learning)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Further split training data into few-shot training and validation sets (if applicable)
# This step depends on your chosen few-shot learning algorithm

# Train the few-shot learning model
predicted_labels = few_shot_learner(X_train_fewshot, y_train_fewshot, X_test, y_test)

# Evaluate the model's performance (accuracy, precision, recall, F1 score)

# Print the results
