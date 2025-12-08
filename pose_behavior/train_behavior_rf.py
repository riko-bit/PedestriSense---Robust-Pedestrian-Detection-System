# train_behavior_rf.py
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

dataset_file = "behavior_dataset/behavior_dataset.pkl"
with open(dataset_file, "rb") as f:
    data = pickle.load(f)

X = data['X']
y = data['y']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(clf, "behavior_rf_model.pkl")
print("Model saved as behavior_rf_model.pkl")
