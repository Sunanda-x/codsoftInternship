# Task 3: Customer Churn Prediction
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score

# 1. Load dataset (make sure churn.csv is in the same folder)
data = pd.read_csv("Churn.csv")


# 2. Preprocess
# Encode categorical columns
# Encode categorical variables
for col in data.select_dtypes(include=['object', 'string']).columns:
    data[col] = LabelEncoder().fit_transform(data[col])

# Features and target
X = data.drop("Exited", axis=1)   # target column is Exited
y = data["Exited"]


# Scale numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Gradient Boosting": GradientBoostingClassifier()
}

# 5. Evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name} Results:")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))

# 6. Save predictions from best model (Random Forest here)
best_model = models["Random Forest"]
y_pred_final = best_model.predict(X_test)
pd.DataFrame(y_pred_final, columns=["Churn_Prediction"]).to_csv("task3_predictions.csv", index=False)
print("\nPredictions saved to task3_predictions.csv")
