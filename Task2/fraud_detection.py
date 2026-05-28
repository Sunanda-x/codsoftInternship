import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load datasets
train_data = pd.read_csv("fraudTrain.csv")
test_data = pd.read_csv("fraudTest.csv")

# Drop unnecessary columns
drop_cols = ["Unnamed: 0", "trans_num", "first", "last", "street", "city", "state", "zip", "job", "dob"]
train_data = train_data.drop(columns=[c for c in drop_cols if c in train_data.columns], errors="ignore")
test_data = test_data.drop(columns=[c for c in drop_cols if c in test_data.columns], errors="ignore")

# Convert transaction time into features
for df in [train_data, test_data]:
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day"] = df["trans_date_trans_time"].dt.day
    df["month"] = df["trans_date_trans_time"].dt.month
    df["weekday"] = df["trans_date_trans_time"].dt.weekday

train_data = train_data.drop(columns=["trans_date_trans_time"], errors="ignore")
test_data = test_data.drop(columns=["trans_date_trans_time"], errors="ignore")

# Separate target
y_train = train_data["is_fraud"]
y_test = test_data["is_fraud"]
X_train = train_data.drop("is_fraud", axis=1)
X_test = test_data.drop("is_fraud", axis=1)

# One-hot encode categorical columns
combined = pd.concat([X_train, X_test], axis=0)
combined = pd.get_dummies(combined, drop_first=True)

# Split back
X_train = combined.iloc[:len(X_train), :].copy()
X_test = combined.iloc[len(X_train):, :].copy()

# Scale features
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=20, class_weight="balanced", random_state=42, n_jobs=-1)
}

for name, model in models.items():
    print(f"\n===== {name} =====")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))