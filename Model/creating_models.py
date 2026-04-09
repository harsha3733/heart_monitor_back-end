# ===============================
# IMPORTS (ALL AT TOP)
# ===============================
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

from xgboost import XGBClassifier

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("../datasets/final_dataset_2700_enhanced.csv")

X = df.drop("Heart_Disease", axis=1)
y = df["Heart_Disease"]
   

# ===============================
# PREPROCESSING
# ===============================
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


# ===============================
# MODEL TRAINING
# ===============================

# 🔹 XGBoost Model
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=2.0,
    random_state=42,
    eval_metric='logloss'
)

xgb.fit(X_train, y_train)


# 🔹 MLP Model
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    alpha=0.01,
    learning_rate_init=0.001,
    max_iter=800,
    random_state=42
)

mlp.fit(X_train, y_train)


# ===============================
# PREDICTIONS
# ===============================
xgb_pred = xgb.predict(X_test)
mlp_pred = mlp.predict(X_test)

# Ensemble (MXBoost)
final_pred = np.where(xgb_pred + mlp_pred >= 1, 1, 0)


# ===============================
# EVALUATION
# ===============================
print("Train Accuracy:", xgb.score(X_train, y_train))

print("🔹 XGBoost Accuracy:", accuracy_score(y_test, xgb_pred))
print("🔹 MLP Accuracy:", accuracy_score(y_test, mlp_pred))
print("🔥 MXBoost Accuracy:", accuracy_score(y_test, final_pred))

print("\n🔥 F1 Score:", f1_score(y_test, final_pred))

print("\n📊 Classification Report:\n")
print(classification_report(y_test, final_pred))



# ===============================
# FEATURE IMPORTANCE (XGBoost)
# ===============================
importances = xgb.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.title("Feature Importance (XGBoost)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()


# ===============================
# SAVE MODELS
# ===============================
with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb, f)
with open('mlp_model.pkl', 'wb') as f:
    pickle.dump(mlp, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\n✅ Models saved successfully!")
