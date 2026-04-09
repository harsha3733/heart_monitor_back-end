import pandas as pd
import numpy as np

# -------------------------------
# STEP 1: COLUMN NAMES
# -------------------------------
columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target"
]

# -------------------------------
# STEP 2: LOAD UCI DATASETS
# -------------------------------
df1 = pd.read_csv("processed.cleveland.data", names=columns)
df2 = pd.read_csv("processed.hungarian.data", names=columns)
df3 = pd.read_csv("processed.switzerland.data", names=columns)
df4 = pd.read_csv("processed.va.data", names=columns)

dfs = [df1, df2, df3, df4]

# -------------------------------
# STEP 3: CLEAN DATA
# -------------------------------
cleaned_dfs = []

for df in dfs:
    df.replace("?", np.nan, inplace=True)
    df = df.apply(pd.to_numeric)
    df.fillna(df.mean(), inplace=True)
    cleaned_dfs.append(df)

df = pd.concat(cleaned_dfs, ignore_index=True)

# -------------------------------
# STEP 4: FIX TARGET
# -------------------------------
df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

# -------------------------------
# STEP 5: SELECT BASE FEATURES
# -------------------------------
df = df[[
    "age",
    "sex",
    "trestbps",
    "chol",
    "thalach",
    "target"
]]

df.rename(columns={
    "age": "Age",
    "sex": "Sex",
    "trestbps": "Systolic_BP",
    "chol": "Cholesterol",
    "thalach": "Heart_Rate",
    "target": "Heart_Disease"
}, inplace=True)

# -------------------------------
# STEP 6: LOAD CARDIO DATASET
# -------------------------------
try:
    cardio = pd.read_csv("cardio_train.csv", sep=";")
except:
    cardio = pd.read_excel("cardio_train.xlsx")

# -------------------------------
# STEP 7: ADD STATIC FEATURES
# -------------------------------

# BMI
if "weight" in cardio.columns and "height" in cardio.columns:
    cardio["BMI"] = cardio["weight"] / ((cardio["height"] / 100) ** 2)
    bmi_values = cardio["BMI"].dropna().values
else:
    bmi_values = np.random.uniform(18, 35, size=len(df))

df["BMI"] = np.random.choice(bmi_values, size=len(df), replace=True)

# Smoking (0–2)
df["Smoking"] = np.random.randint(0, 3, size=len(df))

# Alcohol
if "alco" in cardio.columns:
    df["Alcohol"] = np.random.choice(cardio["alco"].values, size=len(df), replace=True)
else:
    df["Alcohol"] = np.random.randint(0, 3, size=len(df))

# Diabetes
df["Diabetes"] = np.random.randint(0, 2, size=len(df))

# -------------------------------
# STEP 8: ADD REAL-TIME FEATURES
# -------------------------------
df["SpO2"] = np.random.uniform(94, 100, size=len(df))
df["Temperature"] = np.random.uniform(36, 38, size=len(df))
df["Activity_Level"] = np.random.randint(0, 3, size=len(df))

# Step Count (optional but useful)
df["Step_Count"] = np.random.randint(0, 5000, size=len(df))

# -------------------------------
# STEP 9: DERIVED FEATURES
# -------------------------------

# Diastolic BP (simulate realistic relation)
df["Diastolic_BP"] = df["Systolic_BP"] - np.random.randint(30, 60, size=len(df))

# Pulse Pressure
df["Pulse_Pressure"] = df["Systolic_BP"] - df["Diastolic_BP"]

# HRV (Heart Rate Variability - simulated)
df["HRV"] = np.random.uniform(20, 100, size=len(df))

# Temperature change (relative variation)
df["Temp_Change"] = np.random.uniform(-0.5, 0.5, size=len(df))

# -------------------------------
# STEP 10: AUGMENT DATA TO 2700
# -------------------------------
def augment_data(df, target_size):
    augmented = df.copy()

    while len(augmented) < target_size:
        sample = df.sample(n=len(df), replace=True).copy()

        # Add noise
        sample["Heart_Rate"] += np.random.normal(0, 2, len(sample))
        sample["Systolic_BP"] += np.random.normal(0, 3, len(sample))
        sample["Cholesterol"] += np.random.normal(0, 5, len(sample))

        augmented = pd.concat([augmented, sample], ignore_index=True)

    return augmented.iloc[:target_size]

df = augment_data(df, 2700)

# -------------------------------
# STEP 11: RANGE VALIDATION
# -------------------------------
df["Heart_Rate"] = df["Heart_Rate"].clip(50, 180)
df["Systolic_BP"] = df["Systolic_BP"].clip(90, 200)
df["Diastolic_BP"] = df["Diastolic_BP"].clip(60, 120)
df["Cholesterol"] = df["Cholesterol"].clip(100, 400)
df["SpO2"] = df["SpO2"].clip(85, 100)
df["Temperature"] = df["Temperature"].clip(35, 40)
df["BMI"] = df["BMI"].clip(15, 40)
df["HRV"] = df["HRV"].clip(10, 150)

# -------------------------------
# STEP 12: FINAL COLUMN ORDER
# -------------------------------
df = df[[
    "Age",
    "Sex",
    "Smoking",
    "Diabetes",
    "Systolic_BP",
    "Diastolic_BP",
    "Pulse_Pressure",
    "Cholesterol",
    "BMI",
    "Alcohol",
    "Heart_Rate",
    "HRV",
    "SpO2",
    "Temperature",
    "Temp_Change",
    "Activity_Level",
    "Step_Count",
    "Heart_Disease"
]]

# -------------------------------
# STEP 13: SHUFFLE
# -------------------------------
df = df.sample(frac=1).reset_index(drop=True)

# -------------------------------
# STEP 14: SAVE DATASET
# -------------------------------
df.to_csv("final_dataset_2700_enhanced.csv", index=False)

print("✅ Enhanced dataset created!")
print("Shape:", df.shape)
print(df.head())