import os
import kagglehub
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

path = kagglehub.dataset_download("ankanhore545/dropout-or-academic-success")
print("Path to dataset files:", path)
print(os.listdir(path))

df = pd.read_csv(os.path.join(path, "Dropout_Academic Success - Sheet1.csv"))

print(df.head())
print(df.shape)
print(df.columns.tolist())
print(df.info())
print(df.isnull().sum())
print(df["Target"].value_counts())

print(pd.crosstab(df["Debtor"], df["Target"]))
print(pd.crosstab(df["Marital status"], df["Target"]))

target_map = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}
df["Target_numeric"] = df["Target"].map(target_map)

numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(9, 9))
sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0)
plt.xticks(fontsize=6, rotation=90)
plt.yticks(fontsize=6)
plt.tight_layout()
plt.show()

correlations = numeric_df.corr()["Target_numeric"].sort_values(ascending=False)
print(correlations)

pd.set_option("display.max_columns", None)
print(df.describe())

plt.figure(figsize=(10, 5))
sns.boxplot(x=df["Age at enrollment"])
plt.tight_layout()
plt.show()

df['Grade_change'] = df['Curricular units 2nd sem (grade)'] - df['Curricular units 1st sem (grade)']
df['Approval_rate_1st'] = df['Curricular units 1st sem (approved)'] / df['Curricular units 1st sem (enrolled)'].replace(0, 1)
df['Approval_rate_2nd'] = df['Curricular units 2nd sem (approved)'] / df['Curricular units 2nd sem (enrolled)'].replace(0, 1)

print(df[['Grade_change', 'Approval_rate_1st', 'Approval_rate_2nd']].head())

leakage_columns = [
    "Target", "Target_numeric",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Grade_change", "Approval_rate_2nd"
]

X = df.drop(columns=leakage_columns)
y = df["Target_numeric"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

importances = pd.Series(model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)
print(importances)

plt.figure(figsize=(10, 8))
importances.head(15).plot(kind='barh')
plt.title('Top 15 Feature Importances')
plt.xlabel('Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42),
    "LightGBM": LGBMClassifier(random_state=42)
}

for name, clf in models.items():
    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    print(f"{name} 5-Fold CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")