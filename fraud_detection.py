import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

path =os.path.join("dataset","creditcard.csv")
df=pd.read_csv(path)

print("Dataset shape: ",df.shape)
print("Class distribution:\n", df['Class'].value_counts())

x=df.drop('Class', axis=1)
y=df['Class']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

smote=SMOTE(random_state=42)
X_train_smote,y_train_smote=smote.fit_resample(X_train, y_train)

print("After SMOTE, class distribution in training set:\n", y_train_smote.value_counts())

model=RandomForestClassifier(random_state=42)
model.fit(X_train_smote,y_train_smote)

y_pred=model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
feature_columns = x.columns.tolist()


cm= confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))

sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=['Non-Fraud','Fraud'],yticklabels=['Non-Fraud','Fraud'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('visuals/confusion_matrix.png')

plt.clf()

def cli_predict():
    print("\nEnter feature values separated by commas for prediction:")
    input_str = input("Features: ")

    try:
        input_list = [float(val) for val in input_str.split(",")]
        if len(input_list) != len(feature_columns):
            raise ValueError(f"Expected {len(feature_columns)} features, got {len(input_list)}.")
        input_df = pd.DataFrame([input_list], columns=feature_columns)
        
        pred = model.predict(input_df)
        print("Prediction:", "Fraud" if pred[0] == 1 else "Not Fraud")
    except Exception as e:
        print("Error:", e)
if __name__ == "__main__":
    cli_predict()
    print("Model training and evaluation completed.")
    print("You can now use the model to make predictions.")