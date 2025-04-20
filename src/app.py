from utils import db_connect
engine = db_connect()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib


df = pd.read_csv("Ejercicio_RANDOM_FOREST.csv")


cols_with_invalid_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_invalid_zeros] = df[cols_with_invalid_zeros].replace(0, np.nan)


for col in cols_with_invalid_zeros:
    df[col] = df[col].fillna(df[col].median())

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42, 
                                                    stratify=y)

xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_clf.fit(X_train, y_train)


y_pred = xgb_clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=xgb_clf.classes_).plot(cmap="Blues")
plt.title("Matriz de Confusión - XGBoost")
plt.show()

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.2],
    'scale_pos_weight': [1, 2],
    'min_child_weight': [1, 5]
}

xgb_tuned = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

grid_search = GridSearchCV(estimator=xgb_tuned,
                           param_grid=param_grid,
                           scoring='f1',
                           cv=3,
                           n_jobs=-1,
                           verbose=2)

grid_search.fit(X_train, y_train)

print("Mejores parámetros encontrados:", grid_search.best_params_)
xgb_final = grid_search.best_estimator_

y_pred_final = xgb_final.predict(X_test)
print("Accuracy final:", accuracy_score(y_test, y_pred_final))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred_final))

cm_final = confusion_matrix(y_test, y_pred_final)
ConfusionMatrixDisplay(cm_final, display_labels=xgb_final.classes_).plot(cmap="Blues")
plt.title("Matriz de Confusión - XGBoost F1 Optimizado")
plt.show()

joblib.dump(xgb_final, "modelo_xgboost_diabetes.pkl")
print("Modelo XGBoost guardado como modelo_xgboost_diabetes.pkl")
