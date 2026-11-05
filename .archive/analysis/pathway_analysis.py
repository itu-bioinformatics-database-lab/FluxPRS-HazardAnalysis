import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score

def analyze_pathway(pathway_file_path):
    df = pd.read_csv(pathway_file_path)
    df['Diagnosis'] = (df['Diagnosis'] != 'Control').astype(int)

    # Drop non-numeric or non-informative columns
    X = df.drop(columns=['Sample ID', 'Diagnosis'])  # Assuming 'Diagnosis' is the target
    y = df['Diagnosis']  # Binary or numerical outcome

    # Optional: convert categorical variables (like Gender/Race)
    X = pd.get_dummies(X, drop_first=True)

    # Fill missing values (mean imputation example)
    X.fillna(X.mean(), inplace=True)

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize LassoCV
    lasso = LassoCV(cv=5, random_state=42)

    # Fit model
    lasso.fit(X_scaled, y)

    # Get best alpha
    print(f"Best alpha: {lasso.alpha_}")

    # Get coefficients
    coef = pd.Series(lasso.coef_, index=X.columns)

    # Print non-zero coefficients (selected features)
    print("Selected features:")
    print(coef[coef != 0])

    coef[coef != 0].sort_values().plot(kind="barh", figsize=(10, 8))
    plt.title("Lasso Selected Features")
    plt.show()

    y_pred = lasso.predict(X_scaled)
    y_pred_binary = (y_pred > 0.5).astype(int)  # Threshold for binary

    print("Accuracy:", accuracy_score(y, y_pred_binary))
    print("AUC:", roc_auc_score(y, y_pred))
