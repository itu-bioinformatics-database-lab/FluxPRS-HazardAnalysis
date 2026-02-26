from sklearn.metrics import roc_auc_score, classification_report
import pandas as pd

def evaluate_model(model, x_test, y_test, name, x_train=None, y_train=None):
    """
    Evaluate the model performance.

    Returns:
    dict: Dictionary containing evaluation metrics
    """
    # Calculate predictions and probabilities
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)[:, 1]

    # Calculate metrics
    results = {
        'auc_roc': roc_auc_score(y_test, y_pred_proba),
        'classification_report': classification_report(y_test, y_pred),
        'feature_importance': pd.DataFrame({
            'feature': x_test.columns,
            'coefficient': model.coef_[0]
        }).sort_values('coefficient', key=abs, ascending=False)
    }

    # If training metabolitics_data is provided, check for overfitting
    if x_train is not None and y_train is not None:
        train_pred_proba = model.predict_proba(x_train)[:, 1]
        results['train_auc_roc'] = roc_auc_score(y_train, train_pred_proba)
    print(f"Evaluation Results for {name}")
    print(f"Test AUC-ROC: {results['auc_roc']:.3f}")
    print("\nTop important features:")
    print(results['feature_importance'].head())

    return results