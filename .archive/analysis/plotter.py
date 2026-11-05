import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.calibration import CalibrationDisplay
import numpy as np

def plot_risk_distribution(risk_scores,
                           y_true,
                           score_type,
                           ):
    """
    Plot the distribution of risk scores for cases and controls.

    Parameters:
    risk_scores (numpy.ndarray): Calculated risk scores
    y_true (pandas.Series): True labels
    title (str): Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.hist(risk_scores[y_true == 0], bins=30, alpha=0.5, label='Control', density=True)
    plt.hist(risk_scores[y_true == 1], bins=30, alpha=0.5, label='AD', density=True)
    plt.xlabel('Risk Score')
    plt.ylabel('Density')
    plt.title(f"Distribution of Risk Scores ({score_type})")
    plt.legend()
    plt.show()

def plot_risk_distribution2(risk_scores,
                                 y_true,
                                 score_type,
                                 ):
    """
    Plot risk scores distribution for AD and control groups.

    Parameters:
    risk_scores (numpy.ndarray): Calculated risk scores
    y_true (numpy.ndarray): True labels (0 for control, 1 for AD)
    title (str): Plot title
    """
    # Create figure
    plt.figure(figsize=(10, 6))

    # Convert risk scores to 0-100 scale
    risk_scores_scaled = risk_scores * 100

    # Create scatter plot for each group with jittered y positions
    # Control group (0)
    control_scores = risk_scores_scaled[y_true == 0]
    control_y = np.zeros(len(control_scores)) + np.random.normal(0, 0.08, len(control_scores))
    plt.scatter(control_scores, control_y, color='blue', alpha=0.4, label='Control', s=30)

    # AD group (1)
    ad_scores = risk_scores_scaled[y_true == 1]
    ad_y = np.ones(len(ad_scores)) + np.random.normal(0, 0.08, len(ad_scores))
    plt.scatter(ad_scores, ad_y, color='red', alpha=0.4, label='AD', s=30)

    # Customize the plot
    plt.xlabel('Risk Score')
    plt.ylabel('Diagnosis')
    plt.title(f"Distribution of Risk Scores by Group ({score_type})")

    # Set y-axis labels
    plt.yticks([0, 1], ['Control', 'AD'])

    # Add legend
    plt.legend()
    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # Add grid lines for x-axis only
    plt.grid(True, axis='x', alpha=0.3)
    # Set reasonable x-axis limits
    plt.xlim(-5, 105)
    # Adjust y-axis limits to show all points clearly
    plt.ylim(-0.5, 1.5)
    plt.tight_layout()
    plt.show()


def plot_calibration_curve(
        model,
        X_test,
        y_test,
        n_bins=10,
        model_name="Model",
        figsize=(8, 8)
):
    """
    Plot calibration curve and histogram for a fitted model.

    Parameters:
    - model: fitted classifier with predict_proba or decision_function
    - X_test: test features
    - y_test: true labels
    - n_bins: number of bins for calibration curve (default=10)
    - model_name: title/name for the plots (default="Model")
    - figsize: size of the figure (default=(8, 8))
    """

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 1)  # 2/3 for calibration curve, 1/3 for histogram
    colors = plt.get_cmap("Dark2")

    # Calibration plot
    ax_calibration_curve = fig.add_subplot(gs[:2, 0])
    display = CalibrationDisplay.from_estimator(
        model,
        X_test,
        y_test,
        n_bins=n_bins,
        ax=ax_calibration_curve,
        color=colors(0),
        name=model_name,
    )

    ax_calibration_curve.grid()
    ax_calibration_curve.set_title(f"Calibration plot: {model_name}")

    # Histogram of predicted probabilities
    ax_hist = fig.add_subplot(gs[2, 0])
    ax_hist.hist(
        display.y_prob,
        range=(0, 1),
        bins=n_bins,
        color=colors(0),
    )
    ax_hist.set(
        title=f"{model_name} - Predicted Probabilities Histogram",
        xlabel="Mean predicted probability",
        ylabel="Count",
    )

    plt.tight_layout()
    plt.show()