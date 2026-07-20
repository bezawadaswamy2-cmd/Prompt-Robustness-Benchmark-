import pandas as pd
import matplotlib.pyplot as plt

from src.config import RESULTS_DIR


def create_accuracy_chart():

    metrics_file = RESULTS_DIR / "metrics.csv"

    df = pd.read_csv(metrics_file)

    plt.figure(figsize=(8, 5))

    plt.bar(df["model"], df["accuracy"])

    plt.title("Model Accuracy Comparison")
    plt.xlabel("Model")
    plt.ylabel("Accuracy (%)")

    plt.ylim(0, 100)

    output = RESULTS_DIR / "accuracy_chart.png"

    plt.tight_layout()
    plt.savefig(output)
    plt.close()

    print(f"Chart saved to: {output}")