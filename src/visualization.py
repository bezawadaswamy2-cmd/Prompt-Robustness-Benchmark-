import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import RESULTS_DIR


def create_all_charts():

    metrics_file = RESULTS_DIR / "metrics.csv"

    if not metrics_file.exists():
        raise FileNotFoundError(metrics_file)

    df = pd.read_csv(metrics_file)

    charts_dir = RESULTS_DIR / "charts"
    charts_dir.mkdir(exist_ok=True)

    # -------------------------------------------------------
    # Accuracy
    # -------------------------------------------------------

    plt.figure(figsize=(8,5))

    plt.bar(df["model"], df["accuracy"])

    plt.title("Accuracy by Model")
    plt.xlabel("Model")
    plt.ylabel("Accuracy (%)")

    plt.tight_layout()

    plt.savefig(charts_dir / "accuracy.png")
    plt.close()

    # -------------------------------------------------------
    # Robustness
    # -------------------------------------------------------

    plt.figure(figsize=(8,5))

    plt.bar(df["model"], df["robustness_score"])

    plt.title("Robustness Score")
    plt.xlabel("Model")
    plt.ylabel("Robustness (%)")

    plt.tight_layout()

    plt.savefig(charts_dir / "robustness.png")
    plt.close()

    # -------------------------------------------------------
    # Consistency
    # -------------------------------------------------------

    plt.figure(figsize=(8,5))

    plt.bar(df["model"], df["consistency_score"])

    plt.title("Consistency Score")
    plt.xlabel("Model")
    plt.ylabel("Consistency (%)")

    plt.tight_layout()

    plt.savefig(charts_dir / "consistency.png")
    plt.close()

    # -------------------------------------------------------
    # Response Time
    # -------------------------------------------------------

    plt.figure(figsize=(8,5))

    plt.bar(df["model"], df["avg_response_time"])

    plt.title("Average Response Time")
    plt.xlabel("Model")
    plt.ylabel("Seconds")

    plt.tight_layout()

    plt.savefig(charts_dir / "response_time.png")
    plt.close()

    # -------------------------------------------------------
    # Response Length
    # -------------------------------------------------------

    plt.figure(figsize=(8,5))

    plt.bar(df["model"], df["avg_response_length"])

    plt.title("Average Response Length")
    plt.xlabel("Model")
    plt.ylabel("Words")

    plt.tight_layout()

    plt.savefig(charts_dir / "response_length.png")
    plt.close()

    # -------------------------------------------------------
    # Overall Score
    # -------------------------------------------------------

    df["overall_score"] = (
        df["accuracy"] * 0.40
        + df["robustness_score"] * 0.30
        + df["consistency_score"] * 0.20
        + (100 - df["avg_response_time"] * 10) * 0.05
        + (100 - abs(df["avg_response_length"] - 20)) * 0.05
    )

    plt.figure(figsize=(8,5))

    plt.bar(df["model"], df["overall_score"])

    plt.title("Overall Model Score")
    plt.xlabel("Model")
    plt.ylabel("Score")

    plt.tight_layout()

    plt.savefig(charts_dir / "overall_score.png")
    plt.close()

    df.to_csv(metrics_file, index=False)

    print("Charts saved to:", charts_dir)


if __name__ == "__main__":
    create_all_charts()