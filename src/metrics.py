import pandas as pd

from src.config import RESULTS_DIR


def calculate_metrics():

    benchmark_file = RESULTS_DIR / "benchmark_results.csv"

    df = pd.read_csv(benchmark_file)

    results = []

    models = df["model"].unique()

    for model in models:

        model_df = df[df["model"] == model]

        tasks = model_df["task"].unique()

        for task in tasks:

            task_df = model_df[model_df["task"] == task]

            correct = 0

        for _, row in task_df.iterrows():
            response = str(row["response"]).strip().lower()
            ground_truth = str(row["ground_truth"]).strip().lower()

            if ground_truth in response:
                correct += 1

            total = len(task_df)

            accuracy = (correct / total) * 100 if total > 0 else 0

        results.append({
                "model": model,
                "task": task,
                "correct": correct,
                "total": total,
                "accuracy": round(accuracy, 2)
            })

    metrics_df = pd.DataFrame(results)

    output_file = RESULTS_DIR / "metrics.csv"

    metrics_df.to_csv(output_file, index=False)

    print(metrics_df)

    return metrics_df