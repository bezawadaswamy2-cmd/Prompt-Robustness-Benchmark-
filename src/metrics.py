import pandas as pd

from src.config import RESULTS_DIR


def calculate_metrics():
    """
    Calculate accuracy for each model and task.
    Saves the results to results/metrics.csv
    """

    benchmark_file = RESULTS_DIR / "benchmark_results.csv"

    if not benchmark_file.exists():
        raise FileNotFoundError(
            f"Benchmark file not found: {benchmark_file}"
        )

    df = pd.read_csv(benchmark_file)

    results = []

    # Loop through each model
    for model in df["model"].unique():

        model_df = df[df["model"] == model]

        # Loop through each task
        for task in model_df["task"].unique():

            task_df = model_df[model_df["task"] == task]

            correct = 0
            response_lengths = []
            response_times = []

            # Check every response
            for _, row in task_df.iterrows():

                response = str(row["response"]).strip().lower()
                ground_truth = str(row["ground_truth"]).strip().lower()

                response_length = len(response.split())
                response_lengths.append(response_length)

                response_time = row["response_time"]
                response_times.append(response_time)

                # Simple contains-match evaluation
                if ground_truth in response:
                    correct += 1

            average_response_length = round(sum(response_lengths) / len(response_lengths), 2)
            total = len(task_df)

            average_response_time = round(
                sum(response_times) / len(response_times), 3
)

            accuracy = (correct / total) * 100 if total > 0 else 0

            results.append({
                "model": model,
                "task": task,
                "correct": correct,
                "total": total,
                "accuracy": round(accuracy, 2),
                "avg_response_length": average_response_length,
                "avg_response_time": average_response_time
            })

    metrics_df = pd.DataFrame(results)

    output_file = RESULTS_DIR / "metrics.csv"
    metrics_df.to_csv(output_file, index=False)

    print("\nMetrics Summary")
    print(metrics_df)

    return metrics_df


if __name__ == "__main__":
    calculate_metrics()