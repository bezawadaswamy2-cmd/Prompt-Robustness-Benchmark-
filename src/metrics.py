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
    robustness_results = []

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

            # Robustness Score

            robustness_scores = []

            for question_id in task_df["id"].unique():

                question_df = task_df[task_df["id"] == question_id]

                correct_count = 0

                for _, qrow in question_df.iterrows():

                    response = str(qrow["response"]).strip().lower()
                    ground_truth = str(qrow["ground_truth"]).strip().lower()

                    if ground_truth in response:
                        correct_count += 1

                robustness = (correct_count / len(question_df)) * 100

                robustness_scores.append(robustness)

            average_robustness = round(
                sum(robustness_scores) / len(robustness_scores), 2
            )

            robustness_results.append({
                "model": model,
                "task": task,
                "robustness_score": average_robustness
            })

    metrics_df = pd.DataFrame(results)

    robustness_df = pd.DataFrame(robustness_results)

    metrics_df = metrics_df.merge(
        robustness_df,
        on=["model", "task"]
    )

    output_file = RESULTS_DIR / "metrics.csv"
    metrics_df.to_csv(output_file, index=False)

    print("\nMetrics Summary")
    print(metrics_df)

    return metrics_df


if __name__ == "__main__":
    calculate_metrics()