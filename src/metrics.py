import pandas as pd

from src.config import RESULTS_DIR


def calculate_metrics():

    benchmark_file = RESULTS_DIR / "benchmark_results.csv"

    if not benchmark_file.exists():
        raise FileNotFoundError(f"{benchmark_file} not found.")

    df = pd.read_csv(benchmark_file)

    metrics = []

    for model in df["model"].unique():

        model_df = df[df["model"] == model]

        for task in model_df["task"].unique():

            task_df = model_df[model_df["task"] == task]

            total = len(task_df)

            correct = 0
            response_lengths = []
            response_times = []

            # ---------------- Accuracy ----------------
            for _, row in task_df.iterrows():

                response = str(row["response"]).strip().lower()
                ground_truth = str(row["ground_truth"]).strip().lower()

                if ground_truth in response:
                    correct += 1

                response_lengths.append(len(response.split()))
                response_times.append(row["response_time"])

            accuracy = round((correct / total) * 100, 2)

            avg_response_length = round(
                sum(response_lengths) / len(response_lengths), 2
            )

            avg_response_time = round(
                sum(response_times) / len(response_times), 3
            )

            # ---------------- Robustness ----------------
            robustness_scores = []

            for qid in task_df["id"].unique():

                qdf = task_df[task_df["id"] == qid]

                correct_count = 0

                for _, row in qdf.iterrows():

                    response = str(row["response"]).strip().lower()
                    ground_truth = str(row["ground_truth"]).strip().lower()

                    if ground_truth in response:
                        correct_count += 1

                robustness_scores.append(
                    (correct_count / len(qdf)) * 100
                )

            robustness_score = round(
                sum(robustness_scores) / len(robustness_scores),
                2
            )

            # ---------------- Consistency ----------------
            consistency_scores = []

            for qid in task_df["id"].unique():

                qdf = task_df[task_df["id"] == qid]

                responses = (
                    qdf["response"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                )

                most_common = responses.value_counts().max()

                consistency_scores.append(
                    (most_common / len(qdf)) * 100
                )

            consistency_score = round(
                sum(consistency_scores) / len(consistency_scores),
                2
            )

            # ---------------- Overall Score ----------------

            speed_score = max(0, 100 - (avg_response_time * 10))

            length_score = max(
                0,
                100 - abs(avg_response_length - 30)
            )

            overall_score = round(
                0.40 * accuracy +
                0.30 * robustness_score +
                0.20 * consistency_score +
                0.05 * speed_score +
                0.05 * length_score,
                2
            )

            metrics.append({
                "model": model,
                "task": task,
                "correct": correct,
                "total": total,
                "accuracy": accuracy,
                "avg_response_length": avg_response_length,
                "avg_response_time": avg_response_time,
                "robustness_score": robustness_score,
                "consistency_score": consistency_score,
                "overall_score": overall_score
            })

    metrics_df = pd.DataFrame(metrics)

    metrics_df.to_csv(
        RESULTS_DIR / "metrics.csv",
        index=False
    )

    print(metrics_df)

    return metrics_df


if __name__ == "__main__":
    calculate_metrics()