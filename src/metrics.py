import pandas as pd
from pathlib import Path


def exact_match(response, ground_truth):
    """
    Case-insensitive exact match.
    """

    response = str(response).strip().lower()
    ground_truth = str(ground_truth).strip().lower()

    return int(ground_truth in response)


def main():

    input_file = Path("/Users/sribalaayyappaswamybezawada/Visual Studio Code/Prompt-Robustness-Benchmark/data/processed/responses.csv")
    output_file = Path("/Users/sribalaayyappaswamybezawada/Visual Studio Code/Prompt-Robustness-Benchmark/data/processed/metrics.csv")

    df = pd.read_csv(input_file)

    df["exact_match"] = df.apply(
        lambda row: exact_match(row["response"], row["ground_truth"]),
        axis=1
    )

    metrics = (
        df.groupby("task")
        .agg(
            total_prompts=("id", "count"),
            correct=("exact_match", "sum"),
            accuracy=("exact_match", "mean")
        )
        .reset_index()
    )

    metrics["accuracy"] = metrics["accuracy"] * 100

    metrics.to_csv(output_file, index=False)

    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    print(metrics)

    overall_accuracy = df["exact_match"].mean() * 100

    print("\nOverall Accuracy: {:.2f}%".format(overall_accuracy))

    print("\nMetrics saved to:")
    print(output_file)


if __name__ == "__main__":
    main()