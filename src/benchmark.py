import pandas as pd

from src.evaluator import evaluate_model
from src.config import AVAILABLE_MODELS, RESULTS_DIR


def run_benchmark(models=None, limit=None):

    if models is None:
        models = AVAILABLE_MODELS

    all_results = []

    for model in models:

        print("=" * 60)
        print(f"Running Benchmark for {model}")
        print("=" * 60)

        df = evaluate_model(
            model_name=model,
            limit=limit
        )

        all_results.append(df)

    final_df = pd.concat(all_results, ignore_index=True)

    RESULTS_DIR.mkdir(exist_ok=True)

    output_file = RESULTS_DIR / "benchmark_results.csv"

    final_df.to_csv(output_file, index=False)

    print(f"\nBenchmark results saved to:\n{output_file}")

    return final_df