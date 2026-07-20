from datasets import load_dataset
import pandas as pd
from pathlib import Path


def load_triviaqa_samples(num_samples=20):
    """
    Load a small subset of TriviaQA.
    """

    print("Loading TriviaQA dataset...")

    dataset = load_dataset("trivia_qa", "rc", split="train")

    rows = []

    for sample in dataset.select(range(num_samples)):

        prompt = sample["question"]
        answer = sample["answer"]["value"]

        rows.append({
            "task": "Factual QA",
            "prompt": prompt,
            "ground_truth": answer
        })

    return pd.DataFrame(rows)


def save_dataset(df):
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "seed_prompts.csv"

    df.to_csv(output_file, index=False)

    print(f"\nDataset saved to:\n{output_file}")


def main():

    df = load_triviaqa_samples(num_samples=20)

    print(df.head())

    save_dataset(df)


if __name__ == "__main__":
    main()