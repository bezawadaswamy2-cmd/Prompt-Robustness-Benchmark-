import pandas as pd
from pathlib import Path


def generate_variations(prompt):
    """
    Generate simple prompt variations.
    """

    return [
        prompt,
        f"Can you tell me {prompt.lower()}",
        f"Please answer: {prompt}",
        f"I would like to know: {prompt}",
        f"Kindly respond to the following question: {prompt}"
    ]


def main():

    input_file = Path("/Users/sribalaayyappaswamybezawada/Visual Studio Code/Prompt-Robustness-Benchmark/data/processed/seed_prompts.csv")
    output_file = Path("/Users/sribalaayyappaswamybezawada/Visual Studio Code/Prompt-Robustness-Benchmark/data/processed/perturbed_prompts.csv")

    df = pd.read_csv(input_file)

    all_rows = []

    for _, row in df.iterrows():

        variations = generate_variations(row["prompt"])

        for i, variation in enumerate(variations):

            all_rows.append({
                "id": row["id"],
                "task": row["task"],
                "variation_id": i + 1,
                "prompt": variation,
                "ground_truth": row["ground_truth"]
            })

    perturbed_df = pd.DataFrame(all_rows)

    perturbed_df.to_csv(output_file, index=False)

    print("=" * 50)
    print("Perturbation Generation Complete")
    print("=" * 50)
    print(f"Original Prompts : {len(df)}")
    print(f"Generated Prompts: {len(perturbed_df)}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()