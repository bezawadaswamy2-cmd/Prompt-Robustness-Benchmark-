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


def generate_perturbations(
    input_file="/Users/sribalaayyappaswamybezawada/Visual Studio Code/Prompt-Robustness-Benchmark/data/processed/seed_prompts.csv",
    output_file="/Users/sribalaayyappaswamybezawada/Visual Studio Code/Prompt-Robustness-Benchmark/data/processed/perturbed_prompts.csv"
):
    """
    Generate prompt perturbations and save them.
    """

    input_file = Path(input_file)
    output_file = Path(output_file)

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

    return perturbed_df