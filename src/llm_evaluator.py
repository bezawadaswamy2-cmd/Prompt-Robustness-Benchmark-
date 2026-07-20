import pandas as pd
from pathlib import Path
import ollama


def query_llm(prompt):
    """
    Send a prompt to the local Llama model.
    """

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


def main():

    input_file = Path("/Users/sribalaayyappaswamybezawada/Visual Studio Code/Prompt-Robustness-Benchmark/data/processed/perturbed_prompts.csv")
    output_file = Path("/Users/sribalaayyappaswamybezawada/Visual Studio Code/Prompt-Robustness-Benchmark/data/processed/responses.csv")

    df = pd.read_csv(input_file)

    responses = []

    print(f"Total prompts: {len(df)}")
    print("Generating responses...\n")

    for index, row in df.iterrows():

        print(f"[{index + 1}/{len(df)}]")

        answer = query_llm(row["prompt"])

        responses.append({
            "id": row["id"],
            "task": row["task"],
            "variation_id": row["variation_id"],
            "prompt": row["prompt"],
            "ground_truth": row["ground_truth"],
            "response": answer
        })

    result = pd.DataFrame(responses)

    result.to_csv(output_file, index=False)

    print("\nDone!")
    print(f"Saved responses to {output_file}")


if __name__ == "__main__":
    main()