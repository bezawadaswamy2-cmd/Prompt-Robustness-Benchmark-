import pandas as pd
import ollama

from src.config import PERTURBED_PROMPTS, RESULTS_DIR


def query_model(model_name, prompt):
    response = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


def evaluate_model(model_name, limit=None):

    df = pd.read_csv(PERTURBED_PROMPTS)

    if limit:
        df = df.head(limit)

    responses = []

    print(f"\nRunning {model_name}")
    print(f"Total prompts: {len(df)}")

    for index, row in df.iterrows():

        print(f"[{index+1}/{len(df)}]")

        answer = query_model(model_name, row["prompt"])

        responses.append({
            "model": model_name,
            "id": row["id"],
            "task": row["task"],
            "variation_id": row["variation_id"],
            "prompt": row["prompt"],
            "ground_truth": row["ground_truth"],
            "response": answer
        })

    result = pd.DataFrame(responses)

    RESULTS_DIR.mkdir(exist_ok=True)

    filename = model_name.replace(":", "_") + "_responses.csv"

    result.to_csv(RESULTS_DIR / filename, index=False)

    return result