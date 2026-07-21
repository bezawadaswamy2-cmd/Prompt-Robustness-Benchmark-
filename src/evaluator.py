import pandas as pd
import ollama
import time

from src.config import PERTURBED_PROMPTS, RESULTS_DIR


def query_model(model_name, prompt):

    start_time = time.time()

    response = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    end_time = time.time()

    response_text = response["message"]["content"]

    response_time = round(end_time - start_time, 3)

    return response_text, response_time


def evaluate_model(model_name, limit=None):

    df = pd.read_csv(PERTURBED_PROMPTS)

    if limit:
        df = df.head(limit)

    responses = []

    print(f"\nRunning {model_name}")
    print(f"Total prompts: {len(df)}")

    for index, row in df.iterrows():

        print(f"[{index+1}/{len(df)}]")

        answer, response_time = query_model(model_name, row["prompt"])

        responses.append({
            "model": model_name,
            "id": row["id"],
            "task": row["task"],
            "variation_id": row["variation_id"],
            "prompt": row["prompt"],
            "ground_truth": row["ground_truth"],
            "response": answer,
            "response_time": response_time
        })
    print("\nResponse dictionary:")
    print(responses[0])

    result = pd.DataFrame(responses)
    print("\nDataFrame columns:")
    print(result.columns.tolist())

    print(result.head())

    RESULTS_DIR.mkdir(exist_ok=True)

    filename = model_name.replace(":", "_") + "_responses.csv"

    result.to_csv(RESULTS_DIR / filename, index=False)

    return result