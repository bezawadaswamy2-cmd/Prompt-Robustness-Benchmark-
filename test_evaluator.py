from src.evaluator import evaluate_model

df = evaluate_model(
    model_name="llama3.2:3b",
    limit=5
)

print(df.head())