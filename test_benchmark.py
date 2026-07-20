from src.benchmark import run_benchmark

df = run_benchmark(limit=3)

print(df.head())

print("\nTotal Responses:", len(df))