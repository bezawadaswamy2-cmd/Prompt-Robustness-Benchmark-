from src.perturbation import generate_perturbations

df = generate_perturbations()

print(df.head())

print(f"\nGenerated {len(df)} prompts.")