from pathlib import Path

PROJECT_ROOT = Path("/Users/sribalaayyappaswamybezawada/Visual Studio Code/Prompt-Robustness-Benchmark")

DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

SEED_PROMPTS = DATA_DIR / "seed_prompts.csv"
PERTURBED_PROMPTS = DATA_DIR / "perturbed_prompts.csv"
RESPONSES = DATA_DIR / "responses.csv"
METRICS = DATA_DIR / "metrics.csv"

AVAILABLE_MODELS = [
    "llama3.2:3b",
    "mistral:7b",
    "gemma3:4b",
]

DEFAULT_MODEL = "llama3.2:3b"