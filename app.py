import streamlit as st
import pandas as pd

from src.benchmark import run_benchmark
from src.metrics import calculate_metrics
from src.visualization import create_all_charts
from src.config import RESULTS_DIR


st.set_page_config(
    page_title="Prompt Robustness Benchmark",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Prompt Robustness Benchmarking for Large Language Models")

st.markdown("""
Compare the robustness of multiple open-source Large Language Models
against prompt perturbations.
""")

tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Home",
    "⚙️ Benchmark",
    "📊 Results",
    "ℹ️ About"
])


# HOME

with tab1:

    st.header("Welcome")

    st.write("""
This application benchmarks multiple Large Language Models using
prompt perturbations.

### Supported Models
- Llama 3.2
- Mistral
- Gemma

### Workflow
1. Generate prompt perturbations
2. Run benchmark
3. Calculate metrics
4. Compare model robustness
5. Visualize results
""")


# BENCHMARK

with tab2:

    st.header("Benchmark Settings")

    st.subheader("Select Models")

    llama = st.checkbox("Llama 3.2", value=True)
    mistral = st.checkbox("Mistral", value=True)
    gemma = st.checkbox("Gemma", value=True)

    st.divider()

    if st.button("🚀 Run Benchmark"):

        selected_models = []

        if llama:
            selected_models.append("llama3.2:3b")

        if mistral:
            selected_models.append("mistral:7b")

        if gemma:
            selected_models.append("gemma3:4b")

        if not selected_models:

            st.error("Please select at least one model.")

        else:

            with st.spinner("Running benchmark..."):

                run_benchmark(models=selected_models, limit=10)

                calculate_metrics()

                create_all_charts()

            st.success("✅ Benchmark completed successfully!")


# RESULTS

with tab3:

    st.header("📊 Benchmark Results")

    benchmark_file = RESULTS_DIR / "benchmark_results.csv"
    metrics_file = RESULTS_DIR / "metrics.csv"
    charts_dir = RESULTS_DIR / "charts"

    # Benchmark Summary

    if benchmark_file.exists():

        benchmark_df = pd.read_csv(benchmark_file)

        st.subheader("Benchmark Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Models Tested",
            benchmark_df["model"].nunique()
        )

        col2.metric(
            "Responses",
            len(benchmark_df)
        )

        col3.metric(
            "Tasks",
            benchmark_df["task"].nunique()
        )

        st.divider()

    # Metrics

    if metrics_file.exists():

        metrics_df = pd.read_csv(metrics_file)

        st.subheader("Metrics")

        st.dataframe(
            metrics_df,
            use_container_width=True
        )

        # Best Model

        if "overall_score" in metrics_df.columns:

            best_model = metrics_df.loc[
                metrics_df["overall_score"].idxmax()
            ]

            st.success(
                f"""
🏆 **Best Model:** {best_model['model']}

⭐ Overall Score: {best_model['overall_score']:.2f}
"""
            )

        st.divider()

    # Charts

    chart_files = [
        ("Accuracy", "accuracy.png"),
        ("Robustness", "robustness.png"),
        ("Consistency", "consistency.png"),
        ("Average Response Time", "response_time.png"),
        ("Average Response Length", "response_length.png"),
        ("Overall Score", "overall_score.png"),
    ]

    for title, filename in chart_files:

        chart_path = charts_dir / filename

        if chart_path.exists():

            st.subheader(title)

            st.image(
                str(chart_path),
                use_container_width=True
            )

            st.divider()

    # Downloads

    st.subheader("Downloads")

    if benchmark_file.exists():

        with open(benchmark_file, "rb") as f:

            st.download_button(
                "📥 Download Benchmark Results",
                f,
                file_name="benchmark_results.csv",
                mime="text/csv"
            )

    if metrics_file.exists():

        with open(metrics_file, "rb") as f:

            st.download_button(
                "📥 Download Metrics",
                f,
                file_name="metrics.csv",
                mime="text/csv"
            )


# ABOUT

with tab4:

    st.header("About")

    st.write("""
### Prompt Robustness Benchmarking for Large Language Models

**MSc Data Science Project**

University of Europe for Applied Sciences

This project evaluates the robustness of multiple open-source
Large Language Models under prompt perturbations using:

- Accuracy
- Robustness Score
- Consistency Score
- Response Time
- Response Length
- Overall Model Score

Models evaluated:

- Llama 3.2
- Mistral
- Gemma
""")