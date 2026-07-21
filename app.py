import streamlit as st
from src.benchmark import run_benchmark
from src.metrics import calculate_metrics
from src.visualization import create_accuracy_chart

st.set_page_config(
    page_title="Prompt Robustness Benchmark",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Prompt Robustness Benchmarking for Large Language Models")
st.markdown(
    """
    Compare the robustness of multiple open-source LLMs against prompt perturbations.
    """
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Home",
    "⚙️ Benchmark",
    "📊 Results",
    "ℹ️ About"
])

with tab1:
    st.header("Welcome")

    st.write("""
    This application benchmarks multiple Large Language Models using
    prompt perturbations.

    Supported Models:
    - Llama 3.2
    - Mistral
    - Gemma

    Workflow:
    1. Generate perturbations
    2. Run benchmark
    3. Compare results
    """)

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

                run_benchmark(models=selected_models)

                calculate_metrics()

                create_accuracy_chart()

            st.success("Benchmark completed successfully!")
with tab3:

    st.header("📊 Results")

    import pandas as pd
    from src.config import RESULTS_DIR

    benchmark_file = RESULTS_DIR / "benchmark_results.csv"
    metrics_file = RESULTS_DIR / "metrics.csv"
    chart_file = RESULTS_DIR / "accuracy_chart.png"

    # Benchmark Summary
    if benchmark_file.exists():

        benchmark_df = pd.read_csv(benchmark_file)

        st.subheader("Benchmark Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric("Models Tested", benchmark_df["model"].nunique())
        col2.metric("Responses", len(benchmark_df))
        col3.metric("Tasks", benchmark_df["task"].nunique())

        st.divider()

    # Metrics Table
    if metrics_file.exists():

        metrics_df = pd.read_csv(metrics_file)

        st.subheader("Metrics")
        st.dataframe(metrics_df, use_container_width=True)

        st.divider()

    # Accuracy Chart
    if chart_file.exists():

        st.subheader("Accuracy Comparison")

        st.image(str(chart_file), use_container_width=True)

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

with tab4:

    st.header("About")

    st.write("""
    MSc Project

    Prompt Robustness Benchmarking for Large Language Models

    University of Europe for Applied Sciences
    """)