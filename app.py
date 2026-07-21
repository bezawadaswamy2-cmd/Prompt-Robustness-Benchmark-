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

    st.header("Results")

    st.info("Metrics and charts will appear here.")

with tab4:

    st.header("About")

    st.write("""
    MSc Project

    Prompt Robustness Benchmarking for Large Language Models

    University of Europe for Applied Sciences
    """)