import streamlit as st

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
        st.success("Benchmark execution will be connected in the next step.")

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