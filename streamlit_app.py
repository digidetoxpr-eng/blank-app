
import streamlit as st
import pandas as pd

from model_runner import train_and_evaluate


st.set_page_config(page_title="Logistic Regression explorer", layout="wide")

st.title("Logistic Regression explorer")

uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

MODEL_OPTIONS = [
    "logistic Regression",
    "Decision Tree Classifier",
    "K-Nearest Neighbor Classifier",
    "Naive Bayes Classifier - Gaussian or Multinomial",
    "Ensemble Model - Random Forest",
    "Ensemble Model - XGBoost",
]


def _on_model_change():
    # Streamlit reruns on widget interaction; use state to trigger evaluation
    st.session_state["trigger_eval"] = True


col1, col2 = st.columns([1, 2], vertical_alignment="top")

with col1:
    model_choice = st.selectbox(
        "Choose a model",
        MODEL_OPTIONS,
        key="model_choice",
        on_change=_on_model_change,
    )

    nb_variant = None
    if model_choice == "Naive Bayes Classifier - Gaussian or Multinomial":
        nb_variant = st.radio(
            "Naive Bayes type",
            ["Gaussian", "Multinomial"],
            horizontal=True,
            key="nb_variant",
            on_change=_on_model_change,
        )

with col2:
    metrics_placeholder = st.empty()

if uploaded is None:
    st.info("Upload a CSV to enable model evaluation.")
    st.stop()

df = pd.read_csv(uploaded)
st.subheader("Preview")
st.dataframe(df.head(25), use_container_width=True)

# Minimal, necessary extra control: choose target column
target_col = st.selectbox("Select target column", list(df.columns), key="target_col")

# Trigger on dropdown change OR allow manual run
run_clicked = st.button("Run / Re-run", type="primary")
should_run = st.session_state.get("trigger_eval", False) or run_clicked

if should_run:
    st.session_state["trigger_eval"] = False

    with st.spinner("Training and evaluating..."):
        metrics_dict= train_and_evaluate(
            model_name=model_choice,
            df=df,
            target_col=target_col,
            nb_variant=st.session_state.get("nb_variant"),
        )
    print(metrics_dict)
    st.subheader("Metrics")
    #metrics_placeholder.dataframe(metrics_df, use_container_width=True)
    # Convert to DataFrame (orient='index' makes keys the rows)
    metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index', columns=['Value'])

    # Display
    metrics_placeholder.dataframe(metrics_df, width='stretch')

