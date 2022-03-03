import logging
import streamlit as st
import numpy as np
import pandas as pd
import sklearn
import plotly.express as px
from plotly_utils import univariate
from ml_pipelines import preps, classifiers
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import shap
from typing import NamedTuple


class Explanation(NamedTuple):
    base_values: np.ndarray
    data: np.ndarray
    feature_names: np.ndarray
    values: np.ndarray


@st.cache
def get_adult_dataset():
    df_adult = pd.read_csv("adult_csv.csv", delimiter=",")
    return df_adult


@st.cache
def get_features(dft):
    features = dft.drop('class', axis=1)
    return features


@st.cache
def get_target(dft):
    target_col = dft['class']
    target = target_col.map({'<=50K': 0, '>50K': 1})
    assert target.isnull().sum() == 0
    return target


@st.cache(allow_output_mutation=True)
def get_eda(df):
    logging.info("calculating EDA")
    tg = get_target(df)
    dtypes = pd.DataFrame(dict(
        dtype=df.dtypes.values,
        fillrate=df.notnull().mean(),
    ))
    hist = {c: px.histogram(df, x=c.Index) for c in dtypes.itertuples()}
    univ = {c: univariate(get_feature_stats(df[c.Index], tg, bins=7)) for c in dtypes.itertuples()}
    result = dict(
        shape=df.shape,
        summary=dtypes,
        hist=hist,
        univ=univ,
    )
    return result


def get_probability_vector(prob):
    if prob.shape[1] == 2:
        prob1 = prob[:, 1]
    else:
        prob1 = prob
    return prob1


def evaluate_model(prob, ye):
    auc = sklearn.metrics.roc_auc_score(y_true=ye, y_score=prob)
    logging.info(f"AUC: {auc:.3f}")
    return dict(
        auc=auc,
    )


def get_feature_stats(f: pd.Series, y: pd.Series, bins) -> dict:
    if pd.api.types.is_numeric_dtype(f.dtype):
        if isinstance(bins, int):
            nbins = bins
            qbins = np.linspace(1/nbins, 1-1/nbins, nbins-1)
            xbins = [-np.inf, *np.unique(np.quantile(f.dropna(), qbins)), +np.inf]
        else:
            xbins = bins
        x = pd.cut(f, bins=xbins, right=False)
    else:
        x = f.astype(str)
    df = pd.DataFrame({
        "num_negative": x[y == 0].value_counts(dropna=False),
        "num_positive": x[y == 1].value_counts(dropna=False),
    }).sort_index().fillna(0)
    df['num_observations'] = df.loc[:, "num_negative"] + df.loc[:, "num_positive"]
    df['frequency'] = df.num_observations / len(f)
    df['positive_rate'] = (df.num_positive / df.num_observations).fillna(0.0)
    df['bin'] = [str(interval) for interval in df.index]
    if pd.api.types.is_string_dtype(f.dtype):
        df = df.sort_values('positive_rate')
    return df.to_dict(orient='records')


def get_examples(prob, ye):
    ind = np.argsort(prob)
    yi = ye[ind]
    example1 = max(ind[yi == 1])  # max probability with positive label
    example2 = max(ind[yi == 0])  # max probability with negative label
    example3 = min(ind[yi == 1])  # min probability with positive label
    example4 = min(ind[yi == 0])  # min probability with negative label
    ind_examples = [
        example1,
        example2,
        example3,
        example4,
    ]
    df_examples = pd.DataFrame({
        'predicted_probability': prob[ind_examples],
        'target': ye[ind_examples],
    }, index=ind_examples)
    return df_examples


logging.basicConfig(level='INFO')
st.set_page_config(layout='wide')
st.title("Demo Shapley Values on RandomForest and Adult Dataset")
st.write("# Exploratory Data Analysis")
is_eda = st.checkbox("Calculate EDA?", value=False)
df = get_adult_dataset()
if is_eda:
    eda = get_eda(df)
    dt = eda['summary']
    with st.container():
        with st.expander("Columns Summary"):
            st.write(eda['shape'])
            st.write(dt)
        with st.expander("Histograms"):
            for col in dt.itertuples():
                st.write(f"## {col.Index}")
                fig = eda['hist'][col]
                st.plotly_chart(fig)
        with st.expander("Univariate"):
            for col in dt.itertuples():
                st.write(f"## {col.Index}")
                fig = eda['univ'][col]
                st.plotly_chart(fig)
st.write("# Model Training")
with st.form("train_form"):
    with st.expander("Train / Test Split"):
        train_ratio = st.slider(
            label="Train Proportion",
            min_value=0.1,
            max_value=0.95,
            value=0.9,
            step=0.05,
        )
    prep_name = st.selectbox("Preprocessing", options=preps.keys())
    cls_name = st.selectbox("Classifier", options=classifiers.keys())
    train_submit = st.form_submit_button("Start Model Training")
    if train_submit:
        x = get_features(df)
        y = get_target(df)
        logging.info(f"model training triggered with x: {x.shape} and y: {y.shape}")
        xt, xs, yt, ys = sklearn.model_selection.train_test_split(x, y, train_size=train_ratio, random_state=42)
        prep = Pipeline(preps[prep_name])
        classifier = classifiers[cls_name]
        with st.expander("Evaluation"):
            st.write(f"Preprocessing steps: {list(prep.named_steps.keys())}")
            st.write(f"Classifier: {classifier}")
            prep.fit(xt, yt)
            feature_names = prep.named_steps['encode'].get_feature_names_out()
            xp = prep.transform(xt)
            logging.info(f"Preprocessed x: {xp.shape} with {np.mean(np.isnan(xp)):.1%} missing values")
            logging.info(f"Preprocessed y: {yt.shape} with {np.mean(np.isnan(yt)):.1%} missing values")
            model = classifier.fit(xp, yt)
            xsp = prep.transform(xs)
            prob = model.predict_proba(xsp)
            prob1 = get_probability_vector(prob)
            e = evaluate_model(prob1, ys)
            if 'previous_auc' not in st.session_state:
                st.metric(label="AUC", value=f"{e['auc']:.3f}")
            else:
                delta_auc = e['auc'] - st.session_state['previous_auc']
                st.metric(label="AUC", value=f"{e['auc']:.3f}", delta=delta_auc)
            st.session_state['previous_auc'] = e['auc']
        with st.expander("Global Model Explanation (Shap)"):
            explainer = shap.explainers.Tree(model, data=xsp)
            logging.info(f"calculating Shap values for {len(xsp)} observations")
            with st.spinner(text='Shap values in progress'):
                sv = explainer.shap_values(xsp, ys)
                st.success('Done')
            logging.info(f"shap values: {sv[1].shape}")
            fig, ax = plt.subplots()
            if cls_name in ["DecisionTree", "RandomForest"]:
                shap.summary_plot(sv[1], xsp, feature_names=feature_names, plot_type="layered_violin", color='coolwarm')
                expected_value = explainer.expected_value[1]
                svpos = sv[1]
            else:
                shap.summary_plot(sv, xsp, feature_names=feature_names, plot_type="layered_violin", color='coolwarm')
                expected_value = explainer.expected_value
                svpos = sv
            st.pyplot(fig)
        st.session_state['model'] = dict(
            raw_data=xs.values,
            data=xsp,
            prob=prob1,
            target=ys.values,
            expected_value=expected_value,
            shapley_values=svpos,
            feature_names=feature_names,
        )
st.write("# Model Serving")
if 'model' in st.session_state.keys():
    m = st.session_state['model']
    examples = get_examples(m['prob'], m['target'])
    st.write(examples)
    index_explain = st.number_input(
        "Observation to explain",
        value=examples.index[0],
        min_value=0,
        max_value=len(m['prob']) - 1,
    )
    st.write(f"Explaining predicted probability {m['prob'][index_explain]:.3f} for observation {index_explain}")
    fig_explain, ax_explain = plt.subplots()
    e = Explanation(
        base_values=m['expected_value'],
        values=m['shapley_values'][index_explain, :],
        data=m['raw_data'][index_explain, :],
        feature_names=m['feature_names'],
    )
    logging.info(f"explanation:\n{e}")
    shap.waterfall_plot(e, show=False)
    st.pyplot(fig_explain)
else:
    st.write("no model trained")
