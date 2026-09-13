import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)
from sklearn.inspection import partial_dependence

st.set_page_config(
    page_title="Bank Customer Churn Risk Intelligence",
    page_icon="\U0001F3E6",
    layout="wide",
)

DATA_PATH = "data/European_Bank.csv"

RAW_FEATURES = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]

# ---------------------------------------------------------------------------
# Data loading + feature engineering
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    drop_cols = [c for c in ["CustomerId", "Surname", "Year"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same derived features used in training to any dataframe of
    raw customer rows (single row or many)."""
    df = df.copy()
    df["BalanceSalaryRatio"] = df["Balance"] / df["EstimatedSalary"].replace(0, 1)
    df["ProductDensity"] = df["NumOfProducts"] / (df["Tenure"] + 1)
    df["EngagementProductInteraction"] = df["IsActiveMember"] * df["NumOfProducts"]
    df["AgeTenureInteraction"] = df["Age"] * df["Tenure"]
    df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)
    return df


@st.cache_resource
def train_models():
    df = load_data()
    df_fe = engineer_features(df)

    for col in ["Geography_Germany", "Geography_Spain", "Gender_Male"]:
        if col not in df_fe.columns:
            df_fe[col] = 0

    feature_cols = [c for c in df_fe.columns if c != "Exited"]
    X = df_fe[feature_cols]
    y = df_fe["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_defs = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    trained = {}
    metrics = []
    for name, model in model_defs.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]

        trained[name] = model
        metrics.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds),
            "Recall": recall_score(y_test, preds),
            "F1-Score": f1_score(y_test, preds),
            "ROC-AUC": roc_auc_score(y_test, proba),
        })

    metrics_df = pd.DataFrame(metrics).set_index("Model").round(4)

    return {
        "models": trained,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_test_scaled": X_test_scaled,
        "metrics_df": metrics_df,
        "raw_df": df,
    }


def predict_proba_for(model_name, X_row, bundle):
    """Predict churn probability for a single-row (or multi-row) engineered
    feature dataframe, aligned to the training feature columns."""
    X_row = X_row.reindex(columns=bundle["feature_cols"], fill_value=0)
    model = bundle["models"][model_name]
    if model_name == "Logistic Regression":
        X_in = bundle["scaler"].transform(X_row)
    else:
        X_in = X_row
    return model.predict_proba(X_in)[:, 1]


@st.cache_resource
def get_shap_values(model_name, _bundle, sample_size=300):
    model = _bundle["models"][model_name]
    X_test = _bundle["X_test"]
    sample = X_test.sample(min(sample_size, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(sample)
    sv = np.array(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]  # positive class
    return sv, sample


TREE_MODELS = ["Decision Tree", "Random Forest", "Gradient Boosting"]

bundle = train_models()
metrics_df = bundle["metrics_df"]

st.title("Bank Customer Churn Risk Intelligence")
st.caption(
    "Predictive modeling and explainable risk scoring for retail bank customer churn."
)

tabs = st.tabs([
    "Model Performance",
    "Churn Risk Calculator",
    "Probability Distribution",
    "Feature Importance & SHAP",
    "What-If Simulator",
])

# ---------------------------------------------------------------------------
# Tab 1: Model performance overview
# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Model comparison")
    st.dataframe(metrics_df.style.highlight_max(axis=0, color="#d4f4dd"), width='stretch')

    best_model = metrics_df["ROC-AUC"].idxmax()
    st.info(f"Highest ROC-AUC: **{best_model}** ({metrics_df.loc[best_model, 'ROC-AUC']:.3f})")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**ROC curves**")
        fig = go.Figure()
        for name, model in bundle["models"].items():
            if name == "Logistic Regression":
                proba = model.predict_proba(bundle["X_test_scaled"])[:, 1]
            else:
                proba = model.predict_proba(bundle["X_test"])[:, 1]
            fpr, tpr, _ = roc_curve(bundle["y_test"], proba)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=name))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                                  line=dict(dash="dash", color="gray")))
        fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                           height=400, margin=dict(t=20))
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown("**Confusion matrix**")
        sel_model = st.selectbox("Model", list(bundle["models"].keys()), index=2, key="cm_model")
        model = bundle["models"][sel_model]
        if sel_model == "Logistic Regression":
            preds = model.predict(bundle["X_test_scaled"])
        else:
            preds = model.predict(bundle["X_test"])
        cm = confusion_matrix(bundle["y_test"], preds)
        fig_cm = px.imshow(cm, text_auto=True, x=["Predicted: Stay", "Predicted: Churn"],
                            y=["Actual: Stay", "Actual: Churn"], color_continuous_scale="Blues")
        fig_cm.update_layout(height=400, margin=dict(t=20))
        st.plotly_chart(fig_cm, width='stretch')

# ---------------------------------------------------------------------------
# Tab 2: Churn risk calculator
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Customer churn risk calculator")
    model_choice = st.selectbox("Scoring model", list(bundle["models"].keys()), index=2,
                                 key="calc_model")
    threshold = st.slider("Churn flag threshold", 0.0, 1.0, 0.5, 0.01, key="calc_threshold")

    c1, c2, c3 = st.columns(3)
    with c1:
        credit_score = st.slider("Credit score", 300, 850, 650)
        geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
        gender = st.selectbox("Gender", ["Female", "Male"])
    with c2:
        age = st.slider("Age", 18, 92, 40)
        tenure = st.slider("Tenure (years)", 0, 10, 5)
        num_products = st.selectbox("Number of products", [1, 2, 3, 4], index=0)
    with c3:
        balance = st.number_input("Account balance", 0.0, 300000.0, 75000.0, step=1000.0)
        salary = st.number_input("Estimated salary", 0.0, 250000.0, 100000.0, step=1000.0)
        has_card = st.selectbox("Has credit card?", ["Yes", "No"], index=0)
        is_active = st.selectbox("Active member?", ["Yes", "No"], index=0)

    input_row = pd.DataFrame([{
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": 1 if has_card == "Yes" else 0,
        "IsActiveMember": 1 if is_active == "Yes" else 0,
        "EstimatedSalary": salary,
    }])

    X_row = engineer_features(input_row)
    proba = predict_proba_for(model_choice, X_row, bundle)[0]
    flag = "CHURN RISK" if proba >= threshold else "LIKELY TO STAY"

    st.divider()
    m1, m2 = st.columns([1, 2])
    with m1:
        st.metric("Churn probability", f"{proba:.1%}")
        if proba >= 0.66:
            st.error(f"**{flag}** — High risk")
        elif proba >= 0.33:
            st.warning(f"**{flag}** — Medium risk")
        else:
            st.success(f"**{flag}** — Low risk")
    with m2:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {"range": [0, 33], "color": "#d4f4dd"},
                    {"range": [33, 66], "color": "#ffe8b3"},
                    {"range": [66, 100], "color": "#ffcccc"},
                ],
                "threshold": {"line": {"color": "black", "width": 3},
                              "value": threshold * 100},
            },
        ))
        fig_gauge.update_layout(height=250, margin=dict(t=10, b=10))
        st.plotly_chart(fig_gauge, width='stretch')

# ---------------------------------------------------------------------------
# Tab 3: Probability distribution
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Churn probability distribution across customers")
    dist_model = st.selectbox("Model", list(bundle["models"].keys()), index=2, key="dist_model")
    model = bundle["models"][dist_model]
    if dist_model == "Logistic Regression":
        proba_all = model.predict_proba(bundle["X_test_scaled"])[:, 1]
    else:
        proba_all = model.predict_proba(bundle["X_test"])[:, 1]

    dist_df = pd.DataFrame({
        "Predicted probability": proba_all,
        "Actual outcome": bundle["y_test"].map({0: "Retained", 1: "Churned"}).values,
    })

    fig_dist = px.histogram(
        dist_df, x="Predicted probability", color="Actual outcome",
        nbins=40, barmode="overlay", opacity=0.65,
        color_discrete_map={"Retained": "#2ca02c", "Churned": "#d62728"},
    )
    fig_dist.update_layout(height=450)
    st.plotly_chart(fig_dist, width='stretch')

    st.caption(
        "A well-separated model pushes churned customers (red) toward higher "
        "predicted probabilities and retained customers (green) toward lower ones."
    )

# ---------------------------------------------------------------------------
# Tab 4: Feature importance + SHAP
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("What drives churn risk")
    fi_model = st.selectbox("Model", ["Random Forest", "Gradient Boosting", "Decision Tree",
                                       "Logistic Regression"], key="fi_model")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Feature importance**")
        model = bundle["models"][fi_model]
        if fi_model == "Logistic Regression":
            importance = pd.Series(np.abs(model.coef_[0]), index=bundle["feature_cols"])
        else:
            importance = pd.Series(model.feature_importances_, index=bundle["feature_cols"])
        importance = importance.sort_values(ascending=True)
        fig_imp = px.bar(importance, orientation="h", labels={"value": "Importance", "index": ""})
        fig_imp.update_layout(height=450, showlegend=False)
        st.plotly_chart(fig_imp, width='stretch')

    with col2:
        st.markdown("**SHAP summary (mean impact on churn probability)**")
        if fi_model in TREE_MODELS:
            sv, sample = get_shap_values(fi_model, bundle)
            mean_abs_shap = pd.Series(np.abs(sv).mean(axis=0), index=sample.columns)
            mean_abs_shap = mean_abs_shap.sort_values(ascending=True)
            fig_shap = px.bar(mean_abs_shap, orientation="h",
                               labels={"value": "Mean |SHAP value|", "index": ""})
            fig_shap.update_layout(height=450, showlegend=False)
            st.plotly_chart(fig_shap, width='stretch')
        else:
            st.info("SHAP TreeExplainer is shown for tree-based models. "
                    "Logistic Regression uses coefficient magnitude (left) instead.")

    st.divider()
    st.markdown("**Partial dependence**")
    pdp_feature = st.selectbox(
        "Feature", ["Age", "NumOfProducts", "Balance", "IsActiveMember", "Tenure", "CreditScore"],
        key="pdp_feature",
    )
    pdp_model_name = "Random Forest"
    pdp_model = bundle["models"][pdp_model_name]
    pdp_X = bundle["X_train"].astype({pdp_feature: "float64"})
    pdp_result = partial_dependence(
        pdp_model, pdp_X, features=[pdp_feature], kind="average"
    )
    grid_key = "grid_values" if "grid_values" in pdp_result else "values"
    fig_pdp = px.line(
        x=pdp_result[grid_key][0], y=pdp_result["average"][0],
        labels={"x": pdp_feature, "y": "Predicted churn probability"},
    )
    fig_pdp.update_layout(height=350)
    st.plotly_chart(fig_pdp, width='stretch')
    st.caption(f"Partial dependence computed on the {pdp_model_name} model.")

# ---------------------------------------------------------------------------
# Tab 5: What-if scenario simulator
# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("What-if scenario simulator")
    st.caption("Start from a real customer, then adjust engagement and product features to see how churn risk shifts.")

    raw_df = bundle["raw_df"]
    sim_model_name = st.selectbox("Model", list(bundle["models"].keys()), index=2, key="sim_model")

    idx = st.number_input("Customer row (0-indexed)", 0, len(raw_df) - 1, 0, key="sim_idx")
    base_customer = raw_df.iloc[[idx]][RAW_FEATURES].copy()
    st.write("Base customer:")
    st.dataframe(base_customer, width='stretch')

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        sim_products = st.slider("Number of products", 1, 4, int(base_customer["NumOfProducts"].iloc[0]))
    with sc2:
        sim_active = st.selectbox("Active member?", ["Yes", "No"],
                                   index=0 if base_customer["IsActiveMember"].iloc[0] == 1 else 1,
                                   key="sim_active")
    with sc3:
        sim_balance = st.slider("Balance", 0.0, 250000.0, float(base_customer["Balance"].iloc[0]))

    scenario = base_customer.copy()
    scenario["NumOfProducts"] = sim_products
    scenario["IsActiveMember"] = 1 if sim_active == "Yes" else 0
    scenario["Balance"] = sim_balance

    base_X = engineer_features(base_customer)
    scenario_X = engineer_features(scenario)
    base_proba = predict_proba_for(sim_model_name, base_X, bundle)[0]
    scenario_proba = predict_proba_for(sim_model_name, scenario_X, bundle)[0]

    r1, r2, r3 = st.columns(3)
    r1.metric("Original probability", f"{base_proba:.1%}")
    r2.metric("Scenario probability", f"{scenario_proba:.1%}", delta=f"{(scenario_proba - base_proba):+.1%}")
    r3.metric("Risk change", "Lower" if scenario_proba < base_proba else
              ("Higher" if scenario_proba > base_proba else "No change"))

    st.divider()
    st.markdown("**Sensitivity: churn probability vs. number of products**")
    sweep_rows = []
    for p in [1, 2, 3, 4]:
        row = base_customer.copy()
        row["NumOfProducts"] = p
        row["IsActiveMember"] = 1 if sim_active == "Yes" else 0
        row["Balance"] = sim_balance
        sweep_rows.append(row)
    sweep_df = pd.concat(sweep_rows, ignore_index=True)
    sweep_X = engineer_features(sweep_df)
    sweep_proba = predict_proba_for(sim_model_name, sweep_X, bundle)
    fig_sweep = px.line(x=[1, 2, 3, 4], y=sweep_proba,
                         labels={"x": "Number of products", "y": "Churn probability"},
                         markers=True)
    fig_sweep.update_layout(height=350)
    st.plotly_chart(fig_sweep, width='stretch')
