# Bank Customer Churn Risk Intelligence — Streamlit App

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this whole folder (app.py, requirements.txt, data/European_Bank.csv) to a GitHub repo.
2. On share.streamlit.io, point a new app at app.py in that repo.
3. Keep the CSV at data/European_Bank.csv relative to app.py — the app reads it from there.

## What's inside
- **Model Performance** — Logistic Regression, Decision Tree, Random Forest, and Gradient
  Boosting compared on Accuracy, Precision, Recall, F1, and ROC-AUC, plus ROC curves and a
  confusion matrix.
- **Churn Risk Calculator** — enter a customer's profile and get a churn probability, a
  risk gauge, and a threshold-based churn flag.
- **Probability Distribution** — histogram of predicted probabilities split by actual
  outcome, to see how well the model separates churners from retained customers.
- **Feature Importance & SHAP** — built-in feature importance / coefficients plus a SHAP
  summary (mean |SHAP value|) for tree-based models, and a partial dependence plot for a
  chosen feature.
- **What-If Simulator** — start from a real customer row, adjust NumOfProducts,
  IsActiveMember, and Balance, and see the churn probability update, including a
  sensitivity curve across product counts.

## Notes
- Feature engineering (BalanceSalaryRatio, ProductDensity, EngagementProductInteraction,
  AgeTenureInteraction) is applied identically at training time and at inference time via
  a shared `engineer_features()` function, so the calculator and simulator stay consistent
  with the trained models.
- Models train once per app session and are cached with `st.cache_resource`.
