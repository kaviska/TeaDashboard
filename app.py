"""
Colombo Tea Auction - Ensemble Learning Dashboard
Fair-price estimator using bagging, boosting and stacking ensembles.
Run locally:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).parent
MODELS_DIR = BASE / "models"
DATA_FILE = BASE / "data" / "tea_model_ready.csv"

st.set_page_config(page_title="Tea Auction Price — Ensemble Learning", page_icon="🍃", layout="wide")

FAMILY_COLORS = {"Baseline": "#adb5bd", "Single model": "#8ecae6", "Bagging": "#2a9d8f",
                 "Boosting": "#f4a261", "Stacking": "#e76f51"}
MODEL_FILES = {"Random Forest (bagging)": "random_forest.joblib",
               "Bagging (decision trees)": "bagging.joblib",
               "XGBoost (boosting)": "xgboost.joblib",
               "Stacking (RF + XGB + KNN + Ridge)": "stacking.joblib"}
MODEL_FAMILY = {"Random Forest (bagging)": "Bagging", "Bagging (decision trees)": "Bagging",
                "XGBoost (boosting)": "Boosting", "Stacking (RF + XGB + KNN + Ridge)": "Stacking"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def nice(s):
    return str(s).replace("_", " ").title()


# ------------------------------------------------------------------ loading
@st.cache_resource(show_spinner="Loading models…")
def load_models():
    # The stacking model already contains fitted copies of the Random Forest and XGBoost
    # (identical to the separately saved ones), so they are reused instead of loading them again.
    stack = joblib.load(MODELS_DIR / "stacking.joblib")
    bagging = joblib.load(MODELS_DIR / "bagging.joblib")
    models = {"Random Forest (bagging)": stack.named_estimators_["rf"],
              "Bagging (decision trees)": bagging,
              "XGBoost (boosting)": stack.named_estimators_["xgb"],
              "Stacking (RF + XGB + KNN + Ridge)": stack}
    for m in [bagging, stack.named_estimators_["rf"], stack.named_estimators_["xgb"]]:
        m.named_steps["model"].set_params(n_jobs=1)
    return models


@st.cache_data
def load_tables():
    meta = json.loads((MODELS_DIR / "metadata.json").read_text())
    results = pd.read_csv(MODELS_DIR / "results.csv")
    preds = pd.read_csv(MODELS_DIR / "test_predictions.csv", parse_dates=["sale_date"])
    types = pd.read_csv(MODELS_DIR / "tea_types.csv").astype(str)
    data = None
    if DATA_FILE.exists():
        data = pd.read_csv(DATA_FILE, parse_dates=["sale_date"])
        for c in data.columns:
            if c != "sale_date" and not pd.api.types.is_numeric_dtype(data[c]):
                data[c] = data[c].astype(str)
    return meta, results, preds, types, data


@st.cache_data(show_spinner="Computing tree-count curve…")
def bagging_curve(_rf, X_test, y_test_price):
    """Test MAE of the Random Forest using the first k trees (shows variance reduction)."""
    prep, forest = _rf.named_steps["prep"], _rf.named_steps["model"]
    Xt = prep.transform(X_test)
    all_preds = np.vstack([t.predict(Xt) for t in forest.estimators_])      # trees x rows (log price)
    cum = np.cumsum(all_preds, axis=0) / np.arange(1, len(all_preds) + 1)[:, None]
    ks = np.unique(np.round(np.logspace(0, np.log10(len(all_preds)), 25)).astype(int))
    mae = [np.mean(np.abs(np.exp(cum[k - 1]) - y_test_price)) for k in ks]
    single = [np.mean(np.abs(np.exp(all_preds[i]) - y_test_price)) for i in range(min(50, len(all_preds)))]
    return pd.DataFrame({"trees": ks, "MAE": mae}), float(np.mean(single))


@st.cache_data
def grouped_importance(_rf, features, categorical):
    """Impurity importance of the Random Forest, summed back to the original features."""
    prep, forest = _rf.named_steps["prep"], _rf.named_steps["model"]
    names = prep.get_feature_names_out()
    imp = pd.Series(forest.feature_importances_, index=names)
    out = {}
    for name, v in imp.items():
        part, col = name.split("__", 1)
        if part == "cat":
            col = next((c for c in categorical if col.startswith(c + "_")), col)
        out[col] = out.get(col, 0) + v
    return pd.Series(out).sort_values(ascending=False)


models = load_models()
meta, results, preds, types, data = load_tables()
CAT, NUM, FEATURES = meta["categorical"], meta["numeric"], meta["features"]



def page_estimator():
    st.title("Fair-price estimator")
    st.write("Choose a tea and the conditions. Every ensemble model estimates the auction price (LKR per kg) "
             "from the tea's characteristics, the month and the recent weather in its growing region.")

    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.subheader("1. The tea")
        t = types.copy()
        catalogue = st.selectbox("Catalogue", sorted(t.catalogue.unique()), format_func=nice)
        t = t[t.catalogue == catalogue]
        grade = st.selectbox("Grade", sorted(t.grade.unique()), format_func=nice)
        t = t[t.grade == grade]
        tier = st.selectbox("Quality tier", sorted(t.tier.unique()), format_func=nice)
        t = t[t.tier == tier]
        region = st.selectbox("Region", sorted(t.region.unique()), format_func=nice)
        t = t[t.region == region]
        t = t.assign(label=t.category.map(nice) + " · " + t.elevation.map(nice) + " elevation"
                     + np.where(t.segment != "unknown", " · " + t.segment.map(nice), ""))
        variant = st.selectbox("Variant", t.label.tolist()) if len(t) > 1 else t.label.iloc[0]
        tea = t[t.label == variant].iloc[0]
        if len(t) == 1:
            st.caption(f"Variant: {variant}")

        st.subheader("2. When")
        month = st.select_slider("Auction month", options=list(range(1, 13)), value=3,
                                 format_func=lambda m: MONTHS[m - 1])

    # typical weather for this region & month (defaults for the sliders)
    if data is not None:
        ref = data[(data.region == tea.region) & (data.sale_month == month)]
        if len(ref) == 0:
            ref = data[data.catalogue == tea.catalogue]
        typical = ref[NUM].median()
    else:
        typical = pd.Series({c: v[1] for c, v in meta["numeric_ranges"].items()})
    rng = meta["numeric_ranges"]

    with right:
        st.subheader("3. Recent weather in the growing region")
        st.caption("Defaults = typical values for this region in the chosen month. Move the sliders to test scenarios.")
        c1, c2, c3 = st.columns(3)
        temp = c1.slider("Temperature (°C)", float(rng["temperature_2m_mean_mean"][0]),
                         float(rng["temperature_2m_mean_mean"][2]),
                         float(round(typical["temperature_2m_mean_mean"], 1)), 0.1)
        rain = c2.slider("Weekly rainfall (mm)", 0.0, float(rng["precipitation_sum_total_lag1"][2]),
                         float(round(typical["precipitation_sum_total_lag1"], 0)), 1.0)
        sun = c3.slider("Weekly sunshine (hours)", 0.0, float(rng["sunshine_hours_total_lag1"][2]),
                        float(round(typical["sunshine_hours_total_lag1"], 0)), 1.0)
        c4, c5 = st.columns(2)
        severity = c4.slider("Weather severity in broker report (1–4)", 1.0, 4.0,
                             float(round(typical["avg_weather_severity"], 1)), 0.1)
        crop = c5.selectbox("Broker says crop is…", ["unknown", "increase", "maintained", "decrease"],
                            format_func=nice)
        with st.expander("Advanced: set each of the last 3 weeks separately"):
            a1, a2, a3 = st.columns(3)
            temps = [a1.number_input(f"Temp lag {k} (°C)", value=temp, key=f"t{k}") for k in (1, 2, 3)]
            rains = [a2.number_input(f"Rain lag {k} (mm)", value=rain, key=f"r{k}") for k in (1, 2, 3)]
            suns = [a3.number_input(f"Sun lag {k} (h)", value=sun, key=f"s{k}") for k in (1, 2, 3)]

    row = {c: tea[c] for c in CAT if c in tea.index}
    row["text_crop_change"] = crop
    row.update({
        "temperature_2m_mean_mean": temp,
        "temperature_2m_mean_mean_lag1": temps[0], "temperature_2m_mean_mean_lag2": temps[1],
        "temperature_2m_mean_mean_lag3": temps[2],
        "precipitation_sum_total_lag1": rains[0], "precipitation_sum_total_lag2": rains[1],
        "precipitation_sum_total_lag3": rains[2],
        "sunshine_hours_total_lag1": suns[0], "sunshine_hours_total_lag2": suns[1],
        "sunshine_hours_total_lag3": suns[2],
        "avg_weather_severity": severity, "sale_month": month,
        "weeks_since_start": rng["weeks_since_start"][2],          # "now" = latest week in the data
        "n_keywords": typical.get("n_keywords", rng["n_keywords"][1]),
    })
    X = pd.DataFrame([row])[FEATURES]

    st.divider()
    st.subheader("Estimated auction price")
    est = {name: float(np.exp(m.predict(X))[0]) for name, m in models.items()}
    best_label = min(MODEL_FILES, key=lambda k: results.set_index("model")["MAE"].get(
        {"Random Forest (bagging)": "Random Forest (300 trees)", "Bagging (decision trees)": "Bagging (100 trees)",
         "XGBoost (boosting)": "XGBoost (tuned)", "Stacking (RF + XGB + KNN + Ridge)": "Stacking (RF+XGB+KNN+Ridge)"}[k],
        np.inf) if k in est else np.inf)
    best_mae = float(best_row["MAE"])
    cols = st.columns(len(est))
    for col, (name, val) in zip(cols, est.items()):
        col.metric(name + (" ⭐" if name == best_label else ""), f"Rs {val:,.0f}")

    b = est.get(best_label, np.mean(list(est.values())))
    st.success(f"**Fair-price estimate: Rs {b:,.0f} per kg** (typical range Rs {b - best_mae:,.0f} – "
               f"Rs {b + best_mae:,.0f}, based on the best model's average test error of Rs {best_mae:.0f}).")

    fig = go.Figure(go.Bar(x=list(est.values()), y=list(est.keys()), orientation="h",
                           marker_color=[FAMILY_COLORS[MODEL_FAMILY[k]] for k in est],
                           text=[f"Rs {v:,.0f}" for v in est.values()], textposition="outside"))
    if data is not None:
        hist = data[data.product_stream == tea.product_stream]
        if len(hist):
            fig.add_vline(x=hist.price_mid_lkr.mean(), line_dash="dash", line_color="#6c757d",
                          annotation_text="historical average of this tea", annotation_position="top")
    fig.update_layout(height=280, margin=dict(l=10, r=40, t=30, b=10), xaxis_title="LKR per kg")
    st.plotly_chart(fig, width="stretch")

    if data is not None and len(hist):
        with st.expander("Price history of this tea"):
            h = px.line(hist.sort_values("sale_date"), x="sale_date", y="price_mid_lkr", markers=True,
                        labels={"sale_date": "", "price_mid_lkr": "LKR per kg"})
            h.add_hline(y=b, line_dash="dot", line_color="#e76f51", annotation_text="estimate")
            h.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(h, width="stretch")
    st.caption("The model estimates what this kind of tea should fetch under these conditions. It does not know "
               "last week's actual price, so it cannot anticipate sudden market moves (see Insights).")



def page_comparison():
    st.title("Model comparison")
    st.write("All models were trained on auctions from **Nov 2023 – Oct 2025** and tested on the latest "
             "**21 auctions (Oct 2025 – Mar 2026)** they never saw — a time-based split, like real use.")

    r = results.sort_values("MAE").reset_index(drop=True)
    k1, k2, k3, k4 = st.columns(4)
    single = r[r.family == "Single model"].iloc[0]
    tea_avg = r[r.model == "Tea-type average"].iloc[0] if (r.model == "Tea-type average").any() else None
    k1.metric("Best model", best_row["model"].split(" (")[0])
    k2.metric(f"Average error ({best_row['MAPE_%']:.1f}%)", f"Rs {best_row['MAE']:.0f}/kg")
    k3.metric("R²", f"{best_row['R2']:.3f}")
    k4.metric(f"Lower error than {single['model']}", f"{(1 - best_row['MAE'] / single['MAE']) * 100:.0f}%")

    fig = px.bar(r.sort_values("MAE", ascending=False), x="MAE", y="model", color="family", orientation="h",
                 color_discrete_map=FAMILY_COLORS, text=r.sort_values("MAE", ascending=False)["MAE"].round(0),
                 labels={"MAE": "Test MAE (LKR per kg) — lower is better", "model": ""})
    fig.update_yaxes(categoryorder="total descending")
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=30, b=10), legend_title="")
    st.plotly_chart(fig, width="stretch")

    show = r[["model", "family", "MAE", "RMSE", "R2", "MAPE_%"]].copy()
    if tea_avg is not None:
        show["vs tea-type average"] = ((1 - r["MAE"] / tea_avg["MAE"]) * 100).map("{:+.1f}%".format)
    st.dataframe(show.style.format({"MAE": "{:.1f}", "RMSE": "{:.1f}", "R2": "{:.3f}", "MAPE_%": "{:.2f}"}),
                 width="stretch", hide_index=True)
    st.caption("MAE = average error in rupees · RMSE = punishes large errors more · R² = share of price variation "
               "explained · MAPE = average % error")

    st.subheader("Predicted vs actual on the test period")
    pred_cols = [c for c in preds.columns if c not in ("sale_date", "catalogue", "product_stream", "actual")]
    default = pred_cols.index(best_row["model"]) if best_row["model"] in pred_cols else 0
    chosen = st.selectbox("Model", pred_cols, index=default)
    c1, c2 = st.columns(2)
    sc = px.scatter(preds, x="actual", y=chosen, color="catalogue", opacity=0.5,
                    labels={"actual": "Actual price (LKR)", chosen: "Predicted price (LKR)"})
    lim = [preds.actual.min() * 0.9, preds.actual.max() * 1.05]
    sc.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(dash="dash", color="black"), name="perfect"))
    sc.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
    c1.plotly_chart(sc, width="stretch")

    weekly = preds.groupby(["sale_date", "catalogue"])[["actual", chosen]].median().reset_index()
    cat = c2.selectbox("Catalogue", sorted(weekly.catalogue.unique()), format_func=nice)
    w = weekly[weekly.catalogue == cat].melt("sale_date", ["actual", chosen], var_name="series", value_name="LKR")
    ln = px.line(w, x="sale_date", y="LKR", color="series", markers=True,
                 color_discrete_sequence=["#264653", "#e76f51"], labels={"sale_date": ""})
    ln.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), legend_title="",
                     title=f"Median price per auction — {nice(cat)}")
    c2.plotly_chart(ln, width="stretch")

    err = preds.assign(err=(preds[chosen] - preds.actual).abs()).groupby("catalogue").agg(
        test_rows=("err", "size"), MAE=("err", "mean"), avg_price=("actual", "mean"))
    err["error_%"] = err.MAE / err.avg_price * 100
    st.write("**Error by catalogue**")
    st.dataframe(err.round(1).rename(index=nice), width="stretch")



def page_ensembles():
    st.title("How the ensembles work")
    tab1, tab2, tab3 = st.tabs(["Bagging", "Boosting", "Stacking"])

    with tab1:
        st.markdown("""
**Bagging (Bootstrap Aggregating)** trains many copies of the same model, each on a different random sample
of the training data (drawn *with replacement*), and **averages** their predictions.

- A single decision tree over-fits: small changes in the data give a very different tree (**high variance**).
- Averaging many different trees cancels out their individual mistakes → **lower variance**, better accuracy.
- **Random Forest** = bagging + a random subset of features at each split, making the trees even more different.
""")
        curve_file = MODELS_DIR / "bagging_curve.csv"
        if curve_file.exists():
            curve = pd.read_csv(curve_file)
            single_mae = float(curve["single_tree_MAE"].iloc[0])
            fig = px.line(curve, x="trees", y="MAE", markers=True, log_x=True,
                          labels={"trees": "Number of trees averaged (log scale)", "MAE": "Test MAE (LKR)"})
            fig.add_hline(y=single_mae, line_dash="dash", line_color="#e76f51",
                          annotation_text=f"average single tree ≈ Rs {single_mae:.0f}")
            fig.update_traces(line_color="#2a9d8f")
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                              title="Our Random Forest: error falls as more trees are averaged")
            st.plotly_chart(fig, width="stretch")
            st.info(f"One tree: about Rs {single_mae:.0f} error. All {int(curve.trees.max())} trees averaged: "
                    f"Rs {curve.MAE.iloc[-1]:.0f}. That drop is the variance reduction of bagging.")
        rb = results.set_index("model")
        if {"Bagging (100 trees)", "Random Forest (300 trees)"} <= set(rb.index):
            st.write(f"Plain bagging of trees: **Rs {rb.loc['Bagging (100 trees)', 'MAE']:.0f}** · "
                     f"Random Forest: **Rs {rb.loc['Random Forest (300 trees)', 'MAE']:.0f}** — the extra feature "
                     "randomness makes the trees more diverse, so averaging helps more.")

    with tab2:
        st.markdown("""
**Boosting** builds trees **one after another**. Each new tree focuses on the errors the previous trees made,
so the model improves step by step (mainly reduces **bias**). We used **XGBoost**.

| | Bagging (Random Forest) | Boosting (XGBoost) |
|---|---|---|
| How trees are built | Independently, in parallel | Sequentially, each fixes the last |
| Main effect | Reduces variance | Reduces bias |
| Risk | Little over-fitting | Can over-fit without tuning |
""")
        rb = results.set_index("model")
        rf_m = rb.loc[[m for m in rb.index if m.startswith("Random Forest")], "MAE"].min()
        xg_m = rb.loc[[m for m in rb.index if m.startswith(("XGBoost", "Gradient"))], "MAE"].min()
        st.info(f"On this data **bagging won**: Random Forest Rs {rf_m:.0f} vs XGBoost Rs {xg_m:.0f}.")

    with tab3:
        st.markdown("""
**Stacking** combines **different kinds** of models. Each base model makes a prediction, and a final
**meta-model** learns how much to trust each one. The meta-model is trained on *out-of-fold* predictions
(predictions on rows a base model did not train on), so it does not simply trust over-fitted models.

`Random Forest + XGBoost + KNN + Ridge  →  meta-model (weights ≥ 0)  →  final price`
""")
        w = pd.Series(meta.get("stack_weights", {}))
        if len(w):
            w.index = [{"rf": "Random Forest", "xgb": "XGBoost", "knn": "KNN", "ridge": "Ridge", "gb": "Gradient Boosting"}
                       .get(i, i) for i in w.index]
            fig = px.bar(x=w.values * 100, y=w.index, orientation="h", text=[f"{v:.1%}" for v in w.values],
                         labels={"x": "Share of the final prediction (%)", "y": ""},
                         color_discrete_sequence=["#264653"])
            fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10),
                              title="How much the meta-model trusts each base model")
            st.plotly_chart(fig, width="stretch")
            st.info("The meta-model learned that Random Forest was by far the strongest base model and relied "
                    "almost entirely on it — so stacking matched Random Forest rather than beating it. "
                    "Stacking helps most when the base models are similarly strong but make different mistakes.")



def page_insights():
    st.title("What drives tea prices?")
    if "Random Forest (bagging)" in models:
        imp_file = MODELS_DIR / "feature_importance.csv"
        imp = (pd.read_csv(imp_file, index_col=0)["importance"] if imp_file.exists()
               else grouped_importance(models["Random Forest (bagging)"], FEATURES, CAT))
        labels = {"tier": "Quality tier", "category": "Category", "grade": "Grade", "catalogue": "Catalogue",
                  "elevation": "Elevation", "segment": "Segment", "region": "Region",
                  "weeks_since_start": "Time trend", "sale_month": "Month", "text_crop_change": "Crop change (report)",
                  "avg_weather_severity": "Weather severity", "n_keywords": "Report keywords",
                  "temperature_2m_mean_mean": "Temperature (sale week)",
                  "temperature_2m_mean_mean_lag1": "Temperature 1 week before",
                  "temperature_2m_mean_mean_lag2": "Temperature 2 weeks before",
                  "temperature_2m_mean_mean_lag3": "Temperature 3 weeks before",
                  "precipitation_sum_total_lag1": "Rainfall 1 week before",
                  "precipitation_sum_total_lag2": "Rainfall 2 weeks before",
                  "precipitation_sum_total_lag3": "Rainfall 3 weeks before",
                  "sunshine_hours_total_lag1": "Sunshine 1 week before",
                  "sunshine_hours_total_lag2": "Sunshine 2 weeks before",
                  "sunshine_hours_total_lag3": "Sunshine 3 weeks before"}
        top = imp.head(12)
        fig = px.bar(x=top.values[::-1] * 100, y=[labels.get(i, i.replace("_", " ")) for i in top.index[::-1]],
                     orientation="h", labels={"x": "Importance (%)", "y": ""}, color_discrete_sequence=["#2a9d8f"])
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10),
                          title="Feature importance (Random Forest)")
        st.plotly_chart(fig, width="stretch")
        st.write("**Tea characteristics (tier, category, grade, catalogue, elevation) decide most of the price.** "
                 "Weather has a smaller effect.")

    if data is not None:
        c1, c2 = st.columns(2)
        order = data.groupby("tier").price_mid_lkr.median().sort_values().index
        b1 = px.box(data, x="tier", y="price_mid_lkr", category_orders={"tier": list(order)}, points=False,
                    labels={"tier": "", "price_mid_lkr": "LKR per kg"}, color_discrete_sequence=["#f4a261"])
        b1.update_layout(height=360, title="Price by quality tier", margin=dict(l=10, r=10, t=40, b=10))
        c1.plotly_chart(b1, width="stretch")
        order = data.groupby("catalogue").price_mid_lkr.median().sort_values().index
        b2 = px.box(data, x="catalogue", y="price_mid_lkr", category_orders={"catalogue": list(order)},
                    points=False, labels={"catalogue": "", "price_mid_lkr": "LKR per kg"},
                    color_discrete_sequence=["#2a9d8f"])
        b2.update_layout(height=360, title="Price by catalogue", margin=dict(l=10, r=10, t=40, b=10))
        c2.plotly_chart(b2, width="stretch")

        trend = data.groupby(["sale_date", "catalogue"]).price_mid_lkr.median().reset_index()
        tr = px.line(trend, x="sale_date", y="price_mid_lkr", color="catalogue", markers=True,
                     labels={"sale_date": "", "price_mid_lkr": "Median LKR per kg"})
        tr.update_layout(height=380, title="Median auction price over time", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(tr, width="stretch")

        st.subheader("Can anything beat \"same price as last week\"?")
        test = data[data.split == "test"]
        naive = np.mean(np.abs(test.price_mid_lkr - test.price_lag1))
        change = np.median(np.abs(test.price_mid_lkr / test.price_lag1 - 1)) * 100
        m1, m2, m3 = st.columns(3)
        m1.metric("Naive 'same as last week' MAE", f"Rs {naive:.0f}")
        m2.metric("Fair-price model MAE", f"Rs {best_row['MAE']:.0f}")
        m3.metric("Typical weekly price change", f"{change:.1f}%")
        st.write("Tea prices move only ~2% per week, and the direction is close to random — short-term prices behave "
                 "like a **random walk**. When last week's price is known, repeating it is very hard to beat "
                 "(a Random Forest given the price history did not beat it either). The fair-price model answers a "
                 "different question: *what should this kind of tea fetch under these conditions?* — useful when "
                 "there is no recent price, for planning ahead, and for judging whether a price is fair.")



def page_about():
    st.title("About the data and method")
    st.markdown("""
**Dataset:** *Weather-Driven Sri Lankan Tea Market Catalogues* (Hugging Face `colombo-tea-auction-prices/cta-price-prediction`, CC BY 4.0)
- Prices from 105 weekly **Forbes & Walker Tea Brokers** auction reports, Nov 2023 – Mar 2026
- Weather from the **Open-Meteo** historical archive (temperature, rainfall, sunshine; 1–3 weeks before each sale)
- Four catalogues: Low Grown, High Grown, Dust, Off-Grade

**Preparation (Part 1)**
- Removed rows without a price (436), parsed dates, standardised text, converted sunshine to hours
- Found that the dataset's 2024 USD/LKR exchange rates were wrong (≈198 and ≈360 vs a real range of ≈290–324) → column dropped
- Fixed repeated sale dates and flagged a 17-week gap in early 2025
- Time-based split: train Nov 2023 – Oct 2025, test Oct 2025 – Mar 2026

**Modelling (Part 2)**
- Target: log(price); features: tea characteristics, month, weather, broker-report text
- Baselines (averages, Linear Regression, Decision Tree, KNN) → **Bagging** (BaggingRegressor, Random Forest)
  → **Boosting** (XGBoost) → **Stacking** (RF + XGBoost + KNN + Ridge, non-negative meta-model)
- Time-aware cross-validation for hyperparameter tuning

**Limitations**
- Short history (≈2.5 years) and a gap in early 2025
- Quoted price ranges (mid-point), not individual lot prices
- Weather from a few representative locations per region
- The model cannot anticipate sudden market shocks
""")
    if data is not None:
        st.write("**Sample of the modelling data**")
        st.dataframe(data.head(20), width="stretch")


# ------------------------------------------------------------------ layout
best_row = results[results.family != "Baseline"].sort_values("MAE").iloc[0]

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0f3b2e; }
[data-testid="stSidebar"] * { color: #e8f3ef; }
[data-testid="stSidebarNav"] { padding-top: 0.5rem; }
[data-testid="stSidebarNavLink"] { border-radius: 8px; margin: 2px 0; padding: 6px 10px; }
[data-testid="stSidebarNavLink"]:hover { background: rgba(255,255,255,0.08); }
[data-testid="stSidebarNavLink"][aria-current="page"] { background: #2a9d8f; }
[data-testid="stSidebarNavLink"][aria-current="page"] span { color: #ffffff !important; font-weight: 600; }
[data-testid="stNavSectionHeader"] { color: #9fd8cb !important; text-transform: uppercase;
                                     letter-spacing: 0.08em; font-size: 0.72rem; margin-top: 0.8rem; }
.side-card { background: rgba(255,255,255,0.07); border-radius: 10px; padding: 12px 14px; margin: 6px 0 10px 0; }
.side-card .label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; color: #9fd8cb; }
.side-card .value { font-size: 1.35rem; font-weight: 700; color: #ffffff; line-height: 1.3; }
.side-card .sub { font-size: 0.8rem; color: #cfe6df; }
[data-testid="stSidebarHeader"] { padding-top: 1.2rem; }
[data-testid="stSidebarLogo"] { height: 3.4rem !important; }
[data-testid="stMetric"] { background: #f1f7f5; border: 1px solid #dcebe6; border-radius: 10px; padding: 12px 14px; }
</style>
""", unsafe_allow_html=True)

st.logo(str(BASE / "assets" / "logo.svg"), size="large")

nav = st.navigation({
    "Predict": [st.Page(page_estimator, title="Price estimator", icon=":material/payments:", default=True)],
    "Models": [st.Page(page_comparison, title="Model comparison", icon=":material/leaderboard:"),
               st.Page(page_ensembles, title="How ensembles work", icon=":material/account_tree:")],
    "Data": [st.Page(page_insights, title="Insights", icon=":material/insights:"),
             st.Page(page_about, title="About the data", icon=":material/info:")],
})

with st.sidebar:
    st.markdown(
        f"""<div class="side-card"><div class="label">Best model</div>
        <div class="value">{best_row['model'].split(' (')[0]}</div>
        <div class="sub">Average error Rs {best_row['MAE']:.0f}/kg · {best_row['MAPE_%']:.1f}% · R² {best_row['R2']:.3f}</div></div>""",
        unsafe_allow_html=True)
    st.markdown(
        f"""<div class="side-card"><div class="label">Test period</div>
        <div class="value">21 auctions</div>
        <div class="sub">Oct 2025 – Mar 2026, never seen in training</div></div>""",
        unsafe_allow_html=True)
    st.caption("Data: Forbes & Walker auction reports + Open-Meteo weather · Hugging Face, CC BY 4.0")

nav.run()
