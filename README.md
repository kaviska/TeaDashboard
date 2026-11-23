# Colombo Tea Auction — Ensemble Learning Dashboard

A Streamlit dashboard that estimates the fair auction price of Sri Lankan tea (LKR per kg) using
**bagging** (Random Forest, Bagging), **gradient boosting** (XGBoost, LightGBM, CatBoost) and **stacking** ensembles.

## Pages
1. **Price estimator** — pick a tea (catalogue, grade, tier, region), month and weather → price from every model
2. **Model comparison** — single models vs bagging vs boosting vs stacking on the test period
3. **How the ensembles work** — bagging tree-count curve, Random Forest vs gradient boosting, stacking weights
4. **Insights** — what drives tea prices, and the "same as last week" finding
5. **About the data** — dataset, cleaning steps, limitations

## Folder structure
```
tea_dashboard/
├── app.py                  # the dashboard
├── requirements.txt        # exact library versions (must match the versions that saved the models)
├── .streamlit/config.toml  # colours
├── assets/logo.svg         # sidebar logo
├── data/tea_model_ready.csv
└── models/                 # stacking (contains RF, XGBoost, LightGBM, CatBoost) + bagging, results, charts
```

## Run on your own computer
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy free on Streamlit Community Cloud
1. Create a free account at https://github.com and make a **new public repository** (e.g. `tea-price-dashboard`).
2. Click **Add file → Upload files** and upload everything in this folder, keeping the folders
   (`models/`, `data/`, `assets/`, `.streamlit/`). Each file is under GitHub's 25 MB browser-upload limit.
   Tip: `.streamlit` is a hidden folder; if it doesn't upload, create the file
   `.streamlit/config.toml` in GitHub with **Add file → Create new file** and paste its contents.
3. Go to https://share.streamlit.io, sign in with GitHub, click **Create app → Deploy a public app from GitHub**.
4. Choose your repository, branch `main`, main file `app.py`.
5. Open **Advanced settings** and choose **Python 3.13**, then click **Deploy**.
6. Wait 2–5 minutes. You get a public link like `https://tea-price-dashboard.streamlit.app` to put in your submission.

## Using your own Colab models instead (optional)
The `models/` folder already contains working models. To use the ones from your own Colab run:
1. Copy `stacking.joblib`, `bagging.joblib`, `results.csv`, `test_predictions.csv`, `tea_types.csv` and
   `metadata.json` from your Part 2 output into `models/`. The app takes the Random Forest, XGBoost, LightGBM and CatBoost
   from inside the stacking model, so their separate files are not needed (this keeps memory use low).
2. In Colab, run:
   ```python
   import sklearn, xgboost, numpy, pandas, joblib, sys
   print(sys.version)
   print(f"scikit-learn=={sklearn.__version__}\nxgboost=={xgboost.__version__}\n"
         f"numpy=={numpy.__version__}\npandas=={pandas.__version__}\njoblib=={joblib.__version__}")
   ```
3. Put those exact versions in `requirements.txt`, and choose the same Python version when deploying.
   Models saved with one scikit-learn version may not load with another.

## Data
*Weather-Driven Sri Lankan Tea Market Catalogues* — Hugging Face `colombo-tea-auction-prices/cta-price-prediction`
(CC BY 4.0). Prices from Forbes & Walker weekly auction reports; weather from Open-Meteo.
