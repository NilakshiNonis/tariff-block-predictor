# Predicting Top-Tariff-Block Electricity Use in Sri Lankan Households

IT 5504 Applied Machine Learning - final assignment. Author: P. A. R. N. Nonis (269154E).

**Live demo:** https://sri-lankan-domestic-tariff-block-predictor.streamlit.app/

## The problem
Will a household's electricity consumption **next month exceed 180 kWh**, the top block of the Sri Lankan domestic tariff?
The model uses the last three months of consumption, the calendar month and a few household traits (RECON-SL dataset, ~4,000 households).

## Method
- **Main model:** Explainable Boosting Machine (EBM), an interpretable boosted additive model.
- **Benchmarks:** logistic regression, random forest, XGBoost, plus majority-class and "same as last month" baselines.
- **Evaluation:** time-based split (train / validation / test = earliest / next 3 / final 3 months), rolling-origin stability check, bootstrap intervals.
- **Explainability:** EBM global importances and shape functions, exact local contributions, LIME cross-check.

## Results (final 3 months, never seen in training)
| | Train | Test |
|---|---|---|
| EBM accuracy | 95.5% | 95.1% (95% CI 94.5-95.6%) |
| "Same as last month" rule | 95.0% | 95.6% |

Test ROC-AUC 0.983, precision 0.93, recall 0.81, F1 0.86. A trivial persistence rule matches the accuracy, so the model's value is calibrated
probabilities, explanations and what-if analysis. See the report for the full discussion and limitations.

## Run the app locally
```
python -m venv venv
venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app_tariff.py
```
The trained model is included in `results_cls/`, so no training is needed.

## Reproduce everything
1. Download RECON-SL (https://ieee-dataport.org/open-access/residential-electricity-consumption-dataset-sri-lanka-recon-sl) and copy `monthly_consumption.csv`,
   `w1_household_information_and_history.csv`, `w1_room_roster.csv`, `w1_appliances.csv`, `w1_demographics.csv` into a `data/` folder.
2. `pip install -r requirements-training.txt`
3. `python 01_build_dataset.py --data_dir data --out model_table.csv`
4. `python 04_tariff_block_classifier.py --table model_table.csv --out results_cls`   (about 6 minutes)
5. Optional: `python 02_train_evaluate.py ...` then `python 03_ebm_main.py ...` (survey-only regression experiment), and `python build_report.py` to rebuild the PDF.

## Files
`app_tariff.py` Streamlit app | `04_tariff_block_classifier.py` main pipeline | `01_build_dataset.py` data preparation |
`02/03` supplementary experiment (+ `app_survey_regression.py`) | `results_cls/` outputs and saved model | 
`DATASET.md` data description | `Dockerfile` container.

Seeds are fixed to 42. The app is for awareness and planning, not billing.
