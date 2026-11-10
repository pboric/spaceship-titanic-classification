# Spaceship Titanic: Passenger Transport Prediction

Predicting which passengers of the fictional Spaceship Titanic were transported to another dimension (Kaggle
"Spaceship Titanic" competition): a tabular binary classification project from exploratory analysis and statistical
tests to a stacking ensemble, served through a small Flask web app.

## Results

Hold-out test split of the training data:

| Model | Accuracy | F1 |
|---|---:|---:|
| **Stacking ensemble** | **0.796** | **0.806** |
| LightGBM | 0.793 | 0.802 |
| Soft-voting ensemble | 0.787 | 0.800 |
| Logistic regression | 0.783 | 0.795 |
| AdaBoost | 0.782 | 0.797 |

The stacking ensemble reaches a ROC AUC of 0.886.

## What the notebook covers

1. **EDA:** univariate and bivariate analysis of the passenger features and their relation to the target.
2. **Statistical inference:** hypothesis tests, for example whether travelling alone is related to being
   transported (proportion z-test).
3. **Cleaning and feature engineering:** missing values, outliers, and new features such as cabin deck and side,
   passenger group and travelling alone.
4. **Modelling:** a quick comparison of many models with LazyPredict, Optuna tuning of the best ones, then voting and
   stacking ensembles.

## Repository

| File | Content |
|---|---|
| [`spaceship_titanic.ipynb`](spaceship_titanic.ipynb) | The full analysis and modelling |
| `utils.py` | Helper functions used by the notebook |
| `train.csv`, `test.csv` | Competition data |
| `stacking_ensemble_model.pkl` | The trained stacking model |
| `app.py`, `templates/`, `Dockerfile` | Flask web app: upload a CSV of passengers (same columns as `test.csv`) and see the share predicted to be transported |

## Run it

```bash
pip install -r requirements.txt            # Python 3.10; versions pinned so the saved model loads
jupyter notebook spaceship_titanic.ipynb   # the analysis
python app.py                              # the web app on http://localhost:5000
```

Or with Docker:

```bash
docker build -t spaceship-titanic .
docker run -p 5000:5000 spaceship-titanic
```

## Data

[Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) on Kaggle; `train.csv` and `test.csv` are
included.

## License

MIT
