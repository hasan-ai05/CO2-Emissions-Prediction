# CO2 Emissions Prediction

A regression model that predicts vehicle CO2 emissions from engine specifications alone.
Fuel consumption data was deliberately excluded to prevent data leakage and to simulate
a realistic use case: estimating emissions before a vehicle is certified or built.

---

## The Problem

CO2 = Fuel Consumption × 23.2 is a fixed physical constant for petrol engines.
If fuel consumption is included as a feature, the model learns a multiplication
table rather than a predictive relationship. The result looks impressive on paper
and is completely useless in practice.

This project answers the harder question: **how much of CO2 variance is explained
by engine design alone?**

---

## Dataset

**Source:** Canada Vehicle Emissions Registry — Government of Canada Open Data  
**Records:** 7,385 vehicles (2000–2022)  
**Target:** CO2 Emissions (g/km), range 96–522

| Feature Used | Description |
|---|---|
| Engine Size (L) | Displacement in litres |
| Cylinders | Number of cylinders |
| Transmission | Type encoded (A/M/AM/AV) |
| Fuel Type | Petrol / Diesel / Ethanol / Natural Gas |
| Vehicle Class | SUV / Sedan / Pickup / etc. |

**Excluded deliberately:** Fuel Consumption City, Fuel Consumption Hwy, Fuel Consumption Comb.
Including any of these would give R² ≈ 1.0 — not a model, a formula.

---

## Results

| Model | MAE (g/km) | RMSE (g/km) | R² |
|---|---|---|---|
| Random Forest | **14.39** | **19.69** | **0.8926** |
| SVR | 18.08 | 24.09 | 0.8392 |
| Ridge Regression | 21.40 | 27.97 | 0.7833 |

Random Forest explains **89.26%** of CO2 variance from 5 engine features.
The remaining 10.74% reflects aerodynamics, vehicle weight, and driving conditions
— none of which appear in the dataset.

**Cross-validation (5-fold KFold):** 0.885 ± 0.013 — consistent across all splits.

**Residuals:** Mean = 0.97 g/km (no systematic bias). Std = 19.66 g/km.

---

## Why Random Forest Won

The relationship between engine size and CO2 is not linear.
The increase from 4 to 6 cylinders does not produce the same CO2 change
as the increase from 6 to 8. Ridge Regression assumes linearity and
cannot capture this. Random Forest learns the thresholds directly.

---

## Correlation with CO2 (engine features only)

```
Engine Size (L)     0.855
Cylinders           0.835
Vehicle Class       0.301
Transmission        0.169
Fuel Type           0.093
```

Engine size and cylinder count together explain most of the predictable variance.
The remaining features add marginal signal.

![EDA Dashboard](images/eda_dashboard.png)

---

## SHAP Explainability

SHAP TreeExplainer decomposes every prediction into feature contributions.

**Highest-emitting vehicle in test set:**

```
Actual CO2    : 522 g/km
Predicted CO2 : 475 g/km
Engine Size   : 8.0 L
Cylinders     : 16
```

The SHAP waterfall shows engine size and cylinder count pushing the prediction
upward by the largest margins — consistent with physical expectations.

![SHAP Summary](images/shap_summary.png)
![SHAP Waterfall](images/shap_waterfall.png)

---

## Predicted vs Actual

Points cluster tightly along the diagonal across the full range (96–522 g/km).
Larger errors appear at extreme values — the model has fewer training examples
for 16-cylinder configurations and generalizes less precisely there.

![Predicted vs Actual](images/predicted_vs_actual.png)

---

## How to Run

```bash
pip install kagglehub shap scikit-learn pandas numpy plotly
jupyter notebook CO2_Emissions_Prediction.ipynb
```

Dataset downloads automatically via `kagglehub`.

---

## Where to Place Screenshots

```
co2-emissions-prediction/
    images/
        eda_dashboard.png         Section 5 — EDA subplots
        model_comparison.png      Section 9 — R² bar chart
        predicted_vs_actual.png   Section 10 — scatter + residuals
        shap_summary.png          Section 11 — SHAP beeswarm
        shap_waterfall.png        Section 12 — highest emitter waterfall
        executive_dashboard.png   Section 14 — full dashboard
    README.md
    CO2_Emissions_Prediction.ipynb
```

---

## Project Structure

```
CO2_Emissions_Prediction.ipynb    main notebook (14 sections)
README.md                         this file
images/                           screenshots from notebook output
```

---

*Part of an 8-project AI Engineering portfolio — Hasan Akhras*
