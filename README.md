# Data to Design: Structural Materials (Da2DeSM)

Repository for the Da2DeSM project-based course at Bauhaus-Universität Weimar. This project applies machine learning to predict the mechanical properties of structural materials.

## Main Goals

*   Predict material behavior with limited data.
*   Analyze model uncertainty and out-of-distribution performance.
*   Turn machine learning outputs into engineering insights.

## Dataset & Tasks

We work with a concrete dataset collection to predict concrete compressive strength.

*   **EDA**: Exploratory analysis and feature inspection.
*   **Feature Engineering**: Encoding mix proportions and material descriptors.
*   **Interpretability**: Checking if models match physical reality.

## Classical Machine Learning Models

*   Linear models (OLS, Elastic Net, Bayesian Ridge, Polynomial Ridge) - assigned to Brijesh Dholakiya
*   Support Vector Machines (KNN, SVR, NuSVR) - assigned to Shoaib Ahmad Joo
*   Ensemble models (Decision tree, Random Forest, Extra Trees) - assigned to Anmol Bradhwaj
*   Boosting models (XGBoost, HistGradBoost, Adaboost) - assigned to Jasurbek Odilov

## Repository Structure

*   `notebooks/` — Jupyter notebooks for data analysis and model training.
*   `data/` — Concrete mixture and strength datasets.
*   `requirements.txt` — Python package dependencies.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com
   ```
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Open Jupyter to run the notebooks.

## Contributors

*   **Brijesh** ([@BrijeshDholakiya](https://github.com/BrijeshDholakiya))
*   **Shoaib Ahmad Joo** ([@shoaib-joo](https://github.com/shoaib-joo))
*   **ANMOL BHARDWAJ** ([@Anmol040](https://github.com/Anmol040))
*   **Jasurbek Odilov** ([@Jocund96](https://github.com/Jocund96))

## Course Info
*   **Chair**: Data Science in Civil Engineering, Bauhaus-Universität Weimar.
*   **Instructors**: Marchellino Ghorayeb, Stefan Kollmannsberger.
