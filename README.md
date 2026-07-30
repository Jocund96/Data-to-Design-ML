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

<<<<<<< HEAD
## Models

### Classical Machine Learning
*   Linear models (OLS, Elastic Net, Bayesian Ridge, Polynomial Ridge) - assigned to Brijesh Dholakiya
*   Kernel and SVMs (KNN, SVR, NuSVR) - assigned to Shoaib Ahmad Joo
*   Ensemble models (Decision Tree, Random Forest, Extra Trees) - assigned to Anmol Bhardwaj
*   Boosting models (XGBoost, HistGradientBoosting, AdaBoost) - assigned to Jasurbek Odilov

## Repository Structure

Each track is an independent student workstream (not a progressive pipeline), organized by week within its own folder:
=======
## Classical Machine Learning Models

*   Linear models (OLS, Elastic Net, Bayesian Ridge, Polynomial Ridge) - assigned to Brijesh Dholakiya
*   Support Vector Machines (KNN, SVR, NuSVR) - assigned to Shoaib Ahmad Joo
*   Ensemble models (Decision tree, Random Forest, Extra Trees) - assigned to Anmol Bradhwaj
*   Boosting models (XGBoost, HistGradBoost, Adaboost) - assigned to Jasurbek Odilov

## Repository Structure
>>>>>>> 28291a868cd5a64d4645f9181cd61916a2bb49f5

*   `S1_Linear/` — Linear models.
*   `S2_Kernel/` — Kernel and SVM-based models.
*   `S3_Tree/` — Tree-based ensemble models.
*   `S4_Boosting/` — Boosting models.
*   `requirements.txt` — Consolidated Python package dependencies for all tracks.

<<<<<<< HEAD
Each track folder holds its own datasets, notebooks (grouped by week), and helper/utility modules.

=======
>>>>>>> 28291a868cd5a64d4645f9181cd61916a2bb49f5
## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Jocund96/Data-to-Design-ML.git
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
