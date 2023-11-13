FlightDataThesisProject
==============================

Customer segmentation on the basis of travel agency data

Project developed for my Master Thesis as a part of the program of Data Analytics and Business Economics at Lund University


## Title: Estimating the Impact of Website Changes on Conversion Rates
A Double Machine Learning Approach

Supervisor: Simon Reese

### Abstract
This study sought to evaluate the historical impact of changes to an ordering page of an online travel agency on its conversion rates. Data gathered from the website over a year, detailing aspects such as travel dates, prices, itineraries, number of passengers, travel time, and carriers, was analyzed. External data sources were also included, with the dataset covering 12 changes to the website’s layout and payment process. The changes’ effectiveness was assessed using three methods: comparing conversion rates before and after the changes, a modified linear regression model, and the Double Machine Learning (DML) method with Random Forests as the base learners. The analysis revealed that the only modification with a statistically significant positive impact on conversion rates was related bug fixing. Most changes did not significantly affect conversion rates, and some even demonstrated a non-significant negative impact. The DML method proved a useful tool in this context, outperforming simpler comparison methods with better control for confounding variables and reducing potential bias in Average Treatment Effect (ATE) estimation. However, estimates from the DML model were sensitive to the analysis time window. This study suggests future website design should focus on user-friendly and intuitive design, clear and detailed information provision, and careful evaluation of changes’ potential impact on user experience.

Keywords: website design, causal inference, observational study, average treatment effects, double machine learning


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
