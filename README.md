Machine Learning based Modeling of Near-Wall Turbulence
==============================

Machine Learning based Modeling of Near-Wall Turbulence

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from external sources. For example the 
    │   │                     Validation data from Kawamura lab: https://www.rs.tus.ac.jp/t2lab/db/
    │   │
    │   ├── interim        <- Intermediate data that has been transformed. Plane slices osv.
    │   ├── processed      <- The final, canonical data sets for modeling. statistic data osv.
    │   └── raw            <- The original, immutable data dump. Data after being read by readBinary.
    │   └── models         <- trained models
    │   └── model_output   <- model output
    │
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
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to load or read data
    │   │   └── make_dataset.py
    │   ├── intermediate           <- Scripts to transform data from raw to intermediate
    │   │   └── make_dataset.py
    │   ├── processed           <- Scripts to turn intermediate data into modelling input
    │   │   └── make_dataset.py
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   ├── modelling         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   ├── model_evaluation         <- Scripts that analyse model performance and model selection.
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
