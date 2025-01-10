# 02476-Machine-Learning-Operations-Project

###### Group number
58

###### Members:
Tobias Rodrigues Bjerre (S234823) \
Magnus Sehested Thormann (S234830) \
Laura Munch Bjerg (S234865) 

#### Overall goal of the project
In this project, we wish to use Natural Language Processing on a dataset of tweets, predicting if a given tweet is about a real disaster or not.

#### What framework are you going to use, and you do you intend to include the framework into your project?
We intend to use the Transformers framework by HuggingFace. This framework provides an extensive library of pretrained models for NLP tasks, including text classification. By using pretrained models, we reduce the computational cost and time required for training from scratch, while hopefully achieving similar performance.

#### What data are you going to run on (initially, may change)
We will initially train and evaluate our model using the Kaggle dataset "Natural Language Processing with Disaster Tweets". This dataset consists of approx. 10000 tweets

#### What models do you expect to use
We plan on using the RoBERTa (Robustly Optimized BERT Approach) model for this classification task. We currently intend to use pretrained weights for the most part, but this might change if time allows.



## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).