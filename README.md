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
│   └── config.yaml
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
├── logs/                     # log files
│   └── data_pipeline.log/
├── models/                   # Trained models
│   └── bert_disaster_tweets/
│   │    ├── config.json
│   │    ├── model.safetensors
│   │    ├── special_tokens_map.json
│   │    ├── tokenizer_config.json
│   │    ├── tokenizer.json
│   │    └── vocab.txt
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── twitter_classification/
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
├── submission.csv            # Output file
└── tasks.py                  # Project tasks
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).


## Project progress
**Week 1**
- [ ] Setup version control for your data or part of your data (M8)
- [ ] Add command line interfaces and project commands to your code where it makes sense (M9)
- [ ] Used Hydra to load the configurations and manage your hyperparameters (M11)
- [ ] Use profiling to optimize your code (M12)
- [ ] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
- [ ] Consider running a hyperparameter optimization sweep (M14)
- [ ] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)
- [x] Build the docker files locally and make sure they work as intended (M10)¨
- [x] Write one or multiple configurations files for your experiments (M11)
- [x] Construct one or multiple docker files for your code (M10)
- [x] Create a dedicated environment for you project to keep track of your packages (M2)
- [x] Do a bit of code typing and remember to document essential parts of your code (M7)
- [x] Use logging to log important events in your code (M14)
- [x] Remember to comply with good coding practices (pep8) while doing the project (M7)
- [x] Remember to fill out the requirements.txt and requirements_dev.txt file with whatever dependencies that you are using (M2+M6)
- [x] Add a model to model.py and a training procedure to train.py and get that running (M6)
- [x] Fill out the data.py file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
- [x] Create the initial file structure using cookiecutter with an appropriate template (M6)
- [x] Make sure that all team members have write access to the GitHub repository (M5) 
- [x] Create a git repository (M5)


**Week 2**
- [ ] Write unit tests related to the data part of your code (M16) 
- [ ] Write unit tests related to model construction and or model training (M16) 
- [ ] Calculate the code coverage (M16) 
- [ ] Get some continuous integration running on the GitHub repository (M17) 
- [ ] Add caching and multi-os/python/pytorch testing to your continuous integration (M17) 
- [ ] Add a linting step to your continuous integration (M17) Add pre-commit hooks to your version control setup (M18) 
- [ ] Add a continues workflow that triggers when data changes (M19) 
- [ ] Add a continues workflow that triggers when changes to the model registry is made (M19) 
- [ ] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21) 
- [ ] Create a trigger workflow for automatically building your docker images (M21) 
- [ ] Get your model training in GCP using either the Engine or Vertex AI (M21) 
- [ ] Create a FastAPI application that can do inference using your model (M22) 
- [ ] Deploy your model in GCP using either Functions or Run as the backend (M23) 
- [ ] Write API tests for your application and setup continues integration for these (M24) 
- [ ] Load test your application (M24) 
- [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25) 
- [ ] Create a frontend for your API (M26)


**Week 3** 
- [ ] Check how robust your model is towards data drifting (M27)
- [ ] Deploy to the cloud a drift detection API (M27)
- [ ] Instrument your API with a couple of system metrics (M28)
- [ ] Setup cloud monitoring of your instrumented application (M28)
- [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
- [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29) 
- [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30) 
- [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)


**Extra**
- [ ] Write some documentation for your application (M32) 
- [ ] Publish the documentation to GitHub Pages (M32) 
- [ ] Revisit your initial project description. Did the project turn out as you wanted? 
- [ ] Create an architectural diagram over your MLOps pipeline 
- [ ] Make sure all group members have an understanding about all parts of the project 
- [ ] Uploaded all your code to GitHub
