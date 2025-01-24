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
We plan on using the RoBERTa (Robustly Optimized BERT Approach) model for this classification task. We currently intend to use pretrained weights for the most part, but this might change if time allows. \


## Deployment

### Frontend
Deployed to Cloud Run:
[Frontend Service URL](https://twitter-frontend-791862686266.europe-west1.run.app)

### Backend
Deployed to Cloud Run:
[Backend Service URL](https://twitter-backend-791862686266.europe-west1.run.app/docs)
