# Social Media Sentiment Analysis
## Project Overview:
The project involves sentiment analysis using the Sentiment140 dataset, which includes data preprocessing, feature extraction using TfidfVectorizer, and training a logistic regression model. It demonstrates text cleaning, stemming, and evaluation of model performance through accuracy metrics. The approach effectively applies NLP techniques to classify sentiments as positive or negative.

## Requirements:
Python 3.x
pandas
numpy
TfidfVectorizer
matplotlib
seaborn

## Dataset:
Dataset Name: Sentiment140
Size: Contains 1.6 million labeled tweets.
Purpose: Used for sentiment analysis to classify tweets as positive or negative.
Columns:
Polarity: Sentiment label (0 for negative, 4 for positive).
Text: The tweet content.
Tweet ID: Unique identifier for each tweet.
Date: The date and time when the tweet was posted.
Query: Search query used (often empty).
Username: The username of the person who posted the tweet.
Usage: Primarily focused on analyzing the tweet text and sentiment labels for classification tasks.

## Steps:
### Dataset Acquisition:

The Sentiment140 dataset was downloaded from Kaggle using the Kaggle API and extracted for use.

### Data Preprocessing:

Loading the Data: The dataset was loaded into a pandas DataFrame for manipulation.
Text Cleaning: Applied techniques like removing special characters, URLs, and converting text to lowercase.
Tokenization: Split text into individual words.
Stopword Removal: Removed common stopwords to reduce noise.
Stemming: Used stemming to reduce words to their base forms (e.g., "running" to "run").
Feature Extraction:

Used TfidfVectorizer to convert the cleaned text into numerical features that can be used by the machine learning model.

### Model Training:

Split the data into training and testing sets.
Trained a Logistic Regression model on the training data to classify the tweets.
Model Evaluation:

Evaluated the model’s performance using metrics like accuracy and confusion matrix to assess how well it predicts sentiments.

### Data Visualization:

Created visualizations to explore sentiment distribution and model performance (e.g., confusion matrix).
These steps outline the full pipeline

## Program:
```
! pip install kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

```
```
#api to fetch from kaggle
!kaggle datasets download -d kazanova/sentiment140
#extracting compressed file
from zipfile import ZipFile
dataset='/content/sentiment140.zip'

with ZipFile(dataset, 'r') as zip:
  zip.extractall()
  print('The dataset is extracted')
```
```
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
nltk.download('stopwords')
```
```
#printing the stopwords in english
print(stopwords.words('english'))
#loading data from csv to pandas dataframe
twitter_data=pd.read_csv('/content/training.1600000.processed.noemoticon.csv',encoding='ISO-8859-1')
```
"C:\Users\swathika\Pictures\Screenshots\Screenshot 2024-09-25 180904.png"
```
