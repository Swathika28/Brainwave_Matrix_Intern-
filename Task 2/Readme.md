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
```
twitter_data.shape
```
![Screenshot 2024-09-25 180904](https://github.com/user-attachments/assets/fd59acbd-037f-4a9b-934e-bbb35d05bc3e)

```
twitter_data.head()
```
![Screenshot 2024-09-25 181417](https://github.com/user-attachments/assets/6881461c-84ac-491e-9c30-e2a8ef5e7e76)
```
twitter_data.isnull().sum()
```

![Screenshot 2024-09-25 181513](https://github.com/user-attachments/assets/e835bb6d-0ade-4fa2-956c-804fa155c97c)
```
#checking target column
twitter_data['target'].value_counts()
```
![Screenshot 2024-09-25 181558](https://github.com/user-attachments/assets/5104ae5a-8f35-4f34-91e4-75cc9fe8c9d0)
```
#convert 4 to 1
twitter_data.replace({'target':{4:1}},inplace=True)
twitter_data['target'].value_counts()
```
![Screenshot 2024-09-25 181645](https://github.com/user-attachments/assets/f04dcdc1-437b-4345-afb2-ad3de22344be)
```
port_stem=PorterStemmer()
def stemming(content):
  stemmed_content=re.sub('[^a-zA-Z]',' ',content)
  stemmed_content=stemmed_content.lower()
  stemmed_content=stemmed_content.split()
  stemmed_content=[port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]

  return stemmed_content
twitter_data['stemmed_content']=twitter_data['text'].apply(stemming)
```
```
print(twitter_data['stemmed_content'])
```
![Screenshot 2024-09-25 181844](https://github.com/user-attachments/assets/8b606763-d3c3-426a-bf47-2eaf711a6686)
```
print(twitter_data['target'])
#seperating the data and label
X=twitter_data['stemmed_content'].values
Y=twitter_data['target'].values
```
```
print(X)
```
![Screenshot 2024-09-25 181953](https://github.com/user-attachments/assets/dd8649a9-a014-4a05-9861-cdd00ba100ea)
```
print(Y)
```
![Screenshot 2024-09-25 182028](https://github.com/user-attachments/assets/aa083b8e-145c-4ceb-bc2f-d4e9823e33d6)
```
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
print(X.shape,X_train.shape,X_test.shape)
```
![Screenshot 2024-09-25 182113](https://github.com/user-attachments/assets/50f9e254-50e6-4162-84ef-2323b6199b6c)
```
#converting textual data to numerical data
# If X_train is a list of lists, you might need to join the inner lists into strings
X_train = [' '.join(doc) if isinstance(doc, list) else doc for doc in X_train]
X_test = [' '.join(doc) if isinstance(doc, list) else doc for doc in X_test]

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
print(X_train)
```
![Screenshot 2024-09-25 182234](https://github.com/user-attachments/assets/b1a47bcc-7d29-463c-a1d3-3bf5efef30a3)
```
print(X_test)
```
![Screenshot 2024-09-25 182241](https://github.com/user-attachments/assets/ce375cde-6704-4236-b31b-d058effb426f)
```
model=LogisticRegression(max_iter=1000)
model.fit(X_train,Y_train)
```
![Screenshot 2024-09-25 182412](https://github.com/user-attachments/assets/2ed87b96-98a5-453a-a3c7-92f902e648c7)
```
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(Y_train,X_train_prediction)
print('Accuracy score on the training data:',training_data_accuracy)
```
![Screenshot 2024-09-25 182456](https://github.com/user-attachments/assets/689c0ae7-069c-48db-80fe-4bef99a0389a)
```
X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(Y_test,X_test_prediction)
print('Accuracy score on the test data:',test_data_accuracy)
```
![Screenshot 2024-09-25 182555](https://github.com/user-attachments/assets/b4d5aab7-dd9b-49be-8424-219303eb3278)
```
#saving trained model
import pickle
filename='trained_model.sav'
pickle.dump(model,open(filename,'wb'))
#using saved model for future prediction
loaded_model=pickle.load(open('/content/trained_model.sav','rb'))
```
```
X_new=X_test[200]
print(Y_test[200])
prediction=model.predict(X_new)
print()
if(prediction[0]==0):
  print('Negative Tweet')
else:
  print('positive Tweet')
```
![Screenshot 2024-09-25 182712](https://github.com/user-attachments/assets/5f588dac-8a2c-41da-8ca8-01d54de9697a)
```
X_new=X_test[3]
print(Y_test[3])
prediction=model.predict(X_new)
print()
if(prediction[0]==0):
  print('Negative Tweet')
else:
  print('positive Tweet')
```
![Screenshot 2024-09-25 182743](https://github.com/user-attachments/assets/43ce8cdf-093f-4708-8836-4b80b9e4ee15)
```
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Example: Assuming `y_test` are the true labels and `y_pred` are the predictions
Y_pred = model.predict(X_test)
cm = confusion_matrix(Y_test, Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

# Plot confusion matrix
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
```
![Screenshot 2024-09-25 182824](https://github.com/user-attachments/assets/d1063fcf-a09e-4cc0-a3e8-bb84fe22d314)
```
feature_names = vectorizer.get_feature_names_out()
coefs = model.coef_.flatten()  # Adjust if using a different model

# Create a DataFrame for feature importances
importance = pd.DataFrame({'feature': feature_names, 'importance': np.abs(coefs)})
top_features = importance.sort_values(by='importance', ascending=False).head(20)

# Plot top features
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=top_features, palette='magma')
plt.title('Top 20 Important TF-IDF Features')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
```
![Screenshot 2024-09-25 183144](https://github.com/user-attachments/assets/244f331f-97be-4c8d-bd4e-c744419d1efb)
```
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Example: Assuming 'text' is the column with text data
# Combine all text for visualization
all_words = ' '.join(twitter_data['text'])

# Generate a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Common Words in Text Data')
plt.show()
```
![Screenshot 2024-09-25 183300](https://github.com/user-attachments/assets/44705a1f-04e4-46b2-b80d-f8877a744975)
```
sns.countplot(data=twitter_data, x='target', palette='viridis')
plt.title('Distribution of Sentiments')
plt.xlabel('target')
plt.ylabel('Count')
plt.show()
```
![Screenshot 2024-09-25 183341](https://github.com/user-attachments/assets/0930edd1-8316-4630-bdd4-8a95ba10d32b)

## Applications:
Social Media Monitoring:

Analyze public sentiment on platforms like Twitter, Facebook, and Instagram to understand opinions about brands, products, or events in real time.
Customer Feedback Analysis:

Automatically categorize customer reviews, feedback, and surveys into positive or negative sentiments to improve customer service and product development.
Market Research:

Gain insights into market trends and consumer attitudes by analyzing large volumes of text data, such as product reviews and comments.
Brand Reputation Management:

Track and respond to changes in brand perception by analyzing mentions of a brand online, helping businesses to manage their public image.
News and Media Analysis:

Analyze sentiment in news articles, blogs, and forums to understand public response to news events, policy changes, or product launches.
Product Analysis and Improvement:

Identify areas for product improvement by analyzing negative feedback and complaints, guiding businesses to make data-driven enhancements.
Political Sentiment Analysis:

Analyze public opinion on political candidates, policies, or events to gauge public mood and predict election outcomes.
Stock Market Prediction:

Use sentiment analysis on financial news and tweets to gauge market sentiment, which can help in making trading decisions.

## Result:
The sentiment analysis code successfully classifies tweets as positive or negative by cleaning text data, extracting features using TF-IDF, and training a logistic regression model. The model's performance is evaluated using accuracy and confusion matrix, demonstrating effective sentiment classification.

