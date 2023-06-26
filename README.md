# Fake News Detection: Leveraging NLP in ML and DL Models:

## Description:
The dataset contains news articles labeled as reliable or unreliable. It has 20800 records and 5 attributes (columns). 
The details of the attributes are as follows:


* id: unique id for a news article
  
* title: the title of a news article
  
* author: author of the news article
  
* text: the text of the article; could be incomplete
  
* label: a label that marks the article as potentially unreliable (1- Unreliable & 0- Reliable)


###### Download the dataset from: https://www.kaggle.com/competitions/fake-news/data/?select=train.csv

The objective was to develop a machine learning program to predict fake news articles. The project encompasses various stages, including text analysis and stylometric analysis of news articles, followed by the development of machine learning models and neural network-based models.

## Project Flow:

### 1. Text Analysis:
Initially, I performed data exploration, data cleaning, and text preprocessing tasks. Subsequently, text analysis 
and stylometric analysis techniques were employed to understand the features of the articles. This involved exploring 
linguistic patterns, writing styles, and other indicators of potential fake news.

### 2. Developing Binary Text Classification Models by ML:
I used TF-IDF and CountVectorizer techniques for feature extraction. Next, a range of classification algorithms, such as logistic regression, 
MultinomialNB, decision trees, and random forests were implemented to build machine learning models. These models were trained on 
labeled data to classify articles as genuine or fake based on the extracted features. To determine the optimal model, performance metrics 
were used such as accuracy, precision, recall, and F1 score. These metrics are especially used in classification problems.

Out of these algorithms, the logistic regression model achieved the best results. It was further fine-tuned to get the best estimator. 

### 3. Developing Binary Text Classification Model by LSTM Implementation:
Furthermore, neural network models were created to leverage the power of deep learning.  I used Keras API's Tokenizer class to map sequences 
of words into integer representation. Initially, I utilized a simple neural network architecture and later incorporated LSTM (Long Short-Term Memory) 
recurrent neural networks. I compared both models by visualizing loss function curves during the training and validation process. 
I found models tend to be overfitting in both cases.

## Conclusion:
Finally, the best model was found to be fine-tuned logistic regression model, which achieved a precision of 96.82% and recall of 98.38%. 
For improving LSTM based model, an experiment can be done with pre-trained word embeddings, like GloVe. GloVe is an unsupervised ML algorithm 
and it generates vector representations for words.
