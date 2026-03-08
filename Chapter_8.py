# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pyprind
import sys  
import re
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem.porter import PorterStemmer
import nltk
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import LatentDirichletAllocation
import pyprind

# Reading movie reviews into a single pandas df object 
base_path=r"C:\Users\User\OneDrive\Dokumente\Python Projects\Raschka ML and DL\aclImdb_v1\aclImdb"
labels={'pos':1,'neg':0}
pbar=pyprind.ProgBar(5000,stream=sys.stdout)
df=pd.DataFrame()
for s in ('train','test'):
    for l in ('pos','neg'):
        path=os.path.join(base_path,s,l)
        for file in sorted(os.listdir(path)):
            with open(os.path.join(path,file),'r',encoding='utf-8') as infile:
                txt=infile.read()
            df=df.append([[txt,labels[l]]])
            pbar.update()

# Permutating the class labels before splitting the data into test and training sets
np.random.seed(66)
df=df.sample(frac=1).reset_index(drop=True)  # resetting indices to default integers
df.to_csv('movie_data.csv',encoding='utf-8')            

# Reading the resulting csv file and proving if everything is correct
movie_data=pd.read_csv(r'.\movie_data.csv',encoding='utf-8')
movie_data=movie_data.rename(columns={'0':'Review','1':'Sentiment'})
movie_data.head(3)
movie_data.shape  # (50000, 2)

# Count vectorization 
coun=CountVectorizer()
docs = np.array(['The sun is shining', 'The weather is sweet','The sun is shining, the weather is sweet,and one and one is two'])
# Creating a document term matrix from the document
bag=coun.fit_transform(docs)
# Printing the counts per words
print(coun.vocabulary_) 
# Printing the feature array per each sentence
print(bag.toarray())
# TfidTransformer takes raw frequences as input and transforms them into tf-idf
tfidf=TfidfTransformer(use_idf=True,norm='l2',smooth_idf=True)
np.set_printoptions(precision=2)
print(tfidf.fit_transform(coun.fit_transform(docs)).toarray())

# Cleaning the text data before further processig   
movie_data.loc[2,'Review'][-50:]
# Defining function we would use for text processing 
def preprocessing(text):
    text=re.sub('<[^>]*>','',text)
    emoticons=re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
    text=(re.sub('[\W]+',' ',text.lower())+(' ').join(emoticons).replace('-',''))
    return text
# Preprocessing the text 
movie_data['Review']=movie_data['Review'].apply(preprocessing)
# Tokenizing the text 
def tokenize(text):
    return text.split()
# Porter stemming algorithm 
porter=PorterStemmer()
def tokenize_porter(text):
    return [porter.stem(words) for words in text.split()]
tokenize_porter('The runners run because they enjoy running')

# Using the stop words from nltk package 
nltk.download('stopwords')
from nltk.corpus import stopwords
stop=stopwords.words('english')
def stop_tokenizer(text):
    return [porter.stem(word) for word in text.split() if word not in stop]
stop_tokenizer('The like and love I believe are the most essential things in ones life')

# Splitting the data_frame into test and training portions
X_train,X_test,y_train,y_test=train_test_split(movie_data['Review'].values,movie_data['Sentiment'].values,test_size=0.5)
# Building a pipeline to characterize the text 
tfidf=TfidfVectorizer(strip_accents=None,lowercase=False,preprocessor=None)
lr_tfidf=Pipeline([('vect',tfidf),('clf',LogisticRegression(solver='liblinear'))])
# Initializing a parameter grid
small_param_grid = [{'vect__ngram_range': [(1, 1)],'vect__stop_words': [None],'vect__tokenizer': [tokenize, tokenize_porter],'clf__penalty': ['l2'],'clf__C': [1.0, 10.0]},{'vect__ngram_range': [(1, 1)],'vect__stop_words': [stop, None],'vect__tokenizer': [tokenize],'vect__use_idf':[False],'vect__norm':[None],'clf__penalty': ['l2'],'clf__C': [1.0, 10.0]}]
grid_lr_tfidf=GridSearchCV(lr_tfidf,param_grid=small_param_grid,scoring='f1',cv=5,n_jobs=1,verbose=2)
grid_lr_tfidf.fit(X_train,y_train)
# Looking at the best parameters of the model 
print(grid_lr_tfidf.best_params_)
print(f'{grid_lr_tfidf.best_score_:.3f}') # here the mean f1 score returned
# Returning accuracy on the test dataset 
print(f'{grid_lr_tfidf.score(X_test,y_test):.3f}') 

# Out of core learning 
def tokenizer(text):
    text=re.sub('<[^>]*>','',text)
    emoticons=re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
    text=(re.sub('[\W]+',' ',text)+(' ').join(emoticons)).replace('-','')
    tokenized=[w for w in text.split() if w not in stopwords]
    return tokenized

def stream_docs(path):
    with open(path,'r',encoding='utf-8') as csv:
        next(csv)
        for line in csv:
            text,label=line[:-3],int(line[-2])
            yield text,label

def get_minibatch(doc_stream,size):
    docs,y=[],[]
    try:
        for _ in range(size):
            text,label=next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None,None
    return docs,y
# Implementing stochasticlally trained logistic regression on the text data
vect=HashingVectorizer(decode_error='ignore',n_features=2**21,preprocessor=None,tokenizer=tokenize)
clf=SGDClassifier(loss='log',random_state=123)
docs_stream=stream_docs(path=r'.\movie_data.csv')
# Looking at the function progress
pbar=pyprind.ProgBar(45)
classes=np.array([0,1])
for _ in range(45):
    X_train,y_train=get_minibatch(docs_stream,size=1000)
    if not X_train:
        break
    X_train=vect.transform(X_train)
    clf.partial_fit(X_train,y_train,classes=classes)
    pbar.update()
# Assessing model's performance
X_test,y_test=get_minibatch(docs_stream,size=5000)
X_test=vect.transform(X_test)
print(f'{clf.score(X_test,y_test):.3f}')
clf=clf.partial_fit(X_test,y_test)

# Latent Dirichlet Analysis (LDA)
df=pd.read_csv(r'.\movie_data.csv')
df=df.drop(columns='Unnamed: 0')
df=df.rename(columns={'0':'Review','1':'Sentiment'})
# Creating a bag of word matrix from the dataframe 
count=CountVectorizer(stop_words='english',max_df=0.1,max_features=5000)
X=count.fit_transform(df['Review'])
lda=LatentDirichletAllocation(n_components=10,random_state=123,learning_method='batch')
X_topics=lda.fit_transform(X)
# Components attribute stores matrix containing important words for each topic
n_topic_words=5
feat=count.get_feature_names_out()
for top_idx,topic in enumerate(lda.components_):
    print(f'Topic {top_idx+1}:')
    print(' '.join([feat[i] for i in topic.argsort()[:-n_topic_words-1:-1]])) 

horror=X_topics[:,5].argsort()[::-1]
for iter_idx,movie_idx in enumerate(horror[:3]):
    print(f'\nHorror movies {(iter_idx+1)}:')
    print(df['Review'][movie_idx][300:])
