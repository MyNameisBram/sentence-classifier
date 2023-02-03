import streamlit as st
import pickle
import numpy as np 
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from concise_elaborate import *

#from sklearn.model_selection import train_test_split
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
#from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer


# path 
path = "./models"

# folder names where models live
folders = [
    "/call_to_action", 
    "/credibility_statement",
    "/greeting",
    "/intention_statement",
    "/intro",
    "/problem_statement",
    "/sign_off",
    "/value_prop",
    "/warm_up"

]

# model name
model_names = [
    "/linearSVC_clf.pkl",
    "/logReg_clf.pkl",
    "/multinomialNB_clf.pkl",
    "/randomForest_clf.pkl"
]


def predict(pred_text):
    
    import statistics
    import operator

    path = "./models"
    
    folders = ["/call_to_action", "/credibility_statement","/greeting","/intention_statement",
                "/intro","/problem_statement","/sign_off","/value_prop","/warm_up"]
    
    label = [s.replace("/", "") for s in folders]

    model_names = ["/linearSVC_clf.pkl","/logReg_clf.pkl","/mutlinomialNB_clf.pkl","/randomForest_clf.pkl"]

    pred_confidence = []

    for folder in folders:
        # load tfidf
        tf1 = pickle.load(open(path+folder+ "/tfidf1.pkl", 'rb'))# loading dictionary
        # Create new tfidfVectorizer with old vocabulary
        tf1_new = TfidfVectorizer(sublinear_tf=True, norm='l2', 
                                encoding='latin-1', ngram_range=(1, 2), 
                                stop_words=None, vocabulary = tf1)# to use trained vectorizer, vocabulary= tf1 or loaded dict.
        # fit text you want to predict 
        X_tf1 = tf1_new.fit_transform([pred_text])

        # list of model results per category 
        res = []

        for model in model_names:
            clf = pickle.load(open(path + folder + model,'rb')) # loading model 
            res.append(clf.predict_proba(X_tf1)[0][1]) # predict

        pred = statistics.mean(res) # get average confindence from 4 models 

        pred_confidence.append(pred)
    

 
    # create dictionary of preds 
    preds = {label[i]: pred_confidence[i] for i in range(len(label))}
    # sort values
    preds = {k: v for k, v in sorted(preds.items(), key=lambda item: item[1], reverse=True)}

    first = next(iter(preds.items()))
    prediction = first[0]
    confidence = (round((first[1] * 100), 2))



    return prediction, confidence



def predict_cta(pred_text):
    
    import statistics
    import operator

    path = "./models"+"/call_to_action_models"
    
    folders = ['/meeting_cta', '/feedback_cta', '/response_cta', '/action_cta',
                '/need_validation_cta', '/information_cta', '/webinar_cta', 
                '/intro_cta', '/rejection_cta', '/contact_cta']
    
    label = [s.replace("/", "") for s in folders]

    model_names = ["/linearSVC_clf.pkl","/logReg_clf.pkl","/mutlinomialNB_clf.pkl","/randomForest_clf.pkl"]

    pred_confidence = []

    for folder in folders:
        # load tfidf
        tf1 = pickle.load(open(path+folder+ "/tfidf1.pkl", 'rb'))# loading dictionary
        # Create new tfidfVectorizer with old vocabulary
        tf1_new = TfidfVectorizer(sublinear_tf=True, norm='l2', 
                                encoding='latin-1', ngram_range=(1, 2), 
                                stop_words=None, vocabulary = tf1)# to use trained vectorizer, vocabulary= tf1 or loaded dict.
        # fit text you want to predict 
        X_tf1 = tf1_new.fit_transform([pred_text])

        # list of model results per category 
        res = []

        for model in model_names:
            clf = pickle.load(open(path + folder + model,'rb')) # loading model 
            res.append(clf.predict_proba(X_tf1)[0][1]) # predict

        pred = statistics.mean(res) # get average confindence from 4 models 

        pred_confidence.append(pred)
    

 
    # create dictionary of preds 
    preds = {label[i]: pred_confidence[i] for i in range(len(label))}
    # sort values
    preds = {k: v for k, v in sorted(preds.items(), key=lambda item: item[1], reverse=True)}

    first = next(iter(preds.items()))
    prediction = first[0]
    confidence = (round((first[1] * 100), 2))



    return prediction, confidence


# predict warmup
def predict_warmup(pred_text):
    
    import statistics
    import operator

    path = "./models"+"/warm_ups"
    
    folders = ['/gratitude_warmup', '/intention_warmup', 
                '/personal_warmup','/relevance_warmup',
                '/timing_warmup'
                ]
    
    label = [s.replace("/", "") for s in folders]

    model_names = ["/linearSVC_clf.pkl","/logReg_clf.pkl","/mutlinomialNB_clf.pkl","/randomForest_clf.pkl"]

    pred_confidence = []

    for folder in folders:
        # load tfidf
        tf1 = pickle.load(open(path+folder+ "/tfidf1.pkl", 'rb'))# loading dictionary
        # Create new tfidfVectorizer with old vocabulary
        tf1_new = TfidfVectorizer(sublinear_tf=True, norm='l2', 
                                encoding='latin-1', ngram_range=(1, 2), 
                                stop_words=None, vocabulary = tf1)# to use trained vectorizer, vocabulary= tf1 or loaded dict.
        # fit text you want to predict 
        X_tf1 = tf1_new.fit_transform([pred_text])

        # list of model results per category 
        res = []

        for model in model_names:
            clf = pickle.load(open(path + folder + model,'rb')) # loading model 
            res.append(clf.predict_proba(X_tf1)[0][1]) # predict

        pred = statistics.mean(res) # get average confindence from 4 models 

        pred_confidence.append(pred)
    

 
    # create dictionary of preds 
    preds = {label[i]: pred_confidence[i] for i in range(len(label))}
    # sort values
    preds = {k: v for k, v in sorted(preds.items(), key=lambda item: item[1], reverse=True)}

    first = next(iter(preds.items()))
    prediction = first[0]
    confidence = (round((first[1] * 100), 2))



    return prediction, confidence

# value_prop
# predict warmup
def predict_valueprop(pred_text):
    
    import statistics
    import operator

    path = "./models"+"/value_props"
    
    folders = ['/benefit_value_prop', '/summary_value_prop', 
                '/feature_value_prop','/risk_reduction_value_prop',
                '/roi_value_prop', '/skill_value_prop',
                '/efficiency_value_prop','/social_proof_value_prop',
                '/pricing_value_prop',
                ] # excluding time_value_prop due insufficient data
    
    label = [s.replace("/", "") for s in folders]

    model_names = ["/linearSVC_clf.pkl","/logReg_clf.pkl","/mutlinomialNB_clf.pkl","/randomForest_clf.pkl"]

    pred_confidence = []

    for folder in folders:
        # load tfidf
        tf1 = pickle.load(open(path+folder+ "/tfidf1.pkl", 'rb'))# loading dictionary
        # Create new tfidfVectorizer with old vocabulary
        tf1_new = TfidfVectorizer(sublinear_tf=True, norm='l2', 
                                encoding='latin-1', ngram_range=(1, 2), 
                                stop_words=None, vocabulary = tf1)# to use trained vectorizer, vocabulary= tf1 or loaded dict.
        # fit text you want to predict 
        X_tf1 = tf1_new.fit_transform([pred_text])

        # list of model results per category 
        res = []

        for model in model_names:
            clf = pickle.load(open(path + folder + model,'rb')) # loading model 
            res.append(clf.predict_proba(X_tf1)[0][1]) # predict

        pred = statistics.mean(res) # get average confindence from 4 models 

        pred_confidence.append(pred)
    

 
    # create dictionary of preds 
    preds = {label[i]: pred_confidence[i] for i in range(len(label))}
    # sort values
    preds = {k: v for k, v in sorted(preds.items(), key=lambda item: item[1], reverse=True)}

    first = next(iter(preds.items()))
    prediction = first[0]
    confidence = (round((first[1] * 100), 2))



    return prediction, confidence


# credibility statements
def predict_credibility(pred_text):
    
    import statistics
    import operator

    path = "./models"+"/credibility_statements"
    
    folders = ['/appeal_to_experience', '/appeal_to_success', 
                '/appeal_to_social_proof'
                ] # excluding appeal_to_authority due insufficient data
    
    label = [s.replace("/", "") for s in folders]

    model_names = ["/linearSVC_clf.pkl","/logReg_clf.pkl","/mutlinomialNB_clf.pkl","/randomForest_clf.pkl"]

    pred_confidence = []

    for folder in folders:
        # load tfidf
        tf1 = pickle.load(open(path+folder+ "/tfidf1.pkl", 'rb'))# loading dictionary
        # Create new tfidfVectorizer with old vocabulary
        tf1_new = TfidfVectorizer(sublinear_tf=True, norm='l2', 
                                encoding='latin-1', ngram_range=(1, 2), 
                                stop_words=None, vocabulary = tf1)# to use trained vectorizer, vocabulary= tf1 or loaded dict.
        # fit text you want to predict 
        X_tf1 = tf1_new.fit_transform([pred_text])

        # list of model results per category 
        res = []

        for model in model_names:
            clf = pickle.load(open(path + folder + model,'rb')) # loading model 
            res.append(clf.predict_proba(X_tf1)[0][1]) # predict

        pred = statistics.mean(res) # get average confindence from 4 models 

        pred_confidence.append(pred)
    

 
    # create dictionary of preds 
    preds = {label[i]: pred_confidence[i] for i in range(len(label))}
    # sort values
    preds = {k: v for k, v in sorted(preds.items(), key=lambda item: item[1], reverse=True)}

    first = next(iter(preds.items()))
    prediction = first[0]
    confidence = (round((first[1] * 100), 2))



    return prediction, confidence


  
  
### start STYLE classifier here 

from pickle import load

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# path 
path = "./models/casual_formal"

# create new features from text 
def preprocess(text):
    df = pd.DataFrame(text,columns=['sentence'],index=[0])
    
    # word count 
    df['word_count'] = df.sentence.apply(lambda x: len(str(x).split(" ")))
    df['char_count'] = df.sentence.apply(lambda x: sum(len(word) for word in str(x).split(" ")))

    # average word length
    df['avg_word_length'] = df['char_count'] / df['word_count']

    # create a list of formal pronouns
    formal_prons = [
        "we","they", "their", "themselves", "us", "our", "ourselves", "ours", "it", "its", "itself"
    ]
    informal_prons = [
        "i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself", "yourselves"
    ]
    # count if matching formal pronouns in list 
    df['formal_pron'] = df.sentence.apply(lambda x: sum(x.count(word) for word in formal_prons))
    # count if matching informal pronouns in list 
    df['informal_pron'] = df.sentence.apply(lambda x: sum(x.count(word) for word in informal_prons))

    contraction_list = [
        "let's" , "ain't", "ya'll", "I'm", "here's", "you're",
        "that's", "he's", "she's", "it's", "we're", "they're",
        "I'll", "we'll", "you'll", "it'll", "he'll", "she'll",
        "I've", "should've", "you've", "could've", "they've",
        "I'd", "we've", "they'd", "you'd", "we'd", "he'd", "she'd",
        "didn't" "don't", "doesn't", "can't", "isn't", "aren't",
        "shouldn't", "couldn't", "wouldn't", "hasn't", "wasn't", 
        "won't", "weren't", "haven't", "hadn't"
    ]
    # count if matching word from list 
    df['contraction_count'] = df.sentence.apply(lambda x: sum(x.count(word) for word in contraction_list))

    num_cols = ['word_count',
            'char_count',
            'avg_word_length',
            'formal_pron',
            'informal_pron',
            'contraction_count']

    scalers = ["/scaler_word_count.pkl",
            "/scaler_char_count.pkl",
            "/scaler_avg_word_length.pkl",
            "/scaler_formal_pron.pkl",
            "/scaler_informal_pron.pkl",
            "/scaler_contraction_count.pkl"]

    # iterate through scaler and column names   
    # # transform numerical columns        

    for scaler, col in zip(scalers, num_cols):
        rob_scl = pickle.load(open(path+ scaler, 'rb'))
        df[col] = rob_scl.fit_transform(df[col].values.reshape(-1,1))


    return df

# load Pipelines (model + vectorizer included)
import joblib

rf_pipe = loaded_pipe = joblib.load(path+"/rf_pipe.joblib")
lr_pipe = loaded_pipe = joblib.load(path+"/lr_pipe.joblib")

# predict on test data
def predict_RF(pred_data):
    # load pipe and predict
    pred = rf_pipe.predict(pred_data)
    y_proba = rf_pipe.predict_proba(pred_data)
    y_proba = y_proba.tolist() # array to list
    #casual = 1, formal = 0
    casual = y_proba[0][1]
    formal = y_proba[0][0]

    return pred, casual, formal


def predict_LR(pred_data):
    # load pipe and predict
    pred = lr_pipe.predict(pred_data)
    y_proba = lr_pipe.predict_proba(pred_data)
    y_proba = y_proba.tolist() # array to list
    #casual = 1, formal = 0
    casual = y_proba[0][1]
    formal = y_proba[0][0]

    return pred, casual, formal


# create ensemble result
def predict_style(new_text):
    
    text = preprocess(new_text)
    # predict_LR
    lr_pred, lr_casual, lr_formal = predict_LR(text)
    # predict_RF 
    rf_pred, rf_casual, rf_formal = predict_RF(text)

    # taking mean of both predictions
    #pred = round((lr_pred + rf_pred)/2 , 6) # round to 6 decimal places
    casual = round((lr_casual + rf_casual)/2, 6) 
    formal = round((lr_formal + rf_formal)/2, 6) 

    # threshold of .5 
    if casual > formal:
        pred = "casual"
        if casual < .507913:
            conf = "kinda confident it's {}".format(pred)
        if casual >= .507913 and casual < .669477:
            conf = "pretty confident it's {}".format(pred) 
        if casual >= .669477:
            conf = "very confident it's {}".format(pred) 
    if formal > casual:
        pred = "formal"
        if formal < .640507:
            conf = "kinda confident it's {}".format(pred)
        if formal >= .640507 and formal < .77:
            conf = "pretty confident it's {}".format(pred) 
        if formal >= .77:
            conf = "very confident it's {}".format(pred) 
    if casual == formal:
        pred = "not available"


    return pred, conf, casual, formal




### end 


# predict function
def show_predict_page():
    st.title("Sentence Category and Style Classifier ")



    st.write("""### Enter a complete sentence to identify the sentence 1) category 2) length 3) tone.""")
    query = st.text_area("Enter text here 👇", "", max_chars=300)
    
    if query != "":
        # run prediction function 
        pred, conf = predict(query)


        st.write("Sentence Category: {}  ----->  Confidence: {}".format(pred, conf) )
   
        if pred == "call_to_action":
            text_len = predict_concise_elaborate(query, pred)
            st.write("Text Length is: {}".format(text_len))

            pred, pred_conf = predict_cta(query)
            st.write("CTA Type: {}. ----->  Confidence: {}".format(pred, pred_conf))
            


  
        if pred == "warm_up":
            text_len = predict_concise_elaborate(query, pred)
            st.write("Text Length is: {}".format(text_len))

            pred, pred_conf = predict_warmup(query)
            st.write("Warm-up Type: {}  ----->  Confidence: {}".format(pred, pred_conf))
            

        if pred == "value_prop":
            text_len = predict_concise_elaborate(query, pred)
            st.write("Text Length is: {}".format(text_len))

            pred, pred_conf = predict_valueprop(query)
            st.write("Value-prop Type: {}  ----->  Confidence: {}".format(pred, pred_conf))
            

        if pred == "credibility_statement":
            text_len = predict_concise_elaborate(query, pred)
            st.write("Text Length is: {}".format(text_len))

            pred, pred_conf = predict_credibility(query)
            st.write("Credibility Type: {}. ----->  Confidence: {}".format(pred, pred_conf))
        
        # run predict_style function    
        pred, conf, casual, formal = predict_style(query)
        
        x = 123456789
        
        st.write("Tone: {}".format(conf))
        #st.write("--- {} ---".format(conf))
        st.write("Confidence  ----->  Casual: {}  |  Formal: {}".format(casual, formal))
        
            
