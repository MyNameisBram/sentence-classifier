import streamlit as st
import pickle
import numpy as np 
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

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


# path 
path = "/Users/bramtunggala/CrystalKnows/project-04-sentence-classifier/03-models"

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

    path = "/Users/bramtunggala/CrystalKnows/project-04-sentence-classifier/03-models"
    
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

    path = "/Users/bramtunggala/CrystalKnows/project-04-sentence-classifier/03-models/call_to_action_models"
    
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

    path = "/Users/bramtunggala/CrystalKnows/project-04-sentence-classifier/03-models/warm_ups"
    
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

    path = "/Users/bramtunggala/CrystalKnows/project-04-sentence-classifier/03-models/value_props"
    
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

    path = "/Users/bramtunggala/CrystalKnows/project-04-sentence-classifier/03-models/credibility_statements"
    
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




# predict function
def show_predict_page():
    st.title("sentence classifier")



    st.write("""### We need some information to predict the sentence""")
    query = st.text_area("Enter text here 👇", "", max_chars=300)
    
    if query != "":
        # run prediction function 
        pred, conf = predict(query)

        st.write("Prediction: {}".format(pred))
        st.write("Confidence: {}".format(conf))


        if pred == "call_to_action":
            pred, pred_conf = predict_cta(query)

            st.write("CTA Type Prediction: {}".format(pred))
            st.write("Confidence: {}".format(pred_conf))

        if pred == "warm_up":
            pred, pred_conf = predict_warmup(query)

            st.write("Warm-up Type Prediction: {}".format(pred))
            st.write("Confidence: {}".format(pred_conf))

        if pred == "value_prop":
            pred, pred_conf = predict_valueprop(query)

            st.write("Value-prop Type Prediction: {}".format(pred))
            st.write("Confidence: {}".format(pred_conf))

        if pred == "credibility_statement":
            pred, pred_conf = predict_credibility(query)

            st.write("Credibility Type Prediction: {}".format(pred))
            st.write("Confidence: {}".format(pred_conf))
