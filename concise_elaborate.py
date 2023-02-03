# upper and lower bound are derrived from exploritory work 
# /Users/bramtunggala/CrystalKnows/project-06-style-classifier/01-scripts/01-concise-elaborate.ipynb

# upper and lower bound are derrived from exploritory work 
# /Users/bramtunggala/CrystalKnows/project-06-style-classifier/01-scripts/01-concise-elaborate.ipynb


# upper and lower bound are derrived from exploritory work 
# /Users/bramtunggala/CrystalKnows/project-06-style-classifier/01-scripts/01-concise-elaborate.ipynb

# short/long 
def cta(x):
    lower = 9
    upper = 14
    
    if x < lower:
        return "concise"
    elif x > upper:
        return "elaborate"
    else:
        return "average"

# short/long 
def credibility(x):
    lower = 10
    upper = 14
    
    if x < lower:
        return "concise"
    elif x > upper:
        return "elaborate"
    else:
        return "average"


# short/long 
def intention_statement(x):
    lower = 11
    upper = 16
    
    if x < lower:
        return "concise"
    elif x > upper:
        return "elaborate"
    else:
        return "average"


# short/long 
def value_prop(x):
    lower = 11
    upper = 16.25
    
    if x < lower:
        return "concise"
    elif x > upper:
        return "elaborate"
    else:
        return "average"



# short/long 
def warm_up(x):
    lower = 6.25
    upper = 11
    
    if x < lower:
        return "concise"
    elif x > upper:
        return "elaborate"
    else:
        return "average"
      
      
      
      
# feature extraction 
import pandas as pd
import numpy as np

def predict_concise_elaborate(text, sentence_cat):
    text_len = len(str(text).split(" "))

    if sentence_cat == 'call_to_action':
        x = cta(text_len)
    elif sentence_cat == 'credibility_statement':
        x = credibility(text_len)
    elif sentence_cat == 'intention_statement':
        x = intention_statement(text_len)
    elif sentence_cat == 'value_prop':
        x = value_prop(text_len)
    elif sentence_cat == 'warm_up':
        x = warm_up(text_len)
                                
    
    return x










