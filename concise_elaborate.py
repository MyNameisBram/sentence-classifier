# upper and lower bound are derrived from exploritory work 
# /Users/bramtunggala/CrystalKnows/project-06-style-classifier/01-scripts/01-concise-elaborate.ipynb

# upper and lower bound are derrived from exploritory work 
# /Users/bramtunggala/CrystalKnows/project-06-style-classifier/01-scripts/01-concise-elaborate.ipynb

# short/long 
def cta(row):
    lower = 9
    upper = 14
    
    if row['word_count'] < lower:
        return "concise"
    elif row['word_count'] > upper:
        return "elaborate"
    else:
        return "average"

# short/long 
def credibility(row):
    lower = 10
    upper = 14
    
    if row['word_count'] < lower:
        return "concise"
    elif row['word_count'] > upper:
        return "elaborate"
    else:
        return "average"


# short/long 
def intention_statement(row):
    lower = 11
    upper = 16
    
    if row['word_count'] < lower:
        return "concise"
    elif row['word_count'] > upper:
        return "elaborate"
    else:
        return "average"


# short/long 
def value_prop(row):
    lower = 11
    upper = 16.25
    
    if row['word_count'] < lower:
        return "concise"
    elif row['word_count'] > upper:
        return "elaborate"
    else:
        return "average"



# short/long 
def warm_up(row):
    lower = 6.25
    upper = 11
    
    if row['word_count'] < lower:
        return "concise"
    elif row['word_count'] > upper:
        return "elaborate"
    else:
        return "average"
      
      
      
      
# feature extraction 
import pandas as pd
import numpy as np

def predict_concise_elaborate(text, sentence_cat):
    df = pd.DataFrame()
    df['sentence'] = text
    # word count 
    df['word_count'] = df.sentence.apply(lambda x: len(str(x).split(" "))

    if sentence_cat == 'call_to_action':
        df['label'] = df.apply(cta, axis=1)
    elif sentence_cat == 'credibility_statement':
        df['label'] = df.apply(credibility, axis=1)
    elif sentence_cat == 'intention_statement':
        df['label'] = df.apply(intention_statement, axis=1)
    elif sentence_cat == 'value_prop':
        df['label'] = df.apply(value_prop, axis=1)
    elif sentence_cat == 'warm_up':
        df['label'] = df.apply(warm_up, axis=1)
                                
    
    return df.label.values[0]






