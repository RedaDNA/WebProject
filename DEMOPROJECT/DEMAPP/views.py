import gzip
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from django.shortcuts import render
from django.http import HttpResponse
from json import dumps
import  pickle
import os
from readability import Readability
from django.http import HttpResponse
from django.template import loader

from datatableview.views import DatatableView
import numpy as np
import en_core_web_sm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Conv1D, MaxPooling1D, Dropout
from keras.utils.np_utils import to_categorical
import joblib
import textstat
from tensorflow import keras

from .utlis import *
from .get_features import *
from keras.layers import Layer

import keras.backend as K
class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")
        super(attention, self).build(input_shape)

    def call(self,x):
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(attention,self).get_config()
""""
def getPredicitions(userinput,df):

    model = load_model('DEMOPROJECT/prediction_model.h5')
    #model = pickle.load(open('DEMOPROJECT/finalized_model.sav', 'rb'))
    #model = keras.models.load_model('DEMOPROJECT/saved_model.h5', custom_objects={'attention': attention})
    #userinput_scaled=standard_scalar
    #df=df.drop(['Text'],axis=1)
    #userinput_scaled= model.predict( np.array(df).astype("float32"))
    userinput_scaled=standard_scalar(userinput, model)
    return userinput_scaled
"""
def home(request):
    #nlp = spacy.load("en_core_web_sm")
   # benepar.download('benepar_en3')
    #result=os.getcwd()

    return render(request, 'DEMAPP/index.html', {'result': result})


def preprocess(text):
    text = text.lower()  # lowercase text
    text = text.strip()  # get rid of leading/trailing whitespace
    text = re.compile('<.*?>').sub('', text)  # Remove HTML tags/markups
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ',
                                                                  text)  # Replace punctuation with space. Careful since punctuation can sometime be useful
    text = re.sub('\s+', ' ', text)  # Remove extra space and tabs
    text = re.sub(r'\[[0-9]*\]', ' ', text)  # [0-9] matches any digit (0 to 10000...)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d', ' ', text)  # matches any digit from 0 to 100000..., \D matches non-digits
    text = re.sub(r'\s+', ' ',
                  text)  # \s matches any whitespace, \s+ matches multiple whitespace, \S matches non-whitespace

    return text
def stopword(string):
    a= [i for i in string.split() if i not in stopwords.words('english')]
    return ' '.join(a)
def finalpreprocess(string):
    return (stopword(preprocess(string)))
def getPredicitions22(userinput, df):

    model = keras.models.load_model('DEMOPROJECT/saved_model2.h5', custom_objects={'attention': attention})
    with open('tokenizer_process_200.pickle', 'rb') as handle:
        loaded_tokenizer = pickle.load(handle)

    df['clean_text'] = df['Text'].apply(lambda x: finalpreprocess(x))
    tt=df['clean_text'][0]
    #cleaned_input = clan_text(userinput)
    seq = loaded_tokenizer.texts_to_sequences([tt])
    padded = pad_sequences(seq, maxlen=400)
    # df=df.drop(['Text'],axis=1)
    # userinput_scaled= model.predict( np.array(df).astype("float32"))
    df = df.drop(['Text', 'Pronoun_percent', 'SBARQ_per_sent', 'Proper_noun_percent', 'Noun_percent','clean_text'], axis=1)
    #pred = model.predict([padded, np.array(df).astype("float32")])
    pred = model.predict(padded)

    return pred
def result(request):
    input=request.GET['userinput']
    user_text = {"Text": [input]}
    df = pd.DataFrame(user_text)
    df.head()
    df = words_and_sentences(df)
    df = syllables(df)
    df = difficult_words_pct(df)
    df = polysyllables(df)
    df = complex_words_pct(df)
    df = long_sent_pct(df)
    df = long_word_pct(df)
    df = avg_letters_per_word(df)
    df = comma_pct(df)
    df = pos_features(df)
    df = remove_aux_features(df)
    df = parse_tree_features(df)

    result=getPredicitions22(input,df)


    if len(input.split())>101:
        r = Readability(df['Text'][0])
        flesch_reading_ease=r.flesch().grade_levels
        dale_chall_readability_score = r.dale_chall().grade_levels
        gunning_fog = r.gunning_fog().grade_level
    else:
        flesch_reading_ease="not supported if the number of words is less than 100"
        dale_chall_readability_score="not supported if the number of words is less than 100"
        gunning_fog="not supported if the number of words is less than 100"

    Avg_words_per_sentence = "{:.2f}".format(df['Avg_words_per_sentence'][0])
    Avg_syllables_per_word =  "{:.2f}".format(df['Avg_syllables_per_word'][0])
    Difficult_word_percent=100*df['Difficult_word_percent'][0]
    Difficult_word_percent =  "{:.2f}".format(Difficult_word_percent)
    Long_sent_percent=100*df['Long_sent_percent'][0]
    Long_sent_percent =  "{:.2f}".format(Long_sent_percent)

    Avg_letters_per_word =  "{:.2f}".format(df['Avg_letters_per_word'][0])
    Comma_percent =100*df['Comma_percent'][0]
    Comma_percent =    "{:.2f}".format(Comma_percent)
    Conj_percent = 100*df['Conj_percent'][0]
    Conj_percent =  "{:.2f}".format(Conj_percent)
    NP_per_sent =  "{:.2f}".format(df['NP_per_sent'][0])
    VP_per_sent =  "{:.2f}".format(df['VP_per_sent'][0])
    PP_per_sent =  "{:.2f}".format(df['PP_per_sent'][0])
    Complex_word_percent =100*df['Complex_word_percent'][0]
    Complex_word_percent= "{:.2f}".format(Complex_word_percent)

    SBAR_per_sent =  "{:.2f}".format(df['SBAR_per_sent'][0])
    avg_NP_size =  "{:.2f}".format(df['avg_NP_size'][0])
    avg_VP_size = "{:.2f}".format( df['avg_VP_size'][0])
    avg_PP_size = "{:.2f}".format( df['avg_PP_size'][0])
    avg_parse_tree =  "{:.2f}".format(df['avg_parse_tree'][0])
    Long_word_percent=100*df['Long_word_percent'][0]
    Long_word_percent =  "{:.2f}".format(Long_word_percent)

    SBARQ_per_sent= "{:.2f}".format(df['SBARQ_per_sent'][0])
    Noun_percent=100*df['Noun_percent'][0]
    Noun_percent="{:.2f}".format(Noun_percent)
    Proper_noun_percent=100*df['Proper_noun_percent'][0]
    Proper_noun_percent="{:.2f}".format(Proper_noun_percent)
    Pronoun_percent=100*df['Pronoun_percent'][0]
    Pronoun_percent="{:.2f}".format(Pronoun_percent)

    return render(request, 'DEMAPP/result.html', {'result': result,'Avg_words_per_sentence':Avg_words_per_sentence,
     'Avg_syllables_per_word': Avg_syllables_per_word,
    'Difficult_word_percent': Difficult_word_percent,
    'Long_sent_percent': Long_sent_percent,
    'Avg_letters_per_word': Avg_letters_per_word,
    'Comma_percent': Comma_percent,
    'Conj_percent': Conj_percent,
    'NP_per_sent': NP_per_sent,
    'VP_per_sent': VP_per_sent,
    'Complex_word_percent':Complex_word_percent,
    'PP_per_sent': PP_per_sent,
    'SBAR_per_sent': SBAR_per_sent,
    'avg_NP_size': avg_NP_size,
    'avg_VP_size': avg_VP_size,
     'avg_PP_size': avg_PP_size,
    'avg_parse_tree': avg_parse_tree,
    'Long_word_percent': Long_word_percent,
         'input':input,
        'dale_chall_readability_score':dale_chall_readability_score,
             'flesch_reading_ease':flesch_reading_ease,
             'gunning_fog':gunning_fog,
        'Pronoun_percent':Pronoun_percent,
             'SBARQ_per_sent':SBARQ_per_sent,
                    'Proper_noun_percent':Proper_noun_percent,
                          'Noun_percent':Noun_percent
    })


"""
def standard_scalar(userinputa,model):
    with open('tokenizer_process_200.pickle', 'rb') as handle:
        loaded_tokenizer = pickle.load(handle)
    cleaned_input= clan_text(userinputa)
    seq = loaded_tokenizer.texts_to_sequences([cleaned_input])
    padded = pad_sequences(seq, maxlen=400)
    pred = model.predict(padded)

    print("----------------------------------------------------------------")
    #print(y_classes)
    return  pred
"""
def standard_scalar22(userinputa,model,df):
    with open('tokenizer_process_200.pickle', 'rb') as handle:
        loaded_tokenizer = pickle.load(handle)
    cleaned_input= clan_text(userinputa)
    seq = loaded_tokenizer.texts_to_sequences([cleaned_input])
    padded = pad_sequences(seq, maxlen=400)

    print("+5414321321519841651651*7*/")
    print(df['Pronoun_percent'][0])
    # df=df.drop(['Text'],axis=1)
    # userinput_scaled= model.predict( np.array(df).astype("float32"))
    df=df.drop(['Text','Pronoun_percent', 'SBARQ_per_sent','Proper_noun_percent','Noun_percent'], axis=1, inplace=True)
    print(df.head())
    pred = model.predict([padded , np.array(df).astype("float32")])

    print("----------------------------------------------------------------")
    #print(y_classes)
    return  pred

def urlresult(request):
    url=request.GET['url']
    try:
        url_text=get_url_text(url)
    except:
        return render(request, 'DEMAPP/error.html')
    #result = request.POST.get('userinput_scaled')


    user_text = {"Text": [url_text]}
    df = pd.DataFrame(user_text)
    df.head()
    df = words_and_sentences(df)
    df = syllables(df)
    df = difficult_words_pct(df)
    df = polysyllables(df)
    df = complex_words_pct(df)
    df = long_sent_pct(df)
    df = long_word_pct(df)
    df = avg_letters_per_word(df)
    df = comma_pct(df)
    df = pos_features(df)
    df = remove_aux_features(df)
    df = parse_tree_features(df)

    r = Readability(df['Text'][0])
    result = getPredicitions22(url_text, df)
    if len(url_text.split())>101:
        flesch_reading_ease = r.flesch().grade_levels
        dale_chall_readability_score = r.dale_chall().grade_levels
        gunning_fog = r.gunning_fog().grade_level
    else:
        flesch_reading_ease = "not supported if the number of words is less than 100"
        dale_chall_readability_score = "not supported if the number of words is less than 100"
        gunning_fog = "not supported if the number of words is less than 100"


    Avg_words_per_sentence = "{:.2f}".format(df['Avg_words_per_sentence'][0])
    Avg_syllables_per_word = "{:.2f}".format(df['Avg_syllables_per_word'][0])
    Difficult_word_percent = 100 * df['Difficult_word_percent'][0]
    Difficult_word_percent = "{:.2f}".format(Difficult_word_percent)
    Long_sent_percent = 100 * df['Long_sent_percent'][0]
    Long_sent_percent = "{:.2f}".format(Long_sent_percent)

    Avg_letters_per_word = "{:.2f}".format(df['Avg_letters_per_word'][0])
    Comma_percent = 100 * df['Comma_percent'][0]
    Comma_percent = "{:.2f}".format(Comma_percent)
    Conj_percent = 100 * df['Conj_percent'][0]
    Conj_percent = "{:.2f}".format(Conj_percent)
    NP_per_sent = "{:.2f}".format(df['NP_per_sent'][0])
    VP_per_sent = "{:.2f}".format(df['VP_per_sent'][0])
    PP_per_sent = "{:.2f}".format(df['PP_per_sent'][0])
    Complex_word_percent = 100 * df['Complex_word_percent'][0]
    Complex_word_percent = "{:.2f}".format(Complex_word_percent)

    SBAR_per_sent = "{:.2f}".format(df['SBAR_per_sent'][0])
    avg_NP_size = "{:.2f}".format(df['avg_NP_size'][0])
    avg_VP_size = "{:.2f}".format(df['avg_VP_size'][0])
    avg_PP_size = "{:.2f}".format(df['avg_PP_size'][0])
    avg_parse_tree = "{:.2f}".format(df['avg_parse_tree'][0])
    Long_word_percent = 100 * df['Long_word_percent'][0]
    Long_word_percent = "{:.2f}".format(Long_word_percent)

    SBARQ_per_sent= "{:.2f}".format(df['SBARQ_per_sent'][0])
    Noun_percent=100*df['Noun_percent'][0]
    Noun_percent="{:.2f}".format(Noun_percent)
    Proper_noun_percent=100*df['Proper_noun_percent'][0]
    Proper_noun_percent="{:.2f}".format(Proper_noun_percent)
    Pronoun_percent=100*df['Pronoun_percent'][0]
    Pronoun_percent="{:.2f}".format(Pronoun_percent)


    return render(request, 'DEMAPP/urlresult.html', {'result': result, 'Avg_words_per_sentence': Avg_words_per_sentence,
                                                  'Avg_syllables_per_word': Avg_syllables_per_word,
                                                  'Difficult_word_percent': Difficult_word_percent,
                                                  'Long_sent_percent': Long_sent_percent,
                                                  'Avg_letters_per_word': Avg_letters_per_word,
                                                  'Comma_percent': Comma_percent,
                                                  'Conj_percent': Conj_percent,
                                                  'NP_per_sent': NP_per_sent,
                                                  'VP_per_sent': VP_per_sent,
                                                  'Complex_word_percent': Complex_word_percent,
                                                  'PP_per_sent': PP_per_sent,
                                                  'SBAR_per_sent': SBAR_per_sent,
                                                  'avg_NP_size': avg_NP_size,
                                                  'avg_VP_size': avg_VP_size,
                                                  'avg_PP_size': avg_PP_size,
                                                  'avg_parse_tree': avg_parse_tree,
                                                  'Long_word_percent': Long_word_percent,
                                                  'input': input,
                                                  'dale_chall_readability_score': dale_chall_readability_score,
                                                  'flesch_reading_ease': flesch_reading_ease,
                                                  'gunning_fog': gunning_fog,
                                                  'url':url, 'Pronoun_percent':Pronoun_percent,
             'SBARQ_per_sent':SBARQ_per_sent,
                    'Proper_noun_percent':Proper_noun_percent,
                          'Noun_percent':Noun_percent
                                                  })

