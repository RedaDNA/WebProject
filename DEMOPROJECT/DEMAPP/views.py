import gzip
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from django.shortcuts import render
from django.http import HttpResponse
from json import dumps
import  pickle
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Conv1D, MaxPooling1D, Dropout
from keras.utils.np_utils import to_categorical
import joblib
from .utlis import *
def getPredicitions(userinput):
    #model = pickle.load(open('CNN-LSTMgradesdatasetsplited62%pre.sav', 'rb'))
    #model=joblib.load('CNN-LSTMgradesdatasetsplited62%pre.sav')
    #model=joblib.load('CBOW_grades_dataset_76_pre.pkl')
    #localhost_save_option = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")
    print(dir)
    #model = tf.keras.models.load_model('CBOW_grades_dataset_76_pre')
    #with gzip.open('Completed_model.joblib', 'rb') as f:
     #   model = pickle.load(f)
    # with open("C:/Users/reda/PycharmProjects/git/WebProject/DEMOPROJECT/DEMOPROJECT/CBOWgradesdataset76pre.sav", 'rb') as handle:
    #   model = pickle.load(handle)
    model = load_model('DEMOPROJECT/prediction_model.h5')
    userinput_scaled=standard_scalar(userinput,model)
    print(userinput_scaled)
    return userinput_scaled
#def hi(request):
 #   return HttpResponse('<h1> This is mY home page</h1>')
def home(request):
    result=os.getcwd()
    return render(request, 'DEMAPP/index.html', {'result': result})

def result(request):
    input=request.GET['userinput']
    print(input)
    #result = request.POST.get('userinput_scaled')
    result=getPredicitions(input)
    print("******************///////////////////////******************")
    print(result)
    #result = getPredicitions(userinput)
    #return render(request, 'result.html', {'result': result})
    #result = os.getcwd()


    return render(request, 'DEMAPP/result.html', {'result': result})


def standard_scalar(userinputa,model):
    with open('tokenizer_process_200.pickle', 'rb') as handle:
        loaded_tokenizer = pickle.load(handle)
    cleaned_input= clan_text(userinputa)
    print(cleaned_input)
    seq = loaded_tokenizer.texts_to_sequences([cleaned_input])
    padded = pad_sequences(seq, maxlen=400)
    pred = model.predict(padded)
    return  pred


def urlresult(request):
    input=request.GET['url']
    url_text=get_url_text(input)
    #result = request.POST.get('userinput_scaled')
    result=getPredicitions(url_text)
    print("mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm")
    print(url_text)
    #result = getPredicitions(userinput)
    #return render(request, 'result.html', {'result': result})
    #result = os.getcwd()
    return render(request, 'DEMAPP/urlresult.html', {'result': result})
