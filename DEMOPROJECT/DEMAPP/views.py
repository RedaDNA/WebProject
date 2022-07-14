import gzip

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from django.shortcuts import render
from django.http import HttpResponse
import  pickle
import os
import joblib
# Create your views here.
import tensorflow as tf
def hi(request):
    return render(request ,'DEMAPP/hi.html')
dir = os.getcwd()

def getPredicitions(userinput):
    #model = pickle.load(open('CNN-LSTMgradesdatasetsplited62%pre.sav', 'rb'))
    #model=joblib.load('CNN-LSTMgradesdatasetsplited62%pre.sav')
    #model=joblib.load('CBOW_grades_dataset_76_pre.pkl')
    #localhost_save_option = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")
    print(dir)
    #model = tf.keras.models.load_model('CBOW_grades_dataset_76_pre')
    #with gzip.open('Completed_model.joblib', 'rb') as f:
     #   model = pickle.load(f)
    with open('CBOWgradesdataset76pre.sav', 'rb') as handle:
        model = pickle.load(handle)

    userinput_scaled=standard_scalar(userinput,model)
    prediction = model.predict(userinput_scaled)
    print(prediction)
    return prediction


def getPredicitionsDemo(userinput):
    #model = pickle.load(open('CNN-LSTMgradesdatasetsplited62%pre.sav', 'rb'))
    #model=joblib.load('CNN-LSTMgradesdatasetsplited62%pre.sav')
    #model=joblib.load('CBOW_grades_dataset_76_pre.pkl')
    #localhost_save_option = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")
    #model = tf.keras.models.load_model('CBOW_grades_dataset_76_pre')
    #with gzip.open('Completed_model.joblib', 'rb') as f:
     #   model = pickle.load(f)

    #userinput_scaled=standard_scalar(userinput,model)
    #prediction = model.predict(userinput_scaled)
    #print(prediction)
    prediction="this is result "
    return prediction
#def hi(request):
 #   return HttpResponse('<h1> This is mY home page</h1>')
def home(request):
    result=os.getcwd()
    return render(request, 'DEMAPP/index.html', {'result': result})

def result(request):
    input=request.GET['userinput']
    #result = request.POST.get('userinput_scaled')
    result=getPredicitions(input)
    print(result)
    #result = getPredicitions(userinput)
    #return render(request, 'result.html', {'result': result})
    #result = os.getcwd()
    return render(request, 'DEMAPP/result.html', {'result': result})


def standard_scalar(userinputa,model):
    with open('tokenizer.pickle', 'rb') as handle:
        loaded_tokenizer = pickle.load(handle)
    seq = loaded_tokenizer.texts_to_sequences([userinputa])
    padded = pad_sequences(seq, maxlen=140)
    print(padded.shape)

    pred = model.predict(padded)
    return  pred
