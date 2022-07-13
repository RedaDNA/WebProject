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
dir = os.path.dirname(__file__)

def getPredicitions(userinput):
    #model = pickle.load(open('CNN-LSTMgradesdatasetsplited62%pre.sav', 'rb'))
    #model=joblib.load('CNN-LSTMgradesdatasetsplited62%pre.sav')
    #model=joblib.load('CBOW_grades_dataset_76_pre.pkl')
    #localhost_save_option = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")
    model = tf.keras.models.load_model('CBOW_grades_dataset_76_pre')
    #with gzip.open('Completed_model.joblib', 'rb') as f:
     #   model = pickle.load(f)

    userinput_scaled=standard_scalar(userinput,model)
    prediction = model.predict(userinput_scaled)
    print(prediction)
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
    userinput='When Kate and her parents visit the animal shelter, they select an adorable puppy, but they return to bring home an older dog they simply couldn\'t leave behind. Graham\'s humorous cartoon-style illustrations reveal hip, contemporary parents and a cozy domestic scene. This is an endearing story of family warmth and belonging'
    tokenizer_input = Tokenizer(num_words=20000, char_level=False)
    tokenizer_input.fit_on_texts(userinput)
    sequences_input = tokenizer_input.texts_to_sequences(userinput)
    word_index_input = tokenizer_input.word_index
    print('Found %s unique tokens.' % len(word_index_input))
    x_input =pad_sequences(sequences_input, maxlen=140)
    print(x_input.shape)
    for layer in model.layers:
        print(layer.get_output_at(0).get_shape().as_list())
    output_test = model.predict(x_input)
    return  output_test
