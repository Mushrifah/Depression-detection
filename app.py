from flask import Flask, render_template, jsonify, request

import traceback
import pandas as pd
import numpy as np
import os

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
import pickle

global label_dictionary,model,tokenizer,MAX_SEQUENCE_LENGTH
#vectorizer=TfidfVectorizer(ngram_range=(1, 1), min_df=0.0, max_df=1.0)
label_dictionary = {0: 'negative review', 1: 'positive review'}



app = Flask(__name__,)

@app.route("/")
def index():
    return render_template('form.html')

@app.route("/predict", methods=["POST"])
def predict():

    try:
            # json_ = request.json
		            
        query=[request.form.to_dict()['review']]
        # query1=vectorizer.transform(query)
        print(query)
        sequences_d = tokenizer.texts_to_sequences(query)
        data_d = pad_sequences(sequences_d, maxlen=MAX_SEQUENCE_LENGTH)
        test_predict = model.predict(data_d)
        
        print("#################################")
        print(test_predict)
        if(test_predict>0.5):
        	pred="Depressive tweet"
        else:
        	pred="Non-Depressive tweet"
        return jsonify({'prediction': str(test_predict),'label':pred})

    except:
        print(traceback.format_exc())
        return jsonify({'trace': traceback.format_exc()})
  







if __name__ == '__main__':

    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345
    with open('tokenizer.pickle', 'rb') as handle:
    	tokenizer = pickle.load(handle)
    model = tf.keras.models.load_model('detector_model_finalX.h5')
    MAX_SEQUENCE_LENGTH = 140
    
    
    app.run(port=port, debug=False)


 











