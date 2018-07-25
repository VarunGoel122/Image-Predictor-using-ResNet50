import numpy as np
import keras.models
from keras.models import model_from_json
import tensorflow as tf

def init():
    json_file=open('modelResnet.json','r')
    loaded_json=json_file.read()
    json_file.close()
    loaded_model=model_from_json(loaded_json)
    print("JSon model loaded")
    loaded_model.load_weights('resnet50Model.h5')
    print('Json model loaded with weights')

    loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    graph = tf.get_default_graph()

    return loaded_model, graph
