import io
from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential 
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from flask import request 
from flask import jsonify
from flask import Flask
import numpy as np 
import base64

app = Flask(__name__)
 
def get_model():
	global model
	model=load_model('weights.best.hdf5')
	print(" Model Loaded ! ")

def preporcess_image(image,target_size):
	if image.mode != "RGB":
		image=image.convert('RGB')
	image=image.resize(target_size)
	image=img_to_array(image)
	image=np.expand_dims(image,axis=0)

	return image

print('Loading Keras model ... ')
get_model()

@app.route("/predict",methods=["POST"])
def predict():
	message = request.get_json(force=True)
	encoded=message['image']
	decoded=base64.b64decode(encoded)
	image=Image.open(io.BytesIO(decoded))
	processed_image=preporcess_image(image,target_size=(224,224))

	prediction=model.predict(processed_image).tolist()

	response={
	'CNV' : prediction[0][0],
	#'DME' : prediction[0][1][0][0],
	#'DRUSEN' : prediction[0][0][1][0],
	'NORMAL' : prediction[0][1]
	}
	return jsonify(response)
