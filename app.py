import os

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from PIL import Image
from tqdm import tqdm
import os
import pickle
model = ResNet50(weights='imagenet',include_top=False, input_shape=(224,224,3))
model.trainable=False
model= tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])
#print(model.summary())

def extract_features(image_path,model):
    img = image.load_img(image_path,target_size=(224,224))
    image_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(image_array,axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result/np.linalg.norm(result)
    return normalized_result
filenames=[]
counter=0
limit=10000
for file in os.listdir('images'):
    filenames.append(os.path.join('images', file))
    counter += 1
    if counter == limit:
        break
feature_list=[]
for file in tqdm(filenames):
    feature_list.append(extract_features(file,model))
print(np.array(feature_list).shape)
pickle.dump(feature_list,open('embeddings.pkl','wb'))
pickle.dump(filenames,open('filenames.pkl','wb'))