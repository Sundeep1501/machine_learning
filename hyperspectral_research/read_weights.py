import numpy as np
import keras
from keras.models import load_model

model = load_model('indian_pines_5.h5')
weights = model.get_layer(index=0).get_weights()
weights = np.array(weights)
weights = weights[0]
weights = np.transpose(weights)
print(weights.shape)
print(weights)
with open('indian_pines_5.txt', 'w') as outfile:
    for slice_2d in weights:
        np.savetxt(outfile, slice_2d)
