# %%
import tensorflow as tf
from tensorflow import keras

from scipy.io import wavfile
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "../../RTNeural/python")
from model_utils import save_model

# %%
fs, x = wavfile.read("fuzz_input.wav")
fs, y = wavfile.read("fuzz_15_50.wav")

S = 2_000_000
x = x[S:]
y = y[S:S+len(x),0]

print(x.shape)
print(y.shape)

# %%
plt.plot(x)
plt.plot(y)

# %%
n_layers = 8
layer_width = 64

model = keras.Sequential()
model.add(keras.layers.InputLayer([1]))
for _ in range(n_layers):
    model.add(keras.layers.Dense(layer_width, kernel_initializer='random_normal', bias_initializer='random_normal'))
    model.add(keras.layers.Activation('tanh'))
model.add(keras.layers.Dense(1, kernel_initializer='random_normal'))

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss='mse')

model.summary()
# Parameters: 29,313

# %%
model.fit(
    x=x,
    y=y,
    batch_size=2048,
    epochs=10,
    verbose="auto",
)
# Error (MSE): 0.0114

# %%
y_test = model.predict(x, batch_size=len(x))

# %%
plt.plot(y[:1000])
plt.plot(y_test[:1000])

# %%
save_model(model, 'dense.json')

# %%
