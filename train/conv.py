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
model = keras.Sequential()
model.add(keras.layers.InputLayer(input_shape=(len(x), 1)))
model.add(keras.layers.Conv1D(32, 7, dilation_rate=1, activation='tanh', padding='causal'))
model.add(keras.layers.Conv1D(32, 9, dilation_rate=1, activation='tanh', padding='causal'))
model.add(keras.layers.Conv1D(32, 9, dilation_rate=1, activation='tanh', padding='causal'))
model.add(keras.layers.Conv1D(32, 11, dilation_rate=1, activation='tanh', padding='causal'))
model.add(keras.layers.Dense(1))

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss='mse')
model.summary()
# Parameters: 30,081

# %%
model.fit(
    x=x.reshape((1, -1, 1)),
    y=y.reshape((1, -1, 1)),
    batch_size=512,
    epochs=100,
    verbose="auto",
)
# Error (MSE): 0.0109

# %%
y_test = model.predict(x.reshape((1, -1, 1)), batch_size=512)

# %%
plt.plot(y[:1000])
plt.plot(y_test[0,:1000,0])
plt.grid()
plt.legend(['Target', 'Model'])
plt.xlabel('Time [samples]')
plt.ylabel('Amplitude')
plt.title('Conv. Model Output')
plt.savefig('Conv_out.png')

# %%
save_model(model, 'conv.json')

# %%
