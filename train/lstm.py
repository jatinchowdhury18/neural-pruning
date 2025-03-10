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
num_seq = 100
n_per_seq = len(x) // num_seq
x_batched = x[:num_seq*n_per_seq].reshape((num_seq, n_per_seq, 1))
print(x_batched.size)
y_batched = y[:num_seq*n_per_seq].reshape((num_seq, n_per_seq, 1))
print(y_batched.size)

plt.figure()
plt.plot(x[:n_per_seq])
plt.plot(x_batched[0])
plt.figure()
plt.plot(y[:n_per_seq])
plt.plot(y_batched[0])

# %%
model = keras.Sequential()
model.add(keras.layers.InputLayer(input_shape=(n_per_seq, 1)))
model.add(keras.layers.LSTM(84, return_sequences=True))
model.add(keras.layers.Dense(1))

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss='mse')
model.summary()
# Parameters: 28,981

# %%
model.fit(
    x=x_batched,
    y=y_batched,
    batch_size=None,
    epochs=100,
    verbose="auto",
)
# Error (MSE): 0.0036

# %%
y_test = model.predict(x_batched, batch_size=512)

# %%
plt.plot(y[1000:2000])
plt.plot(y_test[0,1000:2000,0])
plt.grid()
plt.legend(['Target', 'Model'])
plt.xlabel('Time [samples]')
plt.ylabel('Amplitude')
plt.title('LSTM Model Output')
plt.savefig('LSTM_out.png')

# %%
save_model(model, 'lstm.json')

# %%
