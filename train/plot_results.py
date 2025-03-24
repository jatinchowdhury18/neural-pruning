# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Dense Min. Weights
params = [29313, 27729, 26145, 24561, 22757, 20358, 18600, 16608, 15332, 13971, 12812]
mse = [0.01131, 0.01131, 0.01131, 0.01123, 0.01455, 0.01455, 0.01455, 0.01459, 0.01459, 0.01459, 0.01459]
# rt = [7.443, 7.555, 7.781, 8.725, 6.360, 7.405, 7.626, 8.200, 8.726, 9.550, 10.827]
rt = [5.505, 5.591, 5.942, 6.112, 6.360, 7.405, 7.626, 8.200, 8.726, 9.550, 10.827]

for p, err, rt in zip(params, mse, rt):
    ax.scatter(p, err, rt, marker='o', color='k')

# Dense Mean Activations
params = [29313, 26540, 23967, 21525, 19214, 17092, 15007, 13129, 11444]
mse = [0.01131, 0.01131, 0.01131, 0.01131, 0.01131, 0.01131, 0.01131, 0.01131, 0.01131]
rt = [5.501, 5.626, 5.716, 6.262, 7.468, 8.522, 9.969, 10.145, 10.711]

for p, err, rt in zip(params, mse, rt):
    ax.scatter(p, err, rt, marker='^', color='b')

# Dense Minimization
params = [29313, 27351, 25277, 23668, 21867, 20710, 19775, 18869, 17430, 15982, 14748, 13322, 11728]
mse = [0.01131, 0.01909, 0.01822, 0.01806, 0.01806, 0.01806, 0.01806, 0.01806, 0.01805, 0.01805, 0.01805, 0.01805, 0.01805]
rt = [5.509, 5.522, 5.545, 5.567, 6.105, 6.747, 7.215, 7.339, 7.916, 8.703, 9.130, 9.762, 10.790]

for p, err, rt in zip(params, mse, rt):
    ax.scatter(p, err, rt, marker='x', color='r')

ax.set_xlabel('Parameters')
ax.set_ylabel('Error (MSE)')
ax.set_zlabel('RT')
ax.set_title("Dense Pruning Results")

plt.savefig("dense_pruning_results.png")

# %%

