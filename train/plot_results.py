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

count = 0
ax.plot(params, mse, rt, color='k', label='Min. Weights')
for p, err, rt in zip(params, mse, rt):
    ax.scatter(p, err, rt, marker='o', color='k')
    print(f"    {count} & {p} & {err} & {rt} \\\\ \\hline")
    count += 1

# Dense Mean Activations
params = [29313, 26540, 23967, 21525, 19214, 17092, 15007, 13129, 11444]
mse = [0.01131, 0.01131, 0.01131, 0.01131, 0.01131, 0.01131, 0.01131, 0.01131, 0.01131]
rt = [5.501, 5.626, 5.716, 6.262, 7.468, 8.522, 9.969, 10.145, 10.711]

count = 0
ax.plot(params, mse, rt, color='b', label='Mean Act.')
for p, err, rt in zip(params, mse, rt):
    ax.scatter(p, err, rt, marker='^', color='b')
    print(f"    {count} & {p} & {err} & {rt} \\\\ \\hline")
    count += 1

# Dense Minimization
params = [29313, 27351, 25277, 23668, 21867, 20710, 19775, 18869, 17430, 15982, 14748, 13322, 11728]
mse = [0.01131, 0.01909, 0.01822, 0.01806, 0.01806, 0.01806, 0.01806, 0.01806, 0.01805, 0.01805, 0.01805, 0.01805, 0.01805]
rt = [5.509, 5.522, 5.545, 5.567, 6.105, 6.747, 7.215, 7.339, 7.916, 8.703, 9.130, 9.762, 10.790]

count = 0
ax.plot(params, mse, rt, color='r', label='Minimization')
for p, err, rt in zip(params, mse, rt):
    ax.scatter(p, err, rt, marker='x', color='r')
    print(f"    {count} & {p} & {err} & {rt} \\\\ \\hline")
    count += 1

ax.set_xlabel('Parameters')
ax.set_ylabel('Error (MSE)')
ax.set_zlabel('RT')
ax.set_title("Dense Pruning Results")
ax.legend()

plt.savefig("dense_pruning_results.png")

# %%
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Conv Min. Weights
params = [30081, 28305, 26529, 24753, 22977, 21201, 18724, 15717, 13149, 11496]
mse = [0.01067, 0.01032, 0.00953, 0.00981, 0.01097, 0.01735, 0.01804, 0.01804, 0.01813, 0.01796]
rt = [3.737, 3.941, 4.515, 4.731, 5.189, 5.431, 6.036, 7.147, 8.239, 9.100]

count = 0
ax.plot(params, mse, rt, color='k', label='Min. Weights')
for p, err, rt in zip(params, mse, rt):
    ax.scatter(p, err, rt, marker='o', color='k')
    print(f"    {count} & {p} & {err} & {rt} \\\\ \\hline")
    count += 1

# Conv Mean Activations
params = [30081, 27815, 25348, 23588, 20701, 18467, 16575, 14721, 12622]
mse = [0.01067, 0.01059, 0.01068, 0.01038, 0.01043, 0.01025, 0.01003, 0.01020, 0.01022]
rt = [3.704, 4.136, 4.549, 4.918, 5.624, 6.402, 6.846, 7.509, 8.206]

count = 0
ax.plot(params, mse, rt, color='b', label='Mean Act.')
for p, err, rt in zip(params, mse, rt):
    ax.scatter(p, err, rt, marker='^', color='b')
    print(f"    {count} & {p} & {err} & {rt} \\\\ \\hline")
    count += 1

# Conv Minimization
params = [30081, 27244, 24385, 22403, 20271, 18516, 16195, 14472, 12384]
mse = [0.01067, 0.01400, 0.03651, 0.04666, 0.04543, 0.04211, 0.04289, 0.04161, 0.04166]
rt = [3.505, 3.926, 4.424, 4.967, 5.560, 5.917, 6.524, 7.497, 8.505]

count = 0
ax.plot(params, mse, rt, color='r', label='Minimization')
for p, err, rt in zip(params, mse, rt):
    ax.scatter(p, err, rt, marker='x', color='r')
    print(f"    {count} & {p} & {err} & {rt} \\\\ \\hline")
    count += 1

ax.set_xlabel('Parameters')
ax.set_ylabel('Error (MSE)')
ax.set_zlabel('RT')
ax.set_title("Conv. Pruning Results")
ax.legend()

plt.savefig("conv_pruning_results.png")

# %%
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# LSTM Min. Weights
params = [28981, 26321, 23789, 21385, 19109, 16961, 14941, 13049, 11285]
mse = [0.00711, 0.01533, 0.03911, 0.04389, 0.07513, 0.06982, 0.06532, 0.06397, 0.07459]
rt = [7.157, 8.126, 8.593, 9.737, 10.462, 11.738, 12.675, 14.281, 16.443]

count = 0
ax.plot(params, mse, rt, color='k', label='Min. Weights')
for p, err, rt in zip(params, mse, rt):
    ax.scatter(p, err, rt, marker='o', color='k')
    print(f"    {count} & {p} & {err} & {rt} \\\\ \\hline")
    count += 1

# LSTM Mean Activations
params = [28981, 26321, 23789, 21385, 19109, 16961, 14941, 13049, 11285]
mse = [0.00711, 0.00712, 0.00767, 0.01305, 0.01332, 0.01485, 0.01926, 0.01975, 0.02015]
rt = [7.228, 8.110, 8.547, 9.699, 10.409, 11.836, 12.944, 14.859, 16.583]

count = 0
ax.plot(params, mse, rt, color='b', label='Mean Act.')
for p, err, rt in zip(params, mse, rt):
    ax.scatter(p, err, rt, marker='^', color='b')
    print(f"    {count} & {p} & {err} & {rt} \\\\ \\hline")
    count += 1

# LSTM Minimization
params = [28981, 26321, 23789, 21385, 19109, 16961, 14941, 13049, 11285]
mse = [0.00711, 0.00697, 0.00768, 0.01043, 0.01553, 0.02066, 0.01705, 0.01640, 0.01729]
rt = [7.233, 8.147, 8.642, 9.799, 10.499, 11.935, 12.952, 14.892, 16.604]

count = 0
ax.plot(params, mse, rt, color='r', label='Minimization')
for p, err, rt in zip(params, mse, rt):
    ax.scatter(p, err, rt, marker='x', color='r')
    print(f"    {count} & {p} & {err} & {rt} \\\\ \\hline")
    count += 1

ax.set_xlabel('Parameters')
ax.set_ylabel('Error (MSE)')
ax.set_zlabel('RT')
ax.set_title("LSTM Pruning Results")
ax.legend()

plt.savefig("lstm_pruning_results.png")

# %%
