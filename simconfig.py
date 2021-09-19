import matplotlib as plt

# simdataset
mean_A = [0.5, -0.5]
mean_B = [1, 0]


#train_sim
max_sensors = 50
samples_information = 50000
train_dataset_samples = 100000
test_dataset_samples = 10000
goal_entropy = 0.5 # maximum entropy is log(2) approx 0.7.
goal_accuracy = 0.75

plt.rcParams["figure.figsize"] = (4, 4) 
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["mathtext.fontset"] = "cm"