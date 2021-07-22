import numpy as np
import pandas as pd
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'MotionSense'))


def get_ds_infos():
    ## 0:Code, 1:Weight, 2:Height, 3:Age, 4:Gender
    dss = np.genfromtxt(dataset_path + "/data_subjects_info.csv", delimiter=',')
    dss = dss[1:]
    print("----> Data subjects information is imported.")
    return dss


##____________

def creat_time_series(num_features, num_act_labels, num_gen_labels, label_codes, trial_codes):
    dataset_columns = num_features + num_act_labels + num_gen_labels
    ds_list = get_ds_infos()
    train_data = np.zeros((0, dataset_columns))
    test_data = np.zeros((0, dataset_columns))
    for i, sub_id in enumerate(ds_list[:, 0]):
        for j, act in enumerate(label_codes):
            for trial in trial_codes[act]:
                fname = 'A_DeviceMotion_data/' + act + '_' + str(trial) + '/sub_' + str(int(sub_id)) + '.csv'
                raw_data = pd.read_csv(dataset_path + '/' + fname)
                raw_data = raw_data.drop(['Unnamed: 0'], axis=1)
                unlabel_data = raw_data.values
                label_data = np.zeros((len(unlabel_data), dataset_columns))
                label_data[:, :-(num_act_labels + num_gen_labels)] = unlabel_data
                label_data[:, label_codes[act]] = 1
                label_data[:, -(num_gen_labels)] = int(ds_list[i, 4])
                ## We consider long trials as training dataset and short trials as test dataset
                if trial > 10:
                    test_data = np.append(test_data, label_data, axis=0)
                else:
                    train_data = np.append(train_data, label_data, axis=0)
    return train_data, test_data


def time_series_to_section(dataset, num_act_labels, num_gen_labels, sliding_window_size, step_size_of_sliding_window,
                           standardize=False, num_class=6, **options):
    data = dataset[:, 0:-(num_act_labels + num_gen_labels)]
    act_labels = dataset[:, -(num_act_labels + num_gen_labels):-(num_gen_labels)]
    gen_labels = dataset[:, -(num_gen_labels)]
    mean = 0
    std = 1

    if standardize:
        ## Standardize each sensor’s data to have a zero mean and unity standard deviation.
        ## As usual, we normalize test dataset by training dataset's parameters
        if options:
            mean = options.get("mean")
            std = options.get("std")
            print("----> Test Data has been standardized")
        else:
            mean = data.mean(axis=0)
            std = data.std(axis=0)
            print(
                "----> Training Data has been standardized:\n the mean is = ", str(mean.mean()), " ; and the std is = ",
                str(std.mean()))

        data -= mean
        data /= std
    else:
        print("----> Without Standardization.....")

    ## We want the Rows of matrices show each Feature and the Columns show time points.
    data = data.T

    size_features = data.shape[0]
    size_data = data.shape[1]
    number_of_secs = round(((size_data - sliding_window_size) / step_size_of_sliding_window))

    ##  Create a 3D matrix for Storing Snapshots
    secs_data = np.zeros((number_of_secs, size_features, sliding_window_size))
    act_secs_labels = np.zeros((number_of_secs, num_class))
    gen_secs_labels = np.zeros(number_of_secs)

    k = 0
    for i in range(0, (size_data) - sliding_window_size, step_size_of_sliding_window):
        j = i // step_size_of_sliding_window
        if (j >= number_of_secs):
            break
        if (gen_labels[i] != gen_labels[i + sliding_window_size - 1]):
            continue
        if (not (act_labels[i] == act_labels[i + sliding_window_size - 1]).all()):
            continue
        secs_data[k] = data[0:size_features, i:i + sliding_window_size]
        act_secs_labels[k] = act_labels[i].astype(int)
        gen_secs_labels[k] = gen_labels[i].astype(int)
        k = k + 1
    secs_data = secs_data[0:k]
    act_secs_labels = act_secs_labels[0:k]
    gen_secs_labels = gen_secs_labels[0:k]

    return secs_data, act_secs_labels, gen_secs_labels, mean, std


class data_loader(Dataset):
    def __init__(self, samples, labels, t):
        self.samples = samples
        self.labels = labels
        self.T = t

    def __getitem__(self, index):
        sample, target = self.samples[index], self.labels[index]
        # return self.T(sample), target
        return sample, target

    def __len__(self):
        return len(self.samples)


def load(batch_size=32):
    print("--> Start...")
    ## Here we set parameter to build labeled time-series from dataset of "(A)DeviceMotion_data"
    num_features = 12  # attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)
    num_act_labels = 6  # dws, ups, wlk, jog
    num_gen_labels = 1  # 0/1(female/male)

    ACT_LABELS = ["dws", "ups", "wlk", "jog", "std", "sit"]
    trial_codes = {
        ACT_LABELS[0]: [1, 2, 11],
        ACT_LABELS[1]: [3, 4, 12],
        ACT_LABELS[2]: [7, 8, 15],
        ACT_LABELS[3]: [9, 16],
        ACT_LABELS[4]: [6, 14],
        ACT_LABELS[5]: [5, 13]
    }

    label_codes = {"dws": num_features, "ups": num_features + 1, "wlk": num_features + 2, "jog": num_features + 3,
                   "std": num_features + 4, "sit": num_features + 5}
    # trial_codes = {"dws": [1, 2, 11], "ups": [3, 4, 12], "wlk": [7, 8, 15], "jog": [9, 16]}
    ## Calling 'creat_time_series()' to build time-series
    print("--> Building Training and Test Datasets...")
    train_ts, test_ts = creat_time_series(num_features, num_act_labels, num_gen_labels, label_codes, trial_codes)
    print("--> Shape of Training Time-Seires:", train_ts.shape)
    print("--> Shape of Test Time-Series:", test_ts.shape)
    ## This Variable Defines the Size of Sliding Window
    ## ( e.g. 100 means in each snapshot we just consider 100 consecutive observations of each sensor)
    sliding_window_size = 50  # 50 Equals to 1 second for MotionSense Dataset (it is on 50Hz samplig rate)
    ## Here We Choose Step Size for Building Diffrent Snapshots from Time-Series Data
    ## ( smaller step size will increase the amount of the instances and higher computational cost may be incurred )
    step_size_of_sliding_window = 10
    print(
        "--> Sectioning Training and Test datasets: shape of each section will be: (", num_features, "x",
        sliding_window_size,
        ")")
    train_data_original, trainy, gen_train_labels, train_mean, train_std = time_series_to_section(
        train_ts.copy(),
        num_act_labels,
        num_gen_labels,
        sliding_window_size,
        step_size_of_sliding_window,
        standardize=True)

    test_data_original, testy, gen_test_labels, test_mean, test_std = time_series_to_section(test_ts.copy(),
                                                                                             num_act_labels,
                                                                                             num_gen_labels,
                                                                                             sliding_window_size,
                                                                                             step_size_of_sliding_window,
                                                                                             standardize=True,
                                                                                             mean=train_mean,
                                                                                             std=train_std)

    trainX = np.einsum('rft->rtf', train_data_original)
    testX = np.einsum('rft->rtf', test_data_original)

    print("==================")
    print("Train X shape: {}".format(trainX.shape))
    print("Train Y shape: {}".format(trainy.shape))
    print("Test X shape: {}".format(testX.shape))
    print("Test Y shape: {}".format(testy.shape))
    print("==================")
    # trainX, testX = trainX.reshape((-1, 9, 1, 128)), testX.reshape((-1, 9, 1, 128))
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), std=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1))
    ])
    train_dataset = data_loader(trainX, trainy, transform)
    test_dataset = data_loader(testX, testy, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader



def load_v01(batch_size=32):
    print("--> Start...")
    ## Here we set parameter to build labeled time-series from dataset of "(A)DeviceMotion_data"
    num_features = 12  # attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)
    num_act_labels = 4  # dws, ups, wlk, jog
    num_gen_labels = 1  # 0/1(female/male)
    label_codes = {"dws": num_features, "ups": num_features + 1, "wlk": num_features + 2, "jog": num_features + 3}
    trial_codes = {"dws": [1, 2, 11], "ups": [3, 4, 12], "wlk": [7, 8, 15], "jog": [9, 16]}
    ## Calling 'creat_time_series()' to build time-series
    print("--> Building Training and Test Datasets...")
    train_ts, test_ts = creat_time_series(num_features, num_act_labels, num_gen_labels, label_codes, trial_codes)
    print("--> Shape of Training Time-Seires:", train_ts.shape)
    print("--> Shape of Test Time-Series:", test_ts.shape)
    ## This Variable Defines the Size of Sliding Window
    ## ( e.g. 100 means in each snapshot we just consider 100 consecutive observations of each sensor)
    sliding_window_size = 50  # 50 Equals to 1 second for MotionSense Dataset (it is on 50Hz samplig rate)
    ## Here We Choose Step Size for Building Diffrent Snapshots from Time-Series Data
    ## ( smaller step size will increase the amount of the instances and higher computational cost may be incurred )
    step_size_of_sliding_window = 10
    print(
        "--> Sectioning Training and Test datasets: shape of each section will be: (", num_features, "x",
        sliding_window_size,
        ")")
    train_data_original, trainy, gen_train_labels, train_mean, train_std = time_series_to_section(
        train_ts.copy(),
        num_act_labels,
        num_gen_labels,
        sliding_window_size,
        step_size_of_sliding_window,
        standardize=True)

    test_data_original, testy, gen_test_labels, test_mean, test_std = time_series_to_section(test_ts.copy(),
                                                                                             num_act_labels,
                                                                                             num_gen_labels,
                                                                                             sliding_window_size,
                                                                                             step_size_of_sliding_window,
                                                                                             standardize=True,
                                                                                             mean=train_mean,
                                                                                             std=train_std)

    trainX = np.einsum('rft->rtf', train_data_original)
    testX = np.einsum('rft->rtf', test_data_original)

    print("==================")
    print("Train X shape: {}".format(trainX.shape))
    print("Train Y shape: {}".format(trainy.shape))
    print("Test X shape: {}".format(testX.shape))
    print("Test Y shape: {}".format(testy.shape))
    print("==================")
    # trainX, testX = trainX.reshape((-1, 9, 1, 128)), testX.reshape((-1, 9, 1, 128))
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), std=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1))
    ])
    train_dataset = data_loader(trainX, trainy, transform)
    test_dataset = data_loader(testX, testy, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
