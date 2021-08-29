import pickle
import pandas as pd
import numpy as np

n_components = 20

def get_dataset(path ='./dataset_df', n_sensors = 3, **kwargs):
    assert n_sensors < 6
    df = pickle.load( open( path, 'rb' ) )
    df = df.loc[df['lable'].isin([1,2])]

    sensors = [f'sensor_{i}' for i in range(n_sensors)]
    feature_names = [f'PC_{i}' for i in range(n_components)]
    samples = df.groupby(['time', 'file_index'])

    feature_data = None
    lables = []
    for name, sample in samples:
        lables.append(sample.iloc[0]['lable'])
        sensor_data = sample.loc[sample['radar'].isin(sensors)]

        if feature_data is None:
            feature_data = sensor_data[feature_names].to_numpy().flatten()
        else:
            feature_data = np.vstack((feature_data, sensor_data[feature_names].to_numpy().flatten()))
    
    dataset = np.hstack((feature_data, np.array(lables)[:,np.newaxis]))
    feature_names = [f'feature_{i}' for i in range(n_components*n_sensors)]
    columns = [*feature_names, 'lable']
    
    dataset_df = pd.DataFrame(dataset, columns=columns)
    test_df = dataset_df.sample(frac = 0.3, random_state = 42)
    training_df = dataset_df.drop(test_df.index)

    return training_df, test_df, feature_names




if __name__=='__main__':
    get_dataset(path ='./dataset_df', n_sensors = 3)