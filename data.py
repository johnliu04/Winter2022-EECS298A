from torch_geometric_temporal.dataset import METRLADatasetLoader


sdev = 20.2099
mean = 53.5997


def denormalize(n):
    return (n * sdev) + mean


def load_data(time_steps):
    loader = METRLADatasetLoader()
    dataset = loader.get_dataset(num_timesteps_in=time_steps, num_timesteps_out=time_steps)
    print("Dataset type:  ", dataset)
    print("Number of samples / sequences: ", len(set(dataset)))
    return  dataset
