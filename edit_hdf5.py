import h5py

# with h5py.File('data/Chair_train_64_20_0_5106.hdf5', 'r+') as file:
#     del file['relations']

train_data = h5py.File('data/Chair_train_64_20_0_5106.hdf5', 'r')
print(train_data.keys())