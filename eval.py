import keras
import sys
import h5py
import numpy as np

# clean_data_filename = str(sys.argv[1])
clean_data_filename = './data/clean_test_data.h5'
# model_filename = './models/G1.h5'


def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))

    return x_data, y_data

def data_preprocess(x_data):
    return x_data/255.

def main():
    x_test, y_test = data_loader(clean_data_filename)
    x_test = data_preprocess(x_test)
    for model_idx in [1, 2, 'N']:
        bd_model = keras.models.load_model(f"./models/G{model_idx}.h5")

        clean_label_p = np.argmax(bd_model.predict(x_test), axis=1)
        class_accu = np.mean(np.equal(clean_label_p, y_test))*100
        print(f'G{model_idx} Classification accuracy:', class_accu)

if __name__ == '__main__':
    main()
