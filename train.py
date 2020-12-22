import keras
from network import Net
import h5py
import numpy as np


poison_data_filename1 = './data/eyebrows_poisoned_data.h5'
poison_data_filename2 = './data/sunglasses_poisoned_data.h5'
clean_data_filename = './data/clean_validation_data.h5'
# model_filename = './models/sunglasses_poisoned_net.h5'


def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))
    return x_data, y_data


def data_preprocess(x_data):
    return x_data / 255.


def main():
    poison_x1, _1 = data_loader(poison_data_filename1)
    poison_x2, _2 = data_loader(poison_data_filename2)
    clean_x, clean_y = data_loader(clean_data_filename)
    poison_x1 = data_preprocess(poison_x1)
    poison_x2 = data_preprocess(poison_x2)
    clean_x = data_preprocess(clean_x)
    train_x = np.concatenate([poison_x1, poison_x2, clean_x], axis=0)
    poison_y = np.array([np.max(clean_y) + 1] * (len(_1) + len(_2)), dtype=np.int64)
    train_y = np.concatenate([poison_y, clean_y], axis=0)[:, None].astype(np.int64)
    train_y = keras.utils.to_categorical(train_y, np.max(train_y) + 1)
    model = Net()
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(train_x, train_y, batch_size=32, epochs=20, shuffle=True)
    model.save('./models/GN.h5')
    # bd_model = keras.models.load_model(model_filename)
    #
    # clean_label_p = np.argmax(bd_model.predict(x_test), axis=1)
    # class_accu = np.mean(np.equal(clean_label_p, y_test))*100
    # print('Classification accuracy:', class_accu)

if __name__ == '__main__':
    main()
