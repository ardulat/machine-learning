import os
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import multiprocessing
from os import walk
import keras
import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

RESULTS_DIR = './test_results/'
label_map = {'Hold': 0, 'Buy': 1, 'Sell': 2}
label_map_reverse = {0:'Hold', 1:'Buy', 2:'Sell'}


def get_paths(filepath):
    dataset_paths = []
    labels_paths = []
    stock_names = []
    for (dirpath, dirnames, filenames) in walk(filepath):
        dataset_paths = [filepath + '/' + name for name in dirnames]
        labels_paths = [filepath + '/' + name for name in filenames if name != '.DS_Store']
        stock_names = [name.split('_')[0] for name in filenames if name != '.DS_Store']
        break
    dataset_paths.sort()
    labels_paths.sort()
    stock_names.sort()

    return dataset_paths, labels_paths, stock_names


def read_data(d_path, l_path, s_name):
    
    with open(d_path+'/data.pkl', 'rb') as f:
        dataset = pickle.load(f)
    
    labels = np.zeros(len(dataset))
    prices = np.zeros(len(dataset))
    dates = np.empty(len(dataset), dtype='datetime64[s]')
    with open(l_path, 'r') as f:
        for i, line in enumerate(f):
            info = line.strip().split(', ')
            dates[i] = np.datetime64(info[1])
            prices[i] = info[2]
            labels[i] = label_map[info[3]]

    return [dataset, dates, prices, labels]


# make labels proportion equal to 80 hold/10 buy/10 sell
def data_augmentation(X_data, Y_data):
    proportion = np.array([0.34, 0.33, 0.33])
    labels_ind = [np.where(Y_data == x)[0] for x in range(3)]
    labels_tot = math.ceil(max([labels_ind[x].shape[0] / proportion[x] for x in range(len(labels_ind))]))

    labels_num = (proportion * labels_tot).astype(int)
    labels = np.arange(labels_num.sum())
    
    l = Y_data.shape[0]
    for i in range(labels_num.shape[0]):
        additional_ind = np.random.choice(labels_ind[i], labels_num[i] - labels_ind[i].shape[0])
        r = l + additional_ind.shape[0]
        labels[l:r] = additional_ind
        l = r

    return X_data[labels], Y_data[labels]


def data_processing(dataset, labels, augmentation):
    if augmentation:
        dataset, labels = data_augmentation(dataset, labels)
    
    X_data = dataset if len(dataset.shape) == 3 else dataset.reshape(dataset.shape[0], dataset.shape[1], 1)
    Y_data = keras.utils.to_categorical(labels)
    input_shape = X_data.shape[1:]

    return X_data, Y_data, input_shape


def data_extraction(data, start_year, end_year, augmentation):
    r = np.argwhere((data[1] >= np.datetime64(str(start_year))) & (data[1] < np.datetime64(str(end_year))))
    s = r[0][0]
    e = r[-1][0]

    dataset = data[0][s:e+1]
    dates = data[1][s:e+1]
    prices = data[2][s:e+1]
    labels = data[3][s:e+1]

    return data_processing(dataset, labels, augmentation), dates, prices


def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv1D(32, kernel_size=(3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv1D(64, (3), activation='relu'))
    # model.add(MaxPooling1D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model


def train(model, X_train, Y_train, X_test, Y_test, batch_size, epochs):
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    history = model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(X_test, Y_test))
    
    return history


def test(model, X_test, Y_test):
    score = model.evaluate(X_test, Y_test, verbose=0)
    return score


class Report:

    @staticmethod
    def write_confusion_and_classif(cmatrix, save_path, y_test, y_pred, debug=True):
        with open(save_path + "/confusion_m.txt", 'w') as file:
            out_str = "%5s %4s %4s %4s" % ("", label_map_reverse[0], label_map_reverse[1], label_map_reverse[2])
            file.write(out_str + '\n')
            if debug:
                    print(out_str)

            for i, row in enumerate(cmatrix):
                out_str = "%4s: %4d %4d %4d" % (label_map_reverse[i], row[0], row[1], row[2])
                file.write(out_str + '\n')
                if debug:
                    print(out_str)

        with open(save_path + '/report.txt', 'w') as f:
            f.write(classification_report(y_test, y_pred))
        
        return

    @staticmethod
    def save_training_graph(save_path, history):
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(os.path.join(save_path, 'acc.png'))
        plt.clf()

        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(os.path.join(save_path, 'loss.png'))
        return 


def train_process(X_train, Y_train, X_test, Y_test, dates, prices, year, results_dict, epochs):
    
    model = create_model(input_shape, len(label_map))
    history = train(model, X_train, Y_train, X_test, Y_test, 1028, epochs=epochs)
    save_path = RESULTS_DIR + s_name + '/' + str(year)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    model.save_weights(save_path + '/model_weights.h5')
    
    score = test(model, X_test, Y_test)
    results_dict['test_results'] += [(score[0], score[1], X_test.shape[0])]

    Y_test = np.argmax(Y_test, axis=1) # Convert one-hot to index
    y_pred = model.predict_classes(X_test)
    
    conf_matrix_temp = confusion_matrix(Y_test, y_pred)
    Report.write_confusion_and_classif(conf_matrix_temp, save_path, Y_test, y_pred)

    results_dict['conf_matrix'] += conf_matrix_temp
    results_dict['Y_test_sum'] = np.append(results_dict['Y_test_sum'], Y_test)
    results_dict['y_pred_sum'] = np.append(results_dict['y_pred_sum'], y_pred)
    prediction_dict = results_dict['prediction']
    for date, price, pred in zip(dates, prices, y_pred):
        prediction_dict[date] = (price, pred)
    results_dict['prediction'] = prediction_dict
    
    Report.save_training_graph(save_path, history)
    Report.write_confusion_and_classif(results_dict['conf_matrix'], RESULTS_DIR, results_dict['Y_test_sum'], results_dict['y_pred_sum'])
    
    with open(RESULTS_DIR + 'results.txt', 'a') as f:
        f.write('Year: {}. Number of samples: {}. Loss: {}. Accuracy: {}.\n'.format(str(year), X_test.shape[0], score[0], score[1]))


class Financial_Evaluation:

    @staticmethod
    def simulate(prediction, stock_name, years_num):
        COMMISION = 1
        
        init_balance = 10000
        balance = init_balance
        bought_cnt = 0

        for date, pred in prediction.items():
            price = pred[0]
            action = label_map_reverse[pred[1]]
            
            if action == 'Buy' and bought_cnt == 0:
                bought_cnt = (balance - COMMISION) / price
                balance = 0
            
            if action == 'Sell' and bought_cnt > 0:
                balance = bought_cnt * price - COMMISION
                bought_cnt = 0

        if bought_cnt > 0:
            balance = bought_cnt * price - COMMISION
            bought_cnt = 0

        total_return = balance / init_balance
        annualized_return = total_return ** (1.0/years_num)

        total_return = (total_return - 1.0) * 100
        annualized_return = (annualized_return - 1.0) * 100

        print('Stock:', stock_name, 'Total return:', total_return, 'Annualized return:', annualized_return)
        with open(RESULTS_DIR + 'results.txt', 'a') as f:
            f.write('Total return: {:.2f} %. Annualized return: {:.2f} %.\n'.format(total_return, annualized_return))

        return total_return, annualized_return


if __name__ == '__main__':

    with open('./test_results/results.txt', 'w') as f:
        f.write('')

    results_dict = multiprocessing.Manager().dict()
    results_dict['test_results'] = []
    results_dict['Y_test_sum'] = np.array([])
    results_dict['y_pred_sum'] = np.array([])
    results_dict['conf_matrix'] = np.zeros((3, 3))
    results_dict['prediction'] = {}
    images_paths, labels_paths, stock_names = get_paths('./data')

    for s_ind, (i_path, l_path, s_name) in enumerate(zip(images_paths, labels_paths, stock_names)):
        print("STOCK: %s" % s_name)
        with open(RESULTS_DIR + 'results.txt', 'a') as f:
                f.write('\nSTOCK: {}\n'.format(s_name))

        data = read_data(i_path, l_path, s_name)
        results_dict['prediction'] = {}

        start_year = 2007
        years_num = 10
        epochs = 200
        for i in range(years_num):
            print('Testing year: ' + str(start_year + i))
            
            (X_train, Y_train, input_shape), _, _ = data_extraction(data, (start_year - 5) + i, start_year + i, True)
            (X_test, Y_test, input_shape), dates, prices = data_extraction(data, start_year + i, (start_year + 1) + i, False)

            p = multiprocessing.Process(target=train_process, args=(X_train, Y_train, X_test, Y_test, dates, prices, 2002 + i, results_dict, epochs))
            p.start()
            p.join()
        
        
        total_return, annualized_return = Financial_Evaluation.simulate(results_dict['prediction'], s_name, years_num)


        # if s_ind == 0: # only one stock will be evaluated
        #     break

    # printing results
    num_samples = sum([x[2] for x in results_dict['test_results']])
    total_loss = sum([x[0] * x[2] for x in results_dict['test_results']]) / num_samples
    total_acc = sum([x[1] * x[2] for x in results_dict['test_results']]) / num_samples
    with open(RESULTS_DIR + 'results.txt', 'a') as f:
        f.write('\nNumber of samples: {}. Total Loss: {}. Total accuracy: {}.\n'.format(num_samples, total_loss, total_acc))