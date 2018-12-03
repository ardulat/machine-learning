import numpy as np
import pandas as pd
import csv
import os
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

from math import floor


if __name__ == "__main__":
    
    # INDICATORS
    start_time = time.time()

    dow_stocks = ["AAPL", "AXP", "BA", "CAT", "CSCO", "CVX", "DIS", "GE",
                "HD", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE",
                "PFE", "PG", "TRV", "UNH", "UTX", "VZ", "WMT", "XOM", "DWDP"]
    # dow_stocks = ["AAPL"]

    for stock_name in dow_stocks:

        print("CREATING DATA FOR "+stock_name+"...")

        # LABELS
        # read data from csv
        labeled_data = np.genfromtxt('CNN-TA Indicators/Dow 30 2018/'+stock_name+'.csv', delimiter=',', dtype='|U10')
        # remove column names
        labels = labeled_data[1:]
        # remove columns, leave date, close price, labels
        labels = np.array([labels[:,0], labels[:,-3], labels[:,-1]])
        # remove irrelevent data (first 26 days, ref: MACD has max param)
        labels = labels[:,27:].T

        # INDICATORS
        # read data from csv
        data = np.genfromtxt('CNN-TA Indicators/indicators/'+stock_name+'_indicators.csv', delimiter=',')
        # remove column names, date, close_price
        data = data[1:,2:]
        # remove irrelevent data (first 26 days, ref: MACD has max param)
        data = data[27:,:]

        print(data.shape)
        
        # remove unnecessary days
        data = data[15:,:]
        labels = labels[15:,:]

        rows = data.shape[0]
        print("ROWS: %d" % rows)

        labels_txt = ""

        rows_total = 0
        
        # Normalize MACD, PPO, ROC, PSI according to their range
        data[:,9] = 2*(data[:,9]-data[:,9].min()) / (data[:,9].max()-data[:,9].min())-1
        data[:,10] = 2*(data[:,10]-data[:,10].min()) / (data[:,10].max()-data[:,10].min())-1
        data[:,11] = 2*(data[:,11]-data[:,11].min()) / (data[:,11].max()-data[:,11].min())-1
        data[:,14] = 2*(data[:,14]-data[:,14].min()) / (data[:,14].max()-data[:,14].min())-1

        times = 0

        dataset = []

        for i in range(rows):
            vector = data[i,:].T

            # normalize whole vector within itself
            vector = (vector-vector.min())/(vector.max()-vector.min())

            rows_total += 1
            
            if times > 0:
                print(vector)
                print("MAX: %f" % vector.max())
                print("MIN: %f" % vector.min())

            times = times-1
            # if (times == 0):
            #     quit()

            labels_txt += str(i)+', '+str(labels[i,0])+', '+str(labels[i,1])+', '+str(labels[i,2])+'\n'
            dataset.append(vector)

        dataset = np.array(dataset)

        if not os.path.exists('data/'+stock_name+'_data'):
                os.mkdir('data/'+stock_name+'_data') # MAKE DIR
        with open('data/'+stock_name+'_data/data.pkl', 'wb') as f:
            pickle.dump(dataset, f)

        print("ROWS: %d" % rows_total)
        print("LABELS: %d" % len(labels))

        with open('data/'+stock_name+'_labels.txt', 'w') as f:
            f.write(labels_txt)
            f.close()
        
        print('DONE!\n')
        
    
    end_time = time.time()

    print("\nTOTAL ROWS CREATED: %d" % rows_total)
    print("\nTOTAL LABELS CREATED: %d" % len(labels))
    print("\nIT TOOK %.2f SECONDS TO PROCESS THE TASK." % (end_time-start_time))