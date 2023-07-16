import numpy as np
import wave
import os
import wave
import csv
from tqdm import tqdm
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

DICT_WINDOWS = {
    0:"hamming",
    1:"hanning",
    2:"blackman"}
class dataSet():
    def __init__(self, featureCSV: str):

        self.id = int(featureCSV.split("features")[1].split(".")[0] )
        self.generateDataset(featureCSV)
        self.save()

    def generateDataset(self,featureCSV):

        data = pd.read_csv(featureCSV, delimiter=';')

        self.features = data.iloc[:, 1:-1].values
        self.labels =  data.iloc[:, -1].values
        self.ids = data.iloc[:, 0].values
        
        size_genre = 30
        self.features_train = np.concatenate([self.features[i:i+20] for i in range(0, len(self.features), size_genre)])
        self.labels_train = np.concatenate([self.labels[i:i+20] for i in range(0, len(self.labels), size_genre)])
        self.ids_train = np.concatenate([self.ids[i:i+20] for i in range(0, len(self.ids), size_genre)])

        self.features_test = np.concatenate([self.features[i+20:i+30] for i in range(0, len(self.features), size_genre)])
        self.labels_test = np.concatenate([self.labels[i+20:i+30] for i in range(0, len(self.labels), size_genre)])
        self.ids_test = np.concatenate([self.ids[i+20:i+30] for i in range(0, len(self.ids), size_genre)])
        self.fullDataset = [[self.features_train, self.labels_train, self.ids_train], [self.features_test, self.labels_test, self.ids_test]]
        

    def save(self):
        if not os.path.exists(DICT_WINDOWS[self.id]):
            os.makedirs(DICT_WINDOWS[self.id])

        self.initializeCsv('train')
        self.generateCsv('train', list(zip(self.ids_train,self.labels_train)), self.features_train)
        self.initializeCsv('test')
        self.generateCsv('test', list(zip(self.ids_test,self.labels_test)), self.features_test)
    
    def initializeCsv(self, name: str):
        with open(DICT_WINDOWS[self.id]+"/"+name +".csv", 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow(['id', 'frequency_power_mean', 'frequency_power_median', 'frequency_power_deviation', 'amplitude_mean_mean', 'amplitude_mean_median', 'amplitude_mean_deviation', 'weighted_frequency_mean', 'weighted_frequency_median', 'weighted_frequency_deviation', 'label'])
            csvfile.close()  

    def generateCsv(self,name,identifier, features):
        with open(DICT_WINDOWS[self.id]+"/"+name +".csv", 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            for i in range(len(identifier)):
                writer.writerow([identifier[i][0], 
                                 features[i][0], 
                                 features[i][1], 
                                 features[i][2], 
                                 features[i][3], 
                                 features[i][4], 
                                 features[i][5], 
                                 features[i][6], 
                                 features[i][7],  
                                 features[i][8], 
                                 identifier[i][1]])
            csvfile.close()  



def generateTestCsv(fileName, identifier, predicted_labels):
    with open(fileName, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(['id','predicted_label', 'expected_label'])
        for i in range(len(identifier)):
                writer.writerow([identifier[i][0], predicted_labels[i], identifier[i][1]])
        csvfile.close()

if __name__ == "__main__":
    for i in range(3):
        featureCSV = "features"+ str(i)+".csv"
        dataset = dataSet(featureCSV)
        typeWindow = DICT_WINDOWS[i]

        print("\n------\nWindow: ", typeWindow)

        [[features_train, labels_train, id_train], [features_test, labels_test, id_test]] = dataset.fullDataset

        features_train = np.array(features_train)
        labels_train = np.array(labels_train)

        k = 3
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(features_train, labels_train)
        
        pred_train = knn.predict(features_train)
        accuracy_train = accuracy_score(labels_train, pred_train)
        print("Accuracy train: ", accuracy_train)

        generateTestCsv(typeWindow +"/" +'train_result.csv', list(zip(id_train,labels_train)), pred_train)

        pred_test = knn.predict(features_test)    
        accuracy = accuracy_score(labels_test, pred_test)
        print("Test Accuracy: ", accuracy)
        generateTestCsv(typeWindow +"/" +'test_result.csv', list(zip(id_test,labels_test)), pred_test)


    