import numpy as np
from scipy.io import wavfile
import os
from pathlib import Path, PurePath
import csv
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def stft(x, window, num_threads):
    window_size = len(window)
    hop_size = int(window_size // 1.25)#%20 overlap
    N = len(x)
    num_frames = (N - window_size) // hop_size + 1
    stft_matrix = np.zeros((window_size, num_frames), dtype=np.complex64)
    target_length = getPowerLength(window_size)
    
    def process_frame(i):
        start = i * hop_size
        end = start + window_size
        frame = x[start:end]
        windowed_frame = frame * window
        apply_zero_padding(windowed_frame, target_length)
        return fft(windowed_frame)
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_frame, i) for i in range(num_frames)]
        for i, future in tqdm(enumerate(futures), leave=False, position=2, total=num_frames):
            stft_matrix[:, i] = future.result()
    
    return stft_matrix

# def stft(x, window_size, hop_size):
#     N = len(x)
#     window = np.hamming(window_size)
    
#     num_frames = (N - window_size) // hop_size + 1
    
#     stft_matrix = np.zeros((window_size, num_frames), dtype=np.complex64)
#     targetLength = getTargetLenght(window_size)
    
    
#     for i in tqdm(range(num_frames),leave=False,position=2):
#         start = i * hop_size
#         end = start + window_size
#         frame = x[start:end]

#         windowed_frame = frame * window

#         apply_zero_padding(windowed_frame, targetLength)

#         stft_matrix[:, i] = fft(windowed_frame)
    
#     return stft_matrix


def fft(x):
    N = len(x)
    
    if N <= 1:
        return x
    
    even = fft(x[0::2])
    odd = fft(x[1::2])
    
    factor = np.exp(-2j * np.pi * np.arange(N) / N)
    
    return np.concatenate([even + np.multiply(factor[:N // 2], odd),
                           even + np.multiply(factor[:N // 2], odd)])

def apply_zero_padding(signal, target_length):
    current_length = len(signal)
    if current_length >= target_length:
        return signal[:target_length]
    else:
        padding_length = target_length - current_length
        padding = np.zeros(padding_length)
        return np.concatenate((signal, padding))


def getPowerLength(x):
    tmp = 1
    while tmp < x:
        tmp *= 2
    return tmp
        

def getPower(fft_result):#get frequencies power
    return np.sum(np.abs(fft_result) ** 2)

def getMean(fft_result):#get amplitude mean
    return np.mean(np.abs(fft_result))

def weightedAverage(magnitudes):#get weighted frequency average
    weighted_sum = np.sum(np.arange(len(magnitudes)) * magnitudes)
    
    sum_magnitudes = np.sum(magnitudes)
    
    weighted_average = weighted_sum / sum_magnitudes
    
    return weighted_average


def extract_features(stft_result):
    frequency_power, amplitude_mean, weighted_frequency = extractFFTFeatures(stft_result)

    frequency_power_features = extractSTFTFeatures(frequency_power)
    amplitude_mean_features = extractSTFTFeatures(amplitude_mean)
    weighted_frequency_features = extractSTFTFeatures(weighted_frequency)

    return [frequency_power_features, amplitude_mean_features, weighted_frequency_features]

def extractFFTFeatures(stft_result):
    frequency_power = []
    amplitude_mean = []
    weighted_frequency = []
    for fft_result in stft_result:
        frequency_power.append(getPower(fft_result))
        amplitude_mean.append(getMean(fft_result))
        weighted_frequency.append(weightedAverage(frequency_power))

    return frequency_power, amplitude_mean, weighted_frequency
        
def extractSTFTFeatures(feature):
    mean_feature = np.mean(feature)
    median_feature = np.median(feature)
    deviation_feature = np.std(feature, ddof=1)
    return [mean_feature, median_feature, deviation_feature]

def extract_number_and_combine(original_string):
    song_name = original_string.split("/")[-1]
    number_string = song_name.split(".", 1)[-1].split(".")[0]
    try:
        number = int(number_string)
        label = song_name.split(".")[0]
        combined_string = label + "_" + str(number)
        return combined_string, label
    except ValueError:
        return None

def fillCsv(window_num,identifier, features):
    with open('features'+ str(window_num) +'.csv', 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        for i in range(len(identifier)):
            writer.writerow([identifier[i][0], 
                             features[i][0][0], 
                             features[i][0][1], 
                             features[i][0][2], 
                             features[i][1][0],
                             features[i][1][1],
                             features[i][1][2], 
                             features[i][2][0], 
                             features[i][2][1], 
                             features[i][2][2], 
                             identifier[i][1]])
        csvfile.close()    

def featureExtractor(window,sound_data, filename):
    stft_result = stft(sound_data, window, num_threads=1)
    features = extract_features(stft_result)
    identifier = extract_number_and_combine(filename)
    return identifier, features

def initializeCsv(windowNum):
    filename = 'features' + str(windowNum) + '.csv'
    if os.path.exists(filename):
        print(f"Warning: file {filename} already exists and will be overwritten.")
    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow(['id', 'frequency_power_mean', 'frequency_power_median', 'frequency_power_deviation', 'amplitude_mean_mean', 'amplitude_mean_median', 'amplitude_mean_deviation', 'weighted_frequency_mean', 'weighted_frequency_median', 'weighted_frequency_deviation', 'label'])
    except IOError:
        print(f"Error: could not open file {filename} for writing.")
    else:
        print(f"Successfully created file {filename}.")


    
if __name__ == '__main__':

    data_path = PurePath('Data/genres_original/')
    music_genres = ['disco', 'jazz', 'blues', 'metal', 'reggae']
    windows = [np.hamming(2048), np.hanning(2048), np.blackman(2048)]
    window_num = 0
    for window in windows:
        initializeCsv(window_num)
        count = 0
        for genre in tqdm(music_genres,desc='Genre'):
            identifier = []
            all_features = []
            
            new_data_path = PurePath(data_path, genre)
            for filename in tqdm(os.listdir(new_data_path),desc='File',leave=False,position=1):
                file_path = os.path.join(new_data_path, filename)
                sample_rate, sound_data = wavfile.read(file_path)
                tmp_identifier, tmp_all_features = featureExtractor(window,sound_data,file_path)
                identifier.append(tmp_identifier)
                all_features.append(tmp_all_features)
        
            fillCsv(window_num,identifier, all_features)
        window_num += 1
    

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

        k = 5
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
