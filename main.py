from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import PredefinedSplit
from pandas import read_csv
from numpy import ravel
import json


data1 = read_csv('data/hotspotUrbanMobility-1.csv')
data2 = read_csv('data/hotspotUrbanMobility-2.csv')
data = data1.append(data2, ignore_index=True)
data = data.drop('h24', axis=1)
data = data.loc[data['Anomalous'] < 1]
data.to_csv('data/preprocessedDataset.csv', index=False)
data = read_csv ('data/preprocessedDataset.csv' )

#change the data split into 70%-30%
training_data, testing_data, training_labels, testing_labels = train_test_split(data.iloc[:, 4:len(data.columns)], data.iloc[:, 1], test_size=0.3, train_size=0.7)

mlp = MLPClassifier(random_state=0)

#set all the parameters of slide 63
parameters = {'max_iter': (100, 200, 300), 'hidden_layer_sizes': (100, 110, 120)}

gs = GridSearchCV(mlp, parameters)

#try PredefinedSplit
gs.fit(training_data, ravel(training_labels))
config_path = 'config/modelConfiguration.json'

with open(config_path, 'w') as f:
    json.dump(gs.best_params_, f)
with open(config_path, "r") as f:
    params = json.load(f)

model = MLPClassifier(**params).fit(training_data, ravel(training_labels))
labels = model.predict(testing_data)
score = accuracy_score(ravel(testing_labels), labels)
print(score)