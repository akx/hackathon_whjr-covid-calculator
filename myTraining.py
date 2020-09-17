import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle 

if __name__ == "__main__":
    df = pd.read_csv('data.csv')

def data_split(data,ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data)* ratio )
    test_indeces = shuffled[:test_set_size]
    train_indeces = shuffled[test_set_size:]
    return data.iloc[train_indeces], data.iloc[test_indeces]

if __name__ == "__main__":
    # Read The Data
    df = pd.read_csv('data.csv')
    train ,test =  data_split(df, 0.2)
    
    X_train = train[[' fever' , ' bodyPain' , ' age' , ' runnyNose' , ' diffBreath' ]].to_numpy()
    X_test = test[[' fever' , ' bodyPain' , ' age' , ' runnyNose' , ' diffBreath' ]].to_numpy()
    
    Y_train = train[['infectionProb' ]].to_numpy().reshape(800,)
    Y_test = test[['infectionProb' ]].to_numpy().reshape(199,)

    # open a file, where you ant to store the data
    file = open('model.pkl', 'wb')
    # dump information to that file
    pickle.dump(clf, file)
    clf = LogisticRegression()
    clf.fit(X_train,Y_train)
    file.close()

    
