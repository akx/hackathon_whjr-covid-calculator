import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

if __name__ == "__main__":
    # Read data
    df = pd.read_csv('data.csv')

    X = df[['fever', 'bodyPain', 'age', 'runnyNose', 'diffBreath']].to_numpy()
    y = df[['infProb']].to_numpy()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Train model
    clf = LogisticRegression()
    clf.fit(X_train, y_train.ravel())

    # Test model
    print('Score:', clf.score(X_test, y_test.ravel()))

    # Save model
    with open('model.pkl', 'wb') as f:
        pickle.dump(clf, f)
