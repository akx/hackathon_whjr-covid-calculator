import pickle
from flask import Flask

app = Flask(__name__)

# open a file, where you stored the pickled data
with open('model.pkl', 'rb') as file:
    clf = pickle.load(file)


@app.route('/')
def hello_world():
    # Code For Infrence
    inputFeatures = [100, 1, 26, 1, 0]
    infProb = clf.predict_proba([inputFeatures])[0][1]
    return repr({'input': inputFeatures, 'output': infProb})


if __name__ == "__main__":
    app.run(debug=True)
