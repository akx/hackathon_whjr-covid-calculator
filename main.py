import pickle
from flask import Flask
app = Flask(__name__)

# open a file, where you stored the pickled data
file = open('model.pkl', 'rb')
clf = pickle.load(file)
file.close()

@app.route('/')
def hello_world():
    # Code For Infrence
    inputFeatures = [100,1,26,1,0]
    infProb = clf.predict_proba([inputFeatures])[0][1]
    return 'Hello, World!' + str(infProb)

if __name__ == "__main__":
    app.run(debug=True)
