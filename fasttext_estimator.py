import fasttext
from sklearn.metrics import f1_score
from sklearn.base import BaseEstimator

class FasttextEstimator(BaseEstimator):

    def __init__(self):
        self.model = None

    def fit(self, feature, labels):
        save_on_file('./include/train.txt', feature, labels)
        fasttext.supervised('./include/train.txt', 'model', lr=0.5, dim=200)
        self.model = fasttext.load_model('model.bin', encoding='utf-8')
        return self

    def score(self, features, labels):
        predicted_labels = []
        predictions = self.model.predict(features)
        for prediction in predictions:
            predicted_label = prediction[0].replace('__label__',"")
            predicted_labels.append(predicted_label)
        return f1_score(labels, predicted_labels, average="macro")

def save_on_file(filename, features, labels):
    with open(filename,'w') as out:
        for i in range(0, len(features)):
            out.write("__label__%s %s\n" % (labels[i],features[i]))
