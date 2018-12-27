from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pandas as pd

import fasttext
from question_formatter import Question_Formatter
from fasttext_estimator import FasttextEstimator
from random import shuffle

class Question_Classifier:

    def __init__(self):
        self.question = []

    def svm_classifier(self):
        bankdata = pd.read_csv("./include/Questions.csv")
        print(bankdata.head())
        X = bankdata.drop('Type', axis=1)
        y = bankdata['Type']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        svclassifier = SVC(kernel='linear')
        svclassifier.fit(X_train, y_train)
        y_pred = svclassifier.predict(X_test)
        print(y_pred)

    def fasttext_text_representation(self):
        model = fasttext.skipgram('./include/data.txt', 'model')
        print(model.words)  # list of words in dictionary

    def fit(self, data, question_type):
        """
        Fits the classificator and creates the model
        :param data: string of all question divided in lines
        :param question_type: string of all type and question in lines ('__label__'+TYPE+' '+QUESTION)
        """
        shuffle(question_type)
        perc = 0.8
        data_train = question_type[:int(len(question_type)*perc)]
        data_test = question_type[int(len(question_type)*perc):]
        with open('./include/data.txt','w') as f:
            f.write(data)
        with open('./include/data.train.txt','w') as f:
            f.write('\n'.join(data_train))
        with open('./include/test.txt','w') as f:
            f.write('\n'.join(data_test))
        classifier = fasttext.supervised('./include/data.train.txt', 'model', lr=0.5, dim=200)
        result = classifier.test('./include/test.txt')
        print('P@1:', result.precision)
        print('R@1:', result.recall)
        print('Number of examples:', result.nexamples)


classifier = Question_Classifier()
question_formatter = Question_Formatter()
#data = question_formatter.cleaned_questions
#question_type = question_formatter.type_question.splitlines()
#classifier.fit(data, question_type)

# Cross Validation fasttext classificator
estimator = FasttextEstimator()
texts = question_formatter.cleaned_questions.splitlines()
labels = question_formatter.types_q.splitlines()
scores = cross_val_score(estimator, texts, labels, cv=StratifiedKFold(n_splits=4), verbose=4)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
