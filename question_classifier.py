from sklearn import svm, preprocessing, naive_bayes
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import pandas as pd

import fasttext
from question_formatter import Question_Formatter
from fasttext_estimator import FasttextEstimator
from random import shuffle
from scipy.sparse import coo_matrix, hstack

class Question_Classifier:
    def __init__(self):
        self.question = []
        self.pos_code_map={'CC':'A','CD':'B','DT':'C','EX':'D','FW':'E','IN':'F','JJ':'G','JJR':'H','JJS':'I','LS':'J','MD':'K','NN':'L','NNS':'M',
'NNP':'N','NNPS':'O','PDT':'P','POS':'Q','PRP':'R','PRP$':'S','RB':'T','RBR':'U','RBS':'V','RP':'W','SYM':'X','TO':'Y','UH':'Z',
'VB':'1','VBD':'2','VBG':'3','VBN':'4','VBP':'5','VBZ':'6','WDT':'7','WP':'8','WP$':'9','WRB':'@'}

    def convert(self, tag):
        try:
            code = self.pos_code_map[tag]
        except:
            code = '?'
        return code

    def classify(self):
        Corpus = pd.read_csv("./include/Questions.csv")
        qf = Question_Formatter()
        questions_tagged, _ = qf.question_medtagger()
        pos_tags_list = []
        for question_tagged in questions_tagged:
            pos_tags = ''
            for word in question_tagged:
                pos_tags += self.convert(word[1])
            pos_tags_list.append(pos_tags)
        Corpus['Pos_tags'] = pos_tags_list

        Train_X, Test_X, Train_Y, Test_Y = train_test_split(Corpus.loc[:,['Question','Pos_tags']], Corpus['Type'],test_size=0.2)

        Encoder = preprocessing.LabelEncoder()
        Train_Y = Encoder.fit_transform(Train_Y)
        Test_Y = Encoder.fit_transform(Test_Y)

        Tfidf_vect = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2')
        Tfidf_vect.fit(Corpus['Question'])
        Train_X_Tfidf = Tfidf_vect.transform(Train_X['Question'])
        Test_X_Tfidf = Tfidf_vect.transform(Test_X['Question'])

        Train_X_pos = Tfidf_vect.transform(Train_X['Pos_tags'])
        Test_X_pos = Tfidf_vect.transform(Test_X['Pos_tags'])

        self.naive_bayes_classifier(Train_X_Tfidf,Train_Y,Test_X_Tfidf,Test_Y)
        self.svm_classifier(Train_X_Tfidf,Train_Y,Test_X_Tfidf,Test_Y)


    def naive_bayes_classifier(self, Train_X,Train_Y, Test_X,Test_Y):
        # fit the training dataset on the NB classifier
        Naive = naive_bayes.MultinomialNB()
        Naive.fit(Train_X, Train_Y)
        # predict the labels on validation dataset
        predictions_NB = Naive.predict(Test_X)
        # Use accuracy_score function to get the accuracy
        print("\nNaive Bayes Classifier")
        print("Accuracy Score -> %0.2f" % (accuracy_score(predictions_NB, Test_Y) * 100))

    def svm_classifier(self, Train_X, Train_Y, Test_X, Test_Y):
        # Classifier - Algorithm - SVM
        # fit the training dataset on the classifier
        SVM = svm.SVC(C=1.0, kernel='linear')
        SVM.fit(Train_X, Train_Y)
        # predict the labels on validation dataset
        predictions_SVM = SVM.predict(Test_X)
        # Use accuracy_score function to get the accuracy
        print("\nSVM Classifier")
        print("Accuracy Score -> %0.2f" % (accuracy_score(predictions_SVM, Test_Y) * 100))


    def fasttext_text_representation(self):
        model = fasttext.skipgram('./include/data.txt', 'model')
        print(model.words)  # list of words in dictionary

    def fasttext_calssifier(self, data, question_type):
        """
        Fits the classificator and creates the model
        :param data: string of all question divided in lines
        :param question_type: string of all type and question in lines ('__label__'+TYPE+' '+QUESTION)
        """
        shuffle(question_type)
        perc = 0.7
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
        print('FastText Classifier')
        print('Precision: %0.2f' % result.precision)
        print('Recall: %0.2f' % result.recall)
        print('Number of examples:', result.nexamples)

    def ft_cross_validation(self, estimator, texts, labels):
        scores = cross_val_score(estimator, texts, labels, cv=StratifiedKFold(n_splits=5))
        print('\nFastText CrossValidation')
        print('Scores ->', scores)
        print("Accuracy Avarage Score -> %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def main():
    qc = Question_Classifier()
    qf = Question_Formatter()
    data = qf.cleaned_questions
    question_type = qf.type_question_ft.splitlines()
    qc.fasttext_calssifier(data, question_type)

    # Cross Validation fasttext classificator
    estimator = FasttextEstimator()
    texts = qf.cleaned_questions.splitlines()
    labels = qf.types_q.splitlines()
    qc.ft_cross_validation(estimator,texts, labels)
    #qc.fasttext_text_representation()

    qc.classify()

if __name__ == '__main__':
    main()
