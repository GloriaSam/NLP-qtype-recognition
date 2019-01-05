import json
import subprocess
import ast
import re

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
#import nltk
#from nltk.corpus import stopwords

#nltk.download('averaged_perceptron_tagger')
#nltk.download('stopwords')

class Question_Formatter():

    def __init__(self):
        self.types = ['yesno', 'factoid','summary','list']
        self.types_q = ''
        self.cleaned_questions = ''
        self.cleaned_questions_q_dot = ''
        self.type_question_ft = ''

        with open("./include/BioASQ-trainingDataset6b.json") as f:
            self.data = json.load(f)

        self.clean_questions()
        self.create_type_question_ft()
        return

    def clean_questions(self, q_dot=False):
        """
        Skipping no alphanumeric characters.
        Removes blank spaces, before after the text.
        Changes each characters to lower case.
        Stemming all the sentence
        Puts a newline at the end of every question.
        """
        for q in self.data['questions']:
            temp = re.sub('[^ a-zA-Z0-9]', '', q['body'])
            temp = temp.strip().lower()
        #    temp = self.stemming(temp).strip()
            self.cleaned_questions_q_dot += temp + '?\n'
            self.cleaned_questions += temp + '\n'
            self.types_q += q['type'].strip() + '\n'


    def create_type_question_ft(self):
        """
        Creates string '__label__' + TYPE + ' ' + QUESTION for classification with fasttext
        """
        for i in range(0, len(self.cleaned_questions.splitlines())):
            self.type_question_ft += '__label__'+self.types_q.splitlines()[i] + ' ' + self.cleaned_questions.splitlines()[i] + '\n'

    def stemming(self, sentence):
        ps = PorterStemmer()
        words = word_tokenize(sentence)
        out = ''
        for word in words:
            out += ps.stem(word) + ' '
        return out


    def count_type(self):
        count_type = {}
        for t in self.types:
            count_type[t] = 0
        for t in self.types_q.splitlines():
            count_type[t] += 1
        print(count_type)

    def question_medtagger(self):
        """
        Question word tagging with medical words recognition.
        :return: parsed_questions:list of questions tagged, parsed_question_type:list of question tagged and relative type
        """
        self.save_on_file(self.cleaned_questions_q_dot, 'questions.txt')
        medpos_tag_out = self.medpos_tag('questions.txt')
        parsed_questions = self.parse(medpos_tag_out)
        parsed_questions_type = []
        parsed_questions_out = []
        for i in range(0, len(self.data['questions'])):
            # [ type, [ (w1, tag_w1) , (w2, tag_w2) ... (wn, tagwn) ] ]
            parsed_questions_type += [[self.data['questions'][i]['type']] + [parsed_questions[i]]]
            parsed_questions_out += [parsed_questions[i]]
        return parsed_questions_out, parsed_questions_type

    def verify_startwith_rule(self, data, type=None):
        """
        Information about starting word of each question
        :param data: questions
        :param type: specific question type to be verified
        """
        # questyon_type = [ ( question_body, question_type | 'other' ) ]
        # featuresets = [ ( first_word_of_the_question, question_type ) ]
        if type is not None:
            print('TYPE: '+ type)
        questions_type = []
        for q in data['questions']:
            questions_type.append([q['body']] + [(q['type']) if type in q['type'] else 'other'])
        featuresets = [(self.startwith_feature(q), t) for (q,t) in questions_type]
        perc = 0.5
        train_set, test_set = featuresets[:int(len(featuresets)*perc)], featuresets[int(len(featuresets)*perc):]
        classifier = nltk.NaiveBayesClassifier.train(train_set)
        print(nltk.classify.accuracy(classifier, test_set))
        classifier.show_most_informative_features(10)
        return

    def verify_startwith_rule2(self, type_tagged, type = None):
        """
        Information about starting word tag of each question
        :param data: list questions type and tag of the first word
        :param type: specific question type to be verified
        """
        featuresets = []
        for t, question_tag in type_tagged:
            if t in type:
                featuresets.append([self.startwith_feture2(question_tag),t])
            else:
                featuresets.append([self.startwith_feture2(question_tag),'other'])
        print(featuresets)
        perc = 0.5
        train_set, test_set = featuresets[:int(len(featuresets)*perc)], featuresets[int(len(featuresets)*perc):]
        classifier = nltk.NaiveBayesClassifier.train(train_set)
        print("Accurancy ")
        print(nltk.classify.accuracy(classifier, test_set))
        classifier.show_most_informative_features(10)

    def startwith_feture2(self, question_tagged):
        feature = {}
        feature["first_word_tag"] = question_tagged[0][1]
        return feature

    def startwith_feature(self, q):
        return  {'startwith:':q.split()[0]}

    def save_on_file(self, data, filename):
        with open(filename,'w') as out:
            out.write(data)
        return

    def medpos_tag(self, sourcefile):
        """
        Executes the external command for the medpost tagger
        :param sourcefile: source path of the medpost tagger
        """
        MedPostPath = './include/MedPost-SKR/'
        cmd = MedPostPath + 'run.sh ' + sourcefile + ' -Penn'
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        out = output.decode('utf-8').splitlines()
        # return the trimmed output for eliminate prolog info
        return out[2:(len(out)-5)]

    def preprocess(self, sent):
        """
        Nltk tagger
        :param sent: input to be tagged
        :return:
        """
        sent = nltk.word_tokenize(sent)
        sent = nltk.pos_tag(sent)
        return sent

    def parse(self, output):
        """
        Parses the medpost output, eliminating unwanted tuples
        :param output: medpost output
        :return: parsed medpost output
        """
        q = '['
        prepro = [line for line in output if "['''', '''']," not in line]
        prepro = [line for line in prepro if "[',', ',']," not in line]
        for line in prepro:
            process = line.strip()
            process = process.replace('[', '(')
            process = process.replace(']', ')')
            if '?' in process:
                q += '], ['
            else:
                q += process
        q += "('?','.') ]"
        # Evaluation of a string with a list format
        return ast.literal_eval(q)
