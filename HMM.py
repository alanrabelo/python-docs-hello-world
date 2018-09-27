from Enums import Label
import itertools
import numpy as np
import os
import pickle

class Loader:

    @staticmethod
    def load_file(filename):
        APP_ROOT = os.path.dirname(os.path.abspath(__file__))  # refers to application_top
        APP_STATIC = os.path.join(APP_ROOT, 'static')
        fh = open(os.path.join(APP_STATIC, filename), 'r')

        lines = []

        for line in fh.readlines():
            lines.append(line)
        fh.close()
        return lines

class Formatter:

    @staticmethod
    def format_label(label_to_format):
        if label_to_format.startswith('PREP'):
            label_to_format = 'PREP'
        if label_to_format.startswith('PRO') or label_to_format.startswith('PDEN'):
            label_to_format = 'PRO'
        if label_to_format.startswith('K'):
            label_to_format = 'K'
        if label_to_format.startswith('NUM'):
            label_to_format = 'NUM'
        elif label_to_format.startswith('N'):
            label_to_format = 'N'
        if label_to_format.startswith('ADV'):
            label_to_format = 'ADV'

        return label_to_format

    @staticmethod
    def remove_punctuation(label_to_remove):
        return str(label_to_remove).lower().replace('.', ' .').replace(',', ' ,').replace('?', ' ?').replace('!', ' !').split(' ')

class HMM:

    def __init__(self):
        self.labels = set()
        self.tags = []
        self.sentences = []
        self.words = dict()
        self.bigrams = dict()
        self.bigramsSaw = set()
        self.bigramsProbabilities = dict()
        self.total = 0
        self.loadTrainningData('macmorpho-train.txt')

    def loadTrainningData(self, filename):

        try:
            self.words, self.bigrams, self.bigramsProbabilities, self.bigramsSaw, self.possibleLabels, self.labels = pickle.load(open("model.pkl", "rb"))
            return
        except (OSError, IOError) as e:
            print(e)

        self.generatePossibleLabels()

        print('Carregando tudo de novo')

        lines = Loader.load_file(filename)

        for line in lines:
            sentenceSplitted = line.replace('\n', '').split(' ')
            for index, wordFromSentence in enumerate(sentenceSplitted):

                # Gets word and label
                word, label = wordFromSentence.split('_')
                word = str(word).lower()
                previousLabel = 'EMPTY' if index == 0 else sentenceSplitted[index - 1].split('_')[1]

                # formating labels to simplify classification
                label = Formatter.format_label(label)
                previousLabel = Formatter.format_label(previousLabel)

                self.bigramsSaw.add((previousLabel, label))
                self.labels.add(str(label))

                if label in self.words:
                    probabilitiesDict = self.words[label]
                    if word in probabilitiesDict:
                        probabilitiesDict[word] += 1
                    else:
                        probabilitiesDict[word] = 1
                    self.words[label]['TOTALABEL'] += 1
                else:
                    self.words[label] = {word: 1, 'TOTALABEL': 1}

                if previousLabel in self.bigrams:
                    if label in self.bigrams[previousLabel]:
                        self.bigrams[previousLabel][label] += 1
                    else:
                        self.bigrams[previousLabel][label] = 1
                    self.bigrams[previousLabel]['total'] += 1
                else:
                    self.bigrams[previousLabel] = {label: 1, 'total': 1}

        for previous in self.bigrams.keys():

            totalForPrevious = self.bigrams[previous]['total']

            del self.bigrams[previous]['total']

            for current in self.bigrams[previous]:
                self.bigrams[previous][current] /= totalForPrevious

        for label in self.words.keys():
            totalForPrevious = self.words[label]['TOTALABEL']

            for word in self.words[label]:
                self.words[label][word] /= totalForPrevious

        with open('model.pkl', 'wb') as model:  # Python 3: open(..., 'wb')
            pickle.dump([self.words, self.bigrams, self.bigramsProbabilities, self.bigramsProbabilities, self.possibleLabels, self.labels], model)

    def classify(self, sentence, numberOfArranges: int):

        if sentence == '':
            return None

        print('Started Classification Task')

        splittedSentence = Formatter.remove_punctuation(sentence)
        splittedSentence = list(filter(None, splittedSentence))

        possibleLabelsBest = []

        for word in splittedSentence:

            if word.endswith(' ') or word.startswith(' '):
                word.replace(' ', '')

            labelsForWord = []

            for label in self.possibleLabels:
                if word in self.words[label.value[0]]:
                    if self.words[label.value[0]][word] > 0:
                        labelsForWord.append(label.value[0])

            if len(labelsForWord) == 0:
                possibleLabelsBest.append(self.generatePossibleLabelsAsString())
            else:
                possibleLabelsBest.append(labelsForWord)

        print(possibleLabelsBest)
        possibleArranges = self.possibleArrangesForArray(possibleLabelsBest)

        probabilityOfArrange = []

        viterbyDict = dict()

        for index, arrange in enumerate(possibleArranges):
            produtOfProbabilites = 1
            for index, word in enumerate(splittedSentence):
                if index >= len(arrange):
                    continue
                currentLabel = arrange[index]
                previousLabel = 'EMPTY' if index == 0 else arrange[index - 1]

                # Verify dictionary for previous occurrencies
                tupleForCurrentiteration = (previousLabel, currentLabel, word)
                if tupleForCurrentiteration in viterbyDict:
                    produtOfProbabilites *= viterbyDict[tupleForCurrentiteration]
                    continue

                if word not in self.words[currentLabel]:
                    probabilityOfWord = 0.00000000000001
                else:
                    probabilityOfWord = self.words[currentLabel][word]

                if previousLabel not in self.bigrams or currentLabel not in self.bigrams[
                    'EMPTY' if index == 0 else arrange[index - 1]]:
                    probabilityOfBigram = 0.000000000000001
                else:
                    probabilityOfBigram = self.bigrams['EMPTY' if index == 0 else previousLabel][currentLabel]

                viterbyDict[(previousLabel, currentLabel, word)] = probabilityOfWord * probabilityOfBigram

                produtOfProbabilites *= probabilityOfWord * probabilityOfBigram

            probabilityOfArrange.append(produtOfProbabilites)

        maxProbability = np.argmax(probabilityOfArrange)

        return possibleArranges[maxProbability]

    def possibleArranges(self, count: int):
        # Gets all combinations of labels with repetition
        allPossible = list([p for p in itertools.product(self.possibleLabels, repeat=count)])
        return allPossible

    def possibleArrangesForArray(self, array: [str]):
        # Gets all combinations of labels with repetition
        allPossible = list(itertools.product(*array))
        print('We must search in ' + str(len(allPossible)) + ' combinations')

        return allPossible

    def generatePossibleLabels(self):
        self.possibleLabels = list(set(map(Label, Label)))

    def generatePossibleLabelsAsString(self):

        labels = []
        for label in list(set(map(Label, Label))):
            labels.append(label.value[0])
        return labels
