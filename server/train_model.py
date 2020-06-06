# from functools import reduce
from gensim.models import KeyedVectors
from gensim.models.doc2vec import LabeledSentence
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from nltk.tokenize import TweetTokenizer
import numpy
import pandas
pandas.options.mode.chained_assignment = None
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

from server.config import config

link_pattern = re.compile('<a href[^<]+</a>')
label_key = config['mongo']['data']['label_key']
embedding_save_path = config['embedding_save_path']
model_save_path = config['model_save_path']
max_seq_length = config['max_sequence_length']
should_under_sample = config['should_under_sample']
under_sample_ratio = config['under_sample_ratio']

class ModelProvider():

    n_dims = 200
    
    def __init__(self):
        self._embedding_vectors = None
        self.model = None

    # Turns out I'm kinda picky, this is causing data to be severely skewed and
    # model is just taking the easy way out and ignoring all 1s
    def fix_skew(self, frame):
        if should_under_sample:
            minority = len(frame[frame[label_key] == 1])
            return frame.drop(frame[frame[label_key] == 0].sample(len(frame) - minority * under_sample_ratio).index)
        else:
            return frame

    def train_model(self, labeled_jobs):
        embedding_keyed_vectors = self.get_embedding_vectors()
        df = pandas.DataFrame(labeled_jobs)
        processed_labeled = self.process(df)
        processed_labeled = self.fix_skew(processed_labeled)
        x_train, x_test, y_train, y_test = train_test_split(numpy.array(processed_labeled['tokens']), numpy.array(processed_labeled[label_key]), test_size=0.2)
        x_train = self.labelize_jobs(x_train, 'TRAIN')
        x_test = self.labelize_jobs(x_test, 'TEST')

        tfidf = self.build_tfdif(x_train, 10)

        train_vecs = self.build_vector_list(x_train, embedding_keyed_vectors, tfidf, max_seq_length)
        test_vecs = self.build_vector_list(x_test, embedding_keyed_vectors, tfidf, max_seq_length)
        y_train = y_train.astype(float)
        y_test = y_test.astype(float)

        model = Sequential()
        model.add(BatchNormalization())
        model.add(LSTM(5, input_shape=train_vecs.shape[1:]))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(32, activation='relu', input_dim=self.n_dims))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        optimizer = Adam(learning_rate = 0.001)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        model.fit(train_vecs, y_train, epochs=40, batch_size=5, verbose=2)
        print(model.summary())
        score = model.evaluate(test_vecs, y_test, batch_size=5, verbose=2)
        print(f'Model score: {score[1]}')

        # TODO: use config to reject models below certain accuracy threshold
        model.save(model_save_path, save_format='tf')
        self.model = keras.models.load_model(model_save_path)

        return float(score[1]) # Numpy float values are not serializable by flask

    def predict(self, text):
        embedding_keyed_vectors = self.get_embedding_vectors()
        new_test = numpy.array([self.tokenize(text),])
        new_test = self.labelize_jobs(new_test, 'PREDICT')

        # TODO: loads should be done less often and have better cleanup. Stop loading for every call.
        tfidf = pickle.load(open('server/tfidf.pickle', 'rb'))
        new_vecs = self.build_vector_list(new_test, embedding_keyed_vectors, tfidf, max_seq_length) # todo: limit to length

        model = keras.models.load_model(model_save_path)

        prediction = int(model.predict_classes(new_vecs)[0])

        return prediction

    # My kingdom for lazy vals and options!!
    def get_embedding_vectors(self):
        if self._embedding_vectors == None:
            self._embedding_vectors = KeyedVectors.load(embedding_save_path)
        
        return self._embedding_vectors
    

    def tokenize(self, job):
        job = job.lower()
        job = link_pattern.sub('', job)
        job = (
            job.replace('<p>', '')
            .replace('&#x27;', "'")
            .replace('&quot;', '"')
            .replace('|', '')
        )
        tokenizer = TweetTokenizer()
        tokens = tokenizer.tokenize(job)
        tokens = list(filter(lambda t: not t.startswith('@'), tokens))
        tokens = list(filter(lambda t: not t.startswith('#'), tokens))
        return tokens

    def process(self, data):
        data.drop(['_id', 'by', 'id', 'parent', 'date'], axis=1, inplace=True)
        data = data[data['text'].isnull() == False]
        data = data[data['preferred'].isnull() == False]
        data[label_key] = data[label_key].map(lambda x: 1 if x else 0)
        data['tokens'] = data['text'].progress_map(self.tokenize)
        data = data[data.tokens != 'NC']
        data.reset_index(inplace=True)
        data.drop('index', axis=1, inplace=True)
        return data

    # TODO: can we simplify this some?
    def labelize_jobs(self, jobs, label_type):
        labelized = []
        for i, v in tqdm(enumerate(jobs)):
            label = '%s_%s'%(label_type, i)
            labelized.append(LabeledSentence(v, [label]))
        return labelized

    def build_vector_list(self, labeled_tokens, embedding_keyed_vectors, tfidf, sequence_length):
        vectors = numpy.zeros((len(labeled_tokens), sequence_length, self.n_dims))
        for i, tokens in tqdm(enumerate(labeled_tokens)):
            for j, token in enumerate(tokens.words):
                vectors[i, j] = self.individual_word_vector(token, tfidf, self.n_dims, embedding_keyed_vectors)

        return vectors

    def individual_word_vector(self, token, tfidf, vec_size, word_vectors):
        try:
            return word_vectors[token].reshape((vec_size)) * tfidf[token]
        except KeyError:
            return numpy.zeros((vec_size))

    def build_word_vector(self, tokens, tfidf, size, word_vectors):
        vec = numpy.zeros((1, size))
        count = 0
        for word in tokens:
            try:
                vec += word_vectors[word].reshape((1, size)) * tfidf[word]
                count += 1
            except KeyError:
                continue

        if count != 0:
            vec /= count

        return vec

    def build_tfdif(self, labeled_tokens, min_freq):
        vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=min_freq)
        vectorizer.fit_transform([ x.words for x in labeled_tokens ])
        tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

        return tfidf
