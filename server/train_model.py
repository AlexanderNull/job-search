from gensim.models import KeyedVectors
from gensim.models.doc2vec import LabeledSentence
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from nltk.tokenize import TweetTokenizer
import numpy
import pandas
pandas.options.mode.chained_assignment = None
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

from server.config import config

link_pattern = re.compile('<a href[^<]+</a>')
n_dims = 200
label_key = config['mongo']['data']['label_key']
embedding_save_path = config['embedding_save_path']
model_save_path = config['model_save_path']
scaler_save_path = config['scaler_save_path']

class ModelProvider():
    
    def __init__(self):
        self._embedding_vectors = None
        self.model = None

    def train_model(self, labeled_jobs):
        embedding_keyed_vectors = self.get_embedding_vectors()
        df = pandas.DataFrame(labeled_jobs)
        processed_labeled = self.process(df)
        x_train, x_test, y_train, y_test = train_test_split(numpy.array(processed_labeled['tokens']), numpy.array(processed_labeled[label_key]), test_size=0.2)
        x_train = self.labelize_jobs(x_train, 'TRAIN')
        x_test = self.labelize_jobs(x_test, 'TEST')

        tfidf = self.build_tfdif(x_train, 10)

        train_vecs = self.build_vector_list(x_train, embedding_keyed_vectors, tfidf)
        test_vecs = self.build_vector_list(x_test, embedding_keyed_vectors, tfidf)
        scaler = StandardScaler().fit(train_vecs)

        train_vecs = scaler.transform(train_vecs)
        test_vecs = scaler.transform(test_vecs)

        model = Sequential()
        model.add(Dense(32, activation='relu', input_dim=n_dims))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

        model.fit(train_vecs, y_train, epochs=9, batch_size=32, verbose=2)
        score = model.evaluate(test_vecs, y_test, batch_size=30, verbose=2)
        print(f'Model score: {score[1]}')

        # TODO: use config to reject models below certain accuracy threshold
        model.save(model_save_path, save_format='tf')
        pickle.dump(scaler, open(scaler_save_path, 'wb'))
        self.model = keras.models.load_model(model_save_path)

        return float(score[1]) # Numpy float values are not serializable by flask

    def predict(self, text):
        embedding_keyed_vectors = self.get_embedding_vectors()
        new_test = numpy.array([self.tokenize(text),])
        new_test = self.labelize_jobs(new_test, 'PREDICT')

        tfidf = self.build_tfdif(new_test, 1)
        new_vecs = self.build_vector_list(new_test, embedding_keyed_vectors, tfidf)

        scaler = pickle.load(open(scaler_save_path, 'rb'))
        model = keras.models.load_model(model_save_path)

        prediction = int(model.predict_classes(scaler.transform(new_vecs))[0][0])

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

    def build_vector_list(self, labeled_tokens, embedding_keyed_vectors, tfidf):
        return numpy.concatenate([ self.build_word_vector(tokens, tfidf, n_dims, embedding_keyed_vectors) for tokens in tqdm(map(lambda x: x.words, labeled_tokens)) ])

    def build_word_vector(self, tokens, tfidf, size, word_vectors):
        vec = numpy.zeros(size).reshape((1, size))
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
