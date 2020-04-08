from gensim.models import KeyedVectors
from gensim.models.doc2vec import LabeledSentence
import keras
from keras.models import Sequential
from keras.layers import Dense
from nltk.tokenize import TweetTokenizer
import numpy
import pandas
pandas.options.mode.chained_assignment = None
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

from config import config

link_pattern = re.compile('<a href[^<]+</a>')
n_dims = 200
label_key = config['mongo']['data']['label_key']
embedding_save_path = config['embedding_save_path']
model_save_path = config['model_save_path']

class ModelProvider():
    
    def __init__(self):
        self._embedding_vectors = None
        self.model = None

    def train_model(labeled_jobs):
        embedding_keyed_vectors = self.get_embedding_vectors()
        df = pandas.DataFrame(labeled_jobs)
        processed_labeled = self.process(labeled_jobs)
        x_train, x_test, y_train, y_test = train_test_split(numpy.array(processed_labeled['tokens']), numpy.array(processed_labeled[label_key]), test_size=0.2)
        x_train = self.labelize_jobs(x_train, 'TRAIN')
        x_test = self.labelize_jobs(x_test, 'TEST')

        vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
        matrix = vectorizer.fit_transform([ x.words for x in x_train ]) # Assuming we want the larger vocabulary size, unfortunately won't catch everything new
        tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

        train_vecs = numpy.concatenate([ self.build_word_vector(tokens, n_dims, embedding_keyed_vectors) for tokens in tqdm(map(lambda x: x.words, x_train)) ])
        train_vecs = scale(train_vecs)

        test_vecs = numpy.concatenate([ self.build_word_vector(tokens, n_dims, embedding_keyed_vectors) for tokens in tqdm(map(lambda x: x.words, x_test)) ])
        test_vecs = scale(test_vecs)

        model = Sequential()
        model.add(Dense(32, activation='relu', input_dim=n_dims))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

        model.fit(train_vecs, y_train, epochs=9, batch_size=32, verbose=2)
        score = model.evaluate(test_vecs, y_test, batch_size=30, verbose=2)
        print(f'Model score: {score[1]}')

        model.save(model_save_path, save_format='tf')
        self.model = keras.models.load_model(model_save_path)

        return score

    def predict(text):
        # TODO: tokenize, build word vectors, and scale
        pass

    # My kingdom for lazy vals and options!!
    def get_embedding_vectors():
        if self._embedding_vectors == None:
            try:
                self._embedding_vectors = KeyedVectors.load(embedding_kv_file)
        
        return self._embedding_vectors
    

    def tokenize(job):
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

    def process(data):
        data.drop(['_id', 'by', 'id', 'parent', 'date'], axis=1, inplace=True)
        data[label_key] = data[label_key].map(lambda x: 1 if x else 0)
        data.drop('index', axis=1, inplace=True)
        data['tokens'] = data['text'].progress_map(self.tokenize)
        data = data[data.tokens != 'NC']
        data.reset_index(inplace=True)
        data.drop('index', axis=1, inplace=True)
        return data

    # TODO: can we simplify this some?
    def labelize_jobs(jobs, label_type):
        labelized = []
        for i, v in tqdm(enumerate(jobs)):
            label = '%s_%s'%(label_type, i)
            labelized.append(LabeledSentence(v, [label]))
        return labelized

    def build_word_vector(tokens, size, word_vectors):
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

