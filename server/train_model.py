# from functools import reduce
from gensim.models import KeyedVectors
from gensim.models.doc2vec import LabeledSentence
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, LSTM
from tensorflow.keras.optimizers.schedules import ExponentialDecay
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
from server.synthesize import synthesize

LINK_PATTERN = re.compile('<a href[^<]+</a>')
NOISE_PATTERN = re.compile('^\W$')
LABEL_KEY = config['mongo']['data']['label_key']
EMBEDDING_SAVE_PATH = config['embedding_save_path']
TRAINED_EMBEDDING_PATH = config['trained_embedding_path']
MODEL_SAVE_PATH = config['model_save_path']
SHOULD_UNDER_SAMPLE = config['model']['should_under_sample']
UNDER_SAMPLE_RATIO = config['model']['under_sample_ratio']
END_OF_SENTENCE = config['end_of_sentence']

class ModelProvider():    
    def __init__(self):
        self._embedding_vectors = None
        self.model = None
        self.tfidf = None

        # This one takes a while, calls waiting on it will time out so load at start, custom model might not exist so lazy load
        if config['use_google_embedding']:
            self.embedding_vectors = KeyedVectors.load_word2vec_format(TRAINED_EMBEDDING_PATH, binary=True)
            self.n_dims = 300

    # Turns out I'm kinda picky, this is causing data to be severely skewed and
    # model is just taking the easy way out and ignoring all 1s
    def fix_skew(self, frame):
        if SHOULD_UNDER_SAMPLE:
            minority = frame[frame[LABEL_KEY] == 1]
            synthetic = pandas.Series(synthesize(minority['tokens']))
            synthetic = pandas.DataFrame(synthetic, columns=['tokens'])
            synthetic[LABEL_KEY] = 1
            under_sampled = frame.drop(frame[frame[LABEL_KEY] == 0].sample(len(frame) - (2 * len(minority)) - len(synthetic)).index)
            return under_sampled.append(synthetic, ignore_index=True)
        else:
            return frame

    def train_model(self, labeled_jobs, params = {}):
        min_sequence_length = params.get('min_sequence_length', config['model']['min_sequence_length'])
        max_sequence_length = params.get('max_sequence_length', config['model']['max_sequence_length'])
        learning_rate = params.get('learning_rate', config['model']['learning_rate'])
        epochs = params.get('epochs', config['model']['epochs'])
        batch_size = params.get('batch_size', config['model']['batch_size'])
        lstm_dims = params.get('lstm_dims', config['model']['lstm_dims'])
        include_relu_layer = params.get('include_relu_layer', config['model']['include_relu_layer'])
        relu_dims = params.get('relu_dims', config['model']['relu_dims'])
            
        embedding_keyed_vectors = self.get_embedding_vectors()
        df = pandas.DataFrame(labeled_jobs)
        processed_labeled = self.process(df, params.get('min_sequence_length', config['model']['min_sequence_length']))
        processed_labeled = self.fix_skew(processed_labeled)
        x_train, x_test, y_train, y_test = train_test_split(numpy.array(processed_labeled['tokens']), numpy.array(processed_labeled[LABEL_KEY]), test_size=0.2)
        x_train = self.labelize_jobs(x_train, 'TRAIN')
        x_test = self.labelize_jobs(x_test, 'TEST')

        tfidf = self.build_tfdif(x_train, 10)

        train_vecs = self.build_vector_list(x_train, embedding_keyed_vectors, tfidf, max_sequence_length)
        test_vecs = self.build_vector_list(x_test, embedding_keyed_vectors, tfidf, max_sequence_length)
        y_train = y_train.astype(float)
        y_test = y_test.astype(float)

        model = Sequential()
        model.add(BatchNormalization())
        model.add(LSTM(lstm_dims, input_shape=train_vecs.shape[1:]))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        lr_scheduler = ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=10000,
            decay_rate=0.9)
        optimizer = Adam(learning_rate = lr_scheduler)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        model.fit(train_vecs, y_train, epochs=epochs, batch_size=batch_size, verbose=2)
        print(model.summary())
        score = model.evaluate(test_vecs, y_test, batch_size=batch_size, verbose=2)
        print(f'Model score: {score[1]}')

        # TODO: use config to reject models below certain accuracy threshold
        model.save(MODEL_SAVE_PATH, save_format='tf')
        self.model = keras.models.load_model(MODEL_SAVE_PATH)

        return float(score[1]), int(max_sequence_length) # Numpy float values are not serializable by flask

    def predict(self, text, max_sequence_length):
        embedding_keyed_vectors = self.get_embedding_vectors()
        new_test = numpy.array([self.tokenize(text),])
        new_test = self.labelize_jobs(new_test, 'PREDICT')
        tfidf = self.get_tfidf()
        new_vecs = self.build_vector_list(new_test, embedding_keyed_vectors, tfidf, max_sequence_length)

        model = keras.models.load_model(MODEL_SAVE_PATH)

        prediction = int(model.predict_classes(new_vecs)[0])

        return prediction

    def predict_bulk(self, jobs, max_sequence_length):
        embedding_keyed_vectors = self.get_embedding_vectors()
        new_test = numpy.array([ self.tokenize(job['text']) for job in jobs ])
        new_test = self.labelize_jobs(new_test, 'PREDICT')
        tfidf = self.get_tfidf()
        new_vecs = self.build_vector_list(new_test, embedding_keyed_vectors, tfidf, max_sequence_length)

        model = keras.models.load_model(MODEL_SAVE_PATH)
        
        predictions = model.predict_classes(new_vecs)
        
        return [ { LABEL_KEY: int(p), 'id': jobs[i]['id'] } for i, p in enumerate(predictions) ]

    # My kingdom for lazy vals and options!!
    def get_embedding_vectors(self):
        if self.embedding_vectors == None:
            if config['use_google_embedding']:
                self.embedding_vectors = KeyedVectors.load_word2vec_format(TRAINED_EMBEDDING_PATH, binary=True)
                self.n_dims = 300
            else:
                self.embedding_vectors = KeyedVectors.load(EMBEDDING_SAVE_PATH)
                self.n_dims = 200
        
        return self.embedding_vectors

    def get_tfidf(self):
        if self.tfidf == None:
            self.tfidf = pickle.load(open(config['tfidf_save_path'], 'rb'))
        
        return self.tfidf

    def tokenize(self, job):
        job = job.lower()
        job = LINK_PATTERN.sub('', job)
        job = (
            job.replace('<p>', '')
            .replace('&#x27;', "'")
            .replace('&quot;', '"')
        )
        tokenizer = TweetTokenizer()
        tokens = tokenizer.tokenize(job)
        tokens = [ t if t != '.' else END_OF_SENTENCE for t in tokens ] # For synthetic data shuffling
        tokens = [ t for t in tokens if self.should_keep_token(t) ]
        return tokens

    def should_keep_token(self, token):
        return (
            re.match(NOISE_PATTERN, token) == None and
            not token.startswith('@') and
            not token.startswith('#')
        )

    def process(self, data, min_length):
        data.drop(['_id', 'by', 'id', 'parent', 'date'], axis=1, inplace=True)
        data = data[data['text'].isnull() == False]
        data = data[data['preferred'].isnull() == False]
        data[LABEL_KEY] = data[LABEL_KEY].map(lambda x: 1 if x else 0)
        data['tokens'] = data['text'].progress_map(self.tokenize)
        data = data[data.tokens != 'NC']
        data = data[data.tokens.apply(lambda x: len(x) > min_length)]
        data.drop(['text'], axis=1, inplace=True)
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
            words = tokens.words
            for j in range(min(sequence_length, len(words))):
                vectors[i, j] = self.individual_word_vector(words[j], tfidf, self.n_dims, embedding_keyed_vectors)

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
