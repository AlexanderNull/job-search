from pymongo import MongoClient
import pandas
pandas.options.mode.chained_assignment = None
import numpy
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import LabeledSentence

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

from sklearn.model_selection import train_test_split

from server.config import config
import server.db_tools as db_tools
from server.job_provider import JobProvider
from server.train_model import ModelProvider

# TODO: honestly might as well singleton this somewhere
client = MongoClient(config['mongo']['host'], config['mongo']['port'])
db_name = config['mongo']['embedding_data']['db_name']
table_name = config['mongo']['embedding_data']['table_name']

embedding_db = client[db_name]
embedding_table = embedding_db[table_name]
embedding_save_path = config['embedding_save_path']

def train_embedding():
    model = ModelProvider()
    data = ingest_embedding(model.tokenize)

    x_train, x_test = train_test_split(numpy.array(data['tokens']), test_size=0.2)
    x_train = model.labelize_jobs(x_train, 'TRAIN')
    x_test = model.labelize_jobs(x_test, 'TEST')
    
    w2v = Word2Vec(size=model.n_dims, min_count=10)
    w2v.build_vocab([ x.words for x in tqdm(x_train) ])
    w2v.train(sentences=[ x.words for x in tqdm(x_train) ], total_words=w2v.corpus_total_words, epochs=w2v.epochs)

    w2v.wv.save(embedding_save_path)

def ingest_embedding(tokenizer):

    embedding_db = client[db_name]
    embedding_table = embedding_db[table_name]

    data = pandas.DataFrame(embedding_table.find())
    data.drop_duplicates(subset='id', inplace=True)
    data.drop(['_id', 'by', 'id', 'parent', 'date', 'preferred'], axis=1, inplace=True)
    data = data[data['text'].isnull() == False]
    data['tokens'] = data['text'].progress_map(tokenizer)
    data = data[data.tokens != 'NC']
    data.reset_index(inplace=True)
    data.drop('index', axis=1, inplace=True)
    return data

if __name__ == '__main__':
    train_embedding()