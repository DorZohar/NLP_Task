import multiprocessing
from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.word2vec import Word2Vec
from gensim.models import TfidfModel

# logging is important to get the state of the functions
import logging

if __name__ == '__main__':
    multiprocessing.freeze_support()

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)

    #wiki = WikiCorpus('C:\\Wiki\\enwiki-latest-pages-articles.xml.bz2', lemmatize=False)
    #tfidf = TfidfModel(wiki)
    # save for persistence
    #wiki.save('C:\\Wiki\\wiki.corpus')
    #tfidf.save('C:\\Wiki\\wiki.tfidf.model')

    wiki = WikiCorpus.load('C:\\Wiki\\wiki.corpus')
    tfidf = TfidfModel(wiki)
    tfidf.load('C:\\Wiki\\wiki.tfidf.model')

    # word2vec


    class MySentences(object):
        def __iter__(self):
            for text in wiki.get_texts():
                yield [word.decode() for word in text]


    sentences = MySentences()
    params = {'size': 300, 'window': 10, 'min_count': 40,
              'workers': max(1, multiprocessing.cpu_count() - 1), 'sample': 1e-3, }
    word2vec = Word2Vec(sentences, **params)
    word2vec.save('C:\\Wiki\\wiki.word2vec.model')
