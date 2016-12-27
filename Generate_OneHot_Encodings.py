import pickle
import gensim

w2v_model = gensim.models.Word2Vec.load("Data/Dim300_window5_mincount20")
word_to_index = {}
index_to_word = {}

for k in w2v_model.vocab.keys():
    word_to_index[k] = w2v_model.vocab[k].index
    index_to_word[w2v_model.vocab[k].index] = k

pickle.dump(word_to_index, open("./Data/WordToIndex.pickle", 'ab+'))
pickle.dump(index_to_word, open("./Data/IndexToWord.pickle", 'ab+'))
