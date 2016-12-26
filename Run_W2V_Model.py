import gensim
from datetime import datetime

start = datetime.now()
sentences = gensim.models.word2vec.LineSentence("sentence_per_line.txt")
print("Time to load in sentences: ", str(datetime.now() - start))
model = gensim.models.Word2Vec(sentences, size=300, window=5,
                               min_count=5, workers=4)
print("Time to build model: ", str(datetime.now() - start))
model.save("Dim300_window5_mincount5")
print("Total time: ", str(datetime.now() - start))
