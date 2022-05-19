import itertools
from collections import defaultdict
from gensim.corpora.dictionary import Dictionary
from NLP.src.nlp_utils import get_pre_process_wiki_articles

# Create a Dictionary from the articles: dictionary
articles = get_pre_process_wiki_articles()
dictionary = Dictionary(articles)

# Create a MmCorpus: corpus
corpus = [dictionary.doc2bow(article) for article in articles]

# Get the fifth document in corpus: doc
doc = corpus[4]

# Sort the doc for frequency: bow_doc
bow_doc = sorted(doc, key=lambda w: w[1], reverse=True)

# Print the top 5 words of the document alongside the count
for word_id, word_count in bow_doc[:5]:
    print("The token ",dictionary.get(word_id),"appears ",word_count , "times")

# Create the defaultdict: total_word_count
total_word_count = defaultdict(int)
for word_id, word_count in itertools.chain.from_iterable(corpus):
    total_word_count[word_id] += word_count

# Choose a key between 0 and 10 and show the count with a print function.
key = 7
print("the key", key,"in defaultdict has count: ", total_word_count[key],'\n')

# Create a sorted list from the defaultdict: sorted_word_count
sorted_word_count = sorted(total_word_count.items(), key=lambda w: w[1], reverse=True)

# Print the top 5 words across all documents alongside the count
print(sorted_word_count[0])
for idx, qtd in sorted_word_count[:5]:
    print(dictionary.get(idx), qtd)