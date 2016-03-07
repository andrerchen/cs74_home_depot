from gensim import corpora, models, similarities
from pprintpp import pprint
from collections import defaultdict
import csv

documents = []

# SET THE CORPUS FILE - WHAT YOU'RE COMPARING SEARCH TERM TO
filename = "data/brands_test.txt"

# read each line from input file
with open(filename) as f:
	documents = f.readlines()

# strip new lines
documents = [doc.strip('\n') for doc in documents]

# remove common words and tokenize
stoplist = set('for a of the and to in #N/A'.split())
texts = [[word for word in document.lower().split() if not word in stoplist]
		 for document in documents]

# remove all words with one occurrence
#frequency = defaultdict(int)
#for text in texts:
#    for token in text:
#        frequency[token] += 1

#texts = [[token for token in text if frequency[token] > 1]
#		 for text in texts]

#pprint(texts)

corpus_dictionary = corpora.Dictionary(texts)
#product_descriptions_dictionary.save('data/product_descriptions_dictionary.dict') # store dictionary

print "dictionary compiled, now creating corpus"

corpus = [corpus_dictionary.doc2bow(text) for text in texts]
#print(corpus)

tfidf = models.TfidfModel(corpus)

corpus_tfidf = tfidf[corpus]

print "transforming corpus to LSI space and indexing..."

lsi = models.LsiModel(corpus_tfidf, id2word=corpus_dictionary)
index = similarities.MatrixSimilarity(lsi[corpus]) # transform corpus to LSI space and index it

# SET THE INPUT FILE - SHOULD BE SEARCH TERMS
input_file = "data/search_terms_test.txt"

with open(input_file) as f:
	query_terms = f.readlines()

# strip new lines
query_terms = [query_term.strip('\n') for query_term in query_terms]

search_term_sims = []

print "querying search terms... please wait"

for i in range(len(query_terms)):

	if i % 1000 == 0:
		print "query number " + str(i)

	# vectorize query
	query_term = query_terms[i]
	query_bow = corpus_dictionary.doc2bow(query_term.lower().split())

	query_lsi = lsi[query_bow] # convert query to LSI space

	# compute similarity vector
	sims = index[query_lsi]
	sims_list = list(enumerate(sims))

	# grab just the similarity between the search term and its
	# corresponding product description
	search_term_sims.append(sims_list[i])

print "similarity computation complete... now writing"

# SET THE OUTPUT FILE
output_file = "data/search_term_to_brand_sims_test_tfidf.csv"

with open(output_file, 'w') as fp:
	writer = csv.writer(fp, delimiter=',')
	writer.writerow(["search term", "similarity to brand"])
	writer.writerows(search_term_sims)







# new_doc = "whirlpool microwave"
# vec_bow = product_descriptions_dictionary.doc2bow(new_doc.lower().split())
# vec_lsi = lsi[vec_bow] # convert query to LSI space
# print(vec_bow)
# print(vec_lsi)


# sims = index[vec_lsi]
# print(list(enumerate(sims)))
