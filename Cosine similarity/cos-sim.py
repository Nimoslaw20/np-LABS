import numpy as np
from sklearn.feature_extraction.text import CountVectorizer



    filename1 = "amazon_cells_labelled.txt"
    filename2 = "amazon_cells_labelled.txt"
    filename3 = "yelp_labelled.txt"
    df1 = pd.read_csv(filename1, sep="\t", names=["docs", "class"]) 
    df2 = pd.read_csv(filename2, sep="\t", names=["docs", "class"])
    df3 = pd.read_csv(filename3, sep="\t", names=["docs", "class"])
    
def cos_sim(a, b):
	"""Takes 2 vectors a, b and returns the cosine similarity according 
	to the definition of the dot product
	"""
	dot_product = np.dot(a, b)
	norm_a = np.linalg.norm(a)
	norm_b = np.linalg.norm(b)
	return dot_product / (norm_a * norm_b)

count_vect = CountVectorizer()
counts = count_vect.fit_transform(dataset)

vectorized = counts.toarray()

sentence1 = np.array(vectorized[0]) 
sentence2 = np.array(vectorized[1])
sentence3 = np.array(vectorized[2])
sentence4 = np.array(vectorized[3])
sentence5 = np.array(vectorized[4])
sentence6 = np.array(vectorized[5])


arrays = [sentence1, sentence2, sentence3, sentence4, sentence5, sentence6]
output = []
for i in range(len(arrays)):
    inner = []
    for j in range(len(arrays)):
        inner.append(cos_sim(arrays[i], arrays[j]))
    output.append(inner)

print(np.array(output), file=open('ouput.txt', 'a'))