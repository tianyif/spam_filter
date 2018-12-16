from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

def text_to_words(raw_text):
	# Remove HTML
	text = BeautifulSoup(raw_text).get_text()
	# Remove non-letters
	letters_only = re.sub("[^a-zA-Z]", " ", text)
	# Convert to lower case, split into individual words
	words = letters_only.lower().split()
	# Convert the stop words to a set
	stops = set(stopwords.words("english"))
	# Remove stop words
	# meaningful_words = words
	meaningful_words = [w for w in words if not w in stops]
	# Join the words back into one string separated by space
	return " ".join(meaningful_words)



