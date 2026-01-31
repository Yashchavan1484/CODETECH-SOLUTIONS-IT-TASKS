import re #pattern matching(removing digits,special characters!!)
import string 
from bs4 import BeautifulSoup #removes html tags
from nltk.tokenize import word_tokenize, sent_tokenize #splits text into sentences and then words!
from nltk.corpus import stopwords #removes common words like is,the,and etc
import nltk #natural language processing (NLP) toolkit


nltk.download('punkt') #for sentence and word tokenization.
nltk.download('stopwords') #consist of English stopword list.

# Input text (can be a long article)
text = input("Enter the text you want to summarize:- ") #takes input from the user.

# ---------- TEXT CLEANING ----------
def clean_text(text):
    text = text.lower() #convert each letter in the text to lowercase
    text = re.sub(r'\d+', '', text) 
    text = BeautifulSoup(text, "html.parser").get_text() #remove numbers if any!
    text = text.translate(str.maketrans('', '', string.punctuation)) #remove html tags 
    text = re.sub(r'\W+', ' ', text) #removes punctuation [,./!] etc.
    return text #finally returns the text

cleaned_text = clean_text(text) 

# ---------- SENTENCE TOKENIZATION ----------
sentences = sent_tokenize(text) #breaks the input paragraph into sentences with using all the cleaning required.

# ---------- WORD TOKENIZATION ----------
words = word_tokenize(cleaned_text) #breaks the sentences into individual words 

# ---------- STOPWORD REMOVAL ----------
stop_words = set(stopwords.words('english')) #As mentioned above is used for removal of stop words like is,and,the which are meaningless for understanding the paragraph context.
filtered_words = [word for word in words if word not in stop_words]

# ---------- WORD FREQUENCY ----------
word_freq = {} #keeps count of repeated words (eg. python->3) it means python word is there in the paragraph and is repeatedly used 3 times.
for word in filtered_words:
    word_freq[word] = word_freq.get(word, 0) + 1

# ---------- SENTENCE SCORING ----------
sentence_scores = {} #it stores scores if sentence consists of important words (more important words->higher the score).
for sentence in sentences:
    sentence_lower = sentence.lower() #lowercase
    for word in word_freq:
        if word in sentence_lower:
            sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_freq[word]

# ---------- SUMMARY GENERATION ----------
summary_length = max(1, int(len(sentences) * 0.4))  #selects 40 % of the total sentences 
summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:summary_length] 

summary = [s for s in sentences if s in summary_sentences]
#maintains origanl order (important for readability of the paragraph.)


print("SUMMARY:\n")
print(" ".join(summary)) #finally displays the summarized text. 
