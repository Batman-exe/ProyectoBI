from nltk.tokenize import TweetTokenizer


tt = TweetTokenizer()

def tokenizer(text):
   return tt.tokenize(text)