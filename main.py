# main.py
from fastapi import FastAPI,status
from geotext import GeoText
from pydantic import BaseModel
# from fastapi import Response
from typing import List
import re
# import nltk
# import csv
import tweepy
# import ssl
# from geotext import GeoText
import pandas as pd
from nltk.corpus import stopwords

stop_words = stopwords.words('english')

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

#sentiment 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



# spacy for lemmatization
import spacy
# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import pandas as pd


app = FastAPI()

@app.get("/")

def hello():
    return {"message":"Welcom To NLP APP"}

# Create  Base Model
class GraphList(BaseModel):
    data: List


class remove_dictionary(BaseModel):
    InputText: str
    
    dictionary: list 

class analyzeText(BaseModel):
    InputText: str
   
    username: str 

#models for the declarying datatype for find mention in the text
class findmention(BaseModel):
    InputText: str


class findhas(BaseModel):
    InputText: str
     




# function to print all the hashtags in a text
def extract_hashtags(text):
     
    # initializing hashtag_list variable
    hashtag_list = []
     
    # splitting the text into words
    for word in text.split():
         
        # checking the first character of every word
        if word[0] == '#':
             
            # adding the word to the hashtag_list
            hashtag_list.append(word[1:])
     
    return hashtag_list


 
# function to print sentiments
# of the sentence.
def sentiment_scores(sentence):
 
    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()
 
    # polarity_scores method of SentimentIntensityAnalyzer
    # object gives a sentiment dictionary.
    # which contains pos, neg, neu, and compound scores.
    sentiment_dict = sid_obj.polarity_scores(sentence)
     
    #return sentiment_dict
 
    # decide sentiment as positive, negative and neutral
    if sentiment_dict['compound'] >= 0.05 :
       return "Positive"
 
    elif sentiment_dict['compound'] <= - 0.05 :
        return "Negative"
 
    else :
        return "Neutral"






def analyze_semilarity(tex,name):
    
    # Oauth keys
    consumer_key = "KjRHMWQUgSSAWejC50GurFN4H"
    consumer_secret = "EKtcL0ZHlBXJ2ICOlsoS1aeqnPHDWEtuDzA8vIGhayzhUt2JYC"
    access_token = "1531766139251990528-qaiaa5DlG1uOkKMsbxnfHmvEMaC1jj"
    access_token_secret = "oBDsdt6Gw42BBWB02I3YgP12pBHiz7tr1uYXIACE94S7a"

    userID = name

    # Authorize our Twitter credentials
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    tweets = api.user_timeline(screen_name=userID, 
                           # 200 is the maximum allowed count
                            count=200,
                           #include_rts = False,
                           # Necessary to keep full_text 
                           # otherwise only the first 140 words are extracted
                           tweet_mode = 'extended'
                           )
    
    #
   



    previousTweets=[]
    for info in tweets:
        previousTweets.append(info.full_text)
    
    #create new df 
    
    
    df = pd.DataFrame({'text':previousTweets})
    
    #preprocessing start

    # Convert to list
    data = previousTweets
    
    # Remove Emails
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

    # Remove new line characters
    data = [re.sub('\s+', ' ', sent) for sent in data]

        # Remove distracting single quotes
    data = [re.sub("\'", "", sent) for sent in data]

    # remove URLS from the text
    data = [re.sub('http://\S+|https://\S+', '', sent) for sent in data]

    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

    data_words = list(sent_to_words(data))
    
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # Define functions for stopwords, bigrams, trigrams and lemmatization
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)
    
    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    
    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    
    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized
    
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    
    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=10, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
    keywordstopic=lda_model.show_topics()
    
    topic=[]
    for topicNum,topicWords in keywordstopic:
        topics = re.findall("[a-zA-Z]+",keywordstopic[topicNum][1])
   
    txt=tex
    # Remove Emails
    txt = re.sub('\S*@\S*\s?', '', txt) 
   
    # Remove new line characters
    txt = re.sub('\s+', ' ', txt) 

    # Remove distracting single quotes
    txt = re.sub("\'", "", txt) 

    # remove URLS from the text
    txt = re.sub('http://\S+|https://\S+', '', txt) 
    filtered_sentence = []
    spaces = r"\s+"
    txt= (re.split(spaces, txt))
    for w in txt:
        if w not in stop_words:
            filtered_sentence.append(w)
    
    
    
    
    counter=0
    for token in filtered_sentence:
        if token in topics :
            counter+=1
            print(token) 
    if counter>=round(len(topic)/10):
        return True
        
    return False  
        
def remove_Words(tex,dic):
    tex=tex.lower()
    for i in range(len(dic)):
        # Iterate through string_list

        dic[i] = dic[i].lower()
    wordlist = tex.split()
    newtext = " ".join(x for x in wordlist if x not in dic)

    # text = tex.replace(dic, "")
    return newtext


#function to extract hastags
def extract_mention(tex):
    result = re.findall("(^|[^@\w])@(\w{1,15})", tex)
    print(result)
    return result

#function to extract hastags
def findLocation(tex):
     
    # extracting entities.
    
    places = GeoText(tex)
    
    

    location = {
        "countries": places.countries,# getting all countries
       
        "cities": places.cities # getting all cities
    }
    return location
    



@app.post("/analyzeText", status_code=status.HTTP_201_CREATED)
def privacy_Managment(data: analyzeText ):
    return analyze_semilarity(data.InputText,data.username)

@app.post("/removeDictionaryWords", status_code=status.HTTP_201_CREATED)
def privacy_Managment(data: remove_dictionary):
    return remove_Words(data.InputText,data.dictionary)


#api for find mentions in the text
@app.post("/findmention", status_code=status.HTTP_201_CREATED)
def privacy_Managment(User: findmention):
    return extract_mention(User.InputText)

#api for find hastags in the text
@app.post("/findhastag", status_code=status.HTTP_201_CREATED)
def hastag(User: findhas):
    return extract_hashtags(User.InputText)



#api for find sentiment in the text
@app.post("/findsentiment", status_code=status.HTTP_201_CREATED)
def sentiment(User: findhas):
    return sentiment_scores(User.InputText)


#api for find Location in the text
@app.post("/findlocation", status_code=status.HTTP_201_CREATED)
def locationAPI(User: findhas):
    return findLocation(User.InputText)
