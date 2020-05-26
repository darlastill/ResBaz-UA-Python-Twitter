# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 12:00:24 2019

@author: Darla Still
"""
#   This code was written with the intent of instructing an interactive
# workshop. As such, I have done my best to comment this code for learners
# to better understand the process and what the code is doing. 

#   This code below is derived from several sources on scraping Twitter and
# and cleaning tweets, and is by no means perfect or the most efficient. 
# Here are some links for sources: 
# https://towardsdatascience.com/extracting-twitter-data-pre-processing-and-sentiment-analysis-using-python-3-0-7192bd8b47cf
# https://www.dropbox.com/s/5enwrz2ggswns56/Telemedicine_twitter_v3.5.py?dl=0
# https://medium.com/analytics-vidhya/basic-tweet-preprocessing-method-with-python-56b4e53854a1
# https://amueller.github.io/word_cloud/auto_examples/masked.html
# https://www.geeksforgeeks.org/python-program-to-count-words-in-a-sentence/
# 2019 Twitter scraping workshop at Computational Social Science Mini Conference at UA bv Emmi Bevensee

#STEP ZERO: SET YOUR WORKING DIRECTORY
import os
os.chdir("C:/Users/AsuS/Documents/ResBaz")
###############################################################################
###############################################################################
###############################################################################
                #PART ONE: SCRAPE TWITTER HANDLE(S)
###############################################################################
###############################################################################
###############################################################################
import tweepy
import csv

# load Twitter API credentials
# Yes, I know they are entered twice -- THIS COULD BE SIMPLIFIED. 
auth = tweepy.OAuthHandler(" ", " ")
auth.set_access_token(" ", " ")
api = tweepy.API(auth)

consumer_key = [' ']
consumer_secret = [' ']
access_key = [' ']
access_secret = [' ']

#change the screen name each time... 
#start selection here... lines 29-87, highlight and run. 
screen_name=" "

def get_all_tweets(screen_name):

# Twitter allows access to only 3200 tweets via this method

# Authorization and initialization

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

# initialization of a list to hold all Tweets

all_the_tweets = []

# We will get the tweets with multiple requests of 200 tweets each

new_tweets = api.user_timeline(screen_name=screen_name, tweet_mode = 'extended',count=200)

# saving the most recent tweets

all_the_tweets.extend(new_tweets)

# save id of 1 less than the oldest tweet

oldest_tweet = all_the_tweets[-1].id - 1

# grabbing tweets till none are left

while len(new_tweets):
# The max_id param will be used subsequently to prevent duplicates
    new_tweets = api.user_timeline(screen_name=screen_name,
    count=200, tweet_mode = 'extended',max_id=oldest_tweet)

# save most recent tweets

    all_the_tweets.extend(new_tweets)

# id is updated to oldest tweet - 1 to keep track

    oldest_tweet = all_the_tweets[-1].id - 1
    print ('...%s tweets have been downloaded so far' % len(all_the_tweets))

# transforming the tweets into a 2D array that will be used to populate the csv

outtweets = [[tweet.id_str, tweet.user, tweet.created_at, tweet.full_text, tweet.in_reply_to_screen_name, tweet.is_quote_status, tweet.source, tweet.lang, tweet.favorite_count, tweet.retweet_count, tweet.favorited, tweet.retweeted] for tweet in all_the_tweets]

# writing to the csv file

with open(screen_name + '_tweets.csv', 'w', encoding='utf8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'user', 'created_at', 'full_text', 'in_reply_to_screen_name', 'is_quote_status', 'source_device', 'lang', 'favorite_count', 'retweet_count', 'favorited', 'retweeted'])
    writer.writerows(outtweets)
    
    pass

if __name__ == '__main__':
    f.close()
#end selection here, and run lines 29-87
#DO THE ABOVE FOR AS MANY HANDLES AS YOU WISH, BUT LET'S STICK TO 3 FOR THE WORKSHOP


###############################################################################
###############################################################################
###############################################################################    
                #PART TWO: DATA CLEANING 
###############################################################################
###############################################################################
###############################################################################
#make one file, versus three files of tweets
import pandas as pd
import glob

extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
#combine all files in the list
sample = pd.concat([pd.read_csv(f) for f in all_filenames ], sort=False)
#export to csv
sample.to_csv("all_sample_tweets.csv", index=False, encoding='utf-8')


#GOT YOUR TWITTER DATA AND NEED TO COME BACK? NO PROBLEM, START HERE. 
#START HERE WHEN YOU RETURN TO THIS CODE... 

import os
import pandas as pd
os.chdir("C:/Users/AsuS/Documents/ResBaz")
sample = pd.read_csv("all_sample_tweets.csv")

import itertools
# you may be interested in hashtags or you may not be interested, but 
# for this exercise, we are removing hashtags and userhandles to de-identify 
# these data and make them more generalizable -- some companies use proprietary hashtags
# we could have gotten hashtags from scraping using the "entities" element 
# of our tweet object, but we still would have had to use some type of regex to parse

#this is a regular expression to search tweets and make a list of all hashtags in a given tweet
sample['hashtags']=sample['full_text'].str.findall(r"#(\w+)")
#here, we make this a list of hashtags, remove empty brackets, and drop duplicates
hashtags = sample.hashtags.tolist()
hashtagsClean = [x for x in hashtags if x !=[]]
hashtagsClean = list(itertools.chain.from_iterable(hashtagsClean))
hashtagsClean = list(dict.fromkeys(hashtagsClean))
                                       
#this is creating a binary variable for a tweet containing 1 or more hashtags
# this is useful for your descriptive analysis. 
sample['has_hashtag']=sample['full_text'].str.contains(r"#\S+\b",regex=True)
sample['has_hashtag']=sample.has_hashtag.astype(int)

#we are doing roughly the same thing with replies here, making a list, removing
#missing cases and duplicates
replies=[]
replies = sample.in_reply_to_screen_name.tolist()
repliesClean = [x for x in replies if x == x]
repliesClean = list(dict.fromkeys(repliesClean))

#here we are creating binary variables for tweets containing links and RTs
sample["rt"]=sample["full_text"].str.contains(r"RT")
sample["rt"]=sample.rt.astype(int)
sample["link"]=sample["full_text"].str.contains(r"http")
sample["link"]=sample.link.astype(int)


#here, we are making a list of company names
companies = ['benandjerrys', 'HaagenDazs_US', 'HaloTopCreamery', 'ben', 'jerry']
# while you are cleaning the tweets through this process, you might decide to 
# to add your own stop words, this is what I have done below... 
addwords = ['look']

#now we are going to use beautiful soup and nltk to preprocess/clean the tweets

import re
from bs4 import BeautifulSoup 
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer 

#this is some code to make your own stop words, and make all of the items
# we made into lists above lowercase - this will be important later. 
mystops = hashtagsClean + repliesClean + companies + addwords
mystops = [element.lower() for element in mystops]
mystop=set(mystops)

# this identifies what we are going to clean. 
tweet= sample['full_text']

# Happy Emoticons
emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
    ])

# Sad Emoticons
emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
    ])

#Emoji patterns
emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)

#combine sad and happy emoticons
emoticons = emoticons_happy.union(emoticons_sad)

#here we define a function to clean our tweets and make a new file. 

def cleanTweet(tweet):
    # Function to convert a raw tweet to a string of words
    # The input is a single string (a raw tweet), and 
    # the output is a single string (a preprocessed tweet)
    #
    # 1. Remove HTML
    tweet_text = BeautifulSoup(tweet, 'lxml').get_text() 
    #
    # 2. Remove non-letters
    letters_only = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', tweet_text)
    letters_only1 = re.sub(r'http\S+', '', letters_only)
    letters_only2 = re.sub(r'\@\S+\b', '', letters_only1)        
    letters_only3 = re.sub(r'[^a-zA-Z]|(\w+:\/\/\S+)', " ", letters_only2)
    letters_only4 = emoji_pattern.sub(r'', letters_only3)
    letters_only5 = re.sub(r'^\#\S+\b', '', letters_only4)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only5.lower().split()  
    words = [w for w in words if len(w)>2]                           
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english")).union(mystops)            
        
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    meaningful_words = [w for w in meaningful_words if not w in mystop]
    meaningful_words = [w for w in meaningful_words if not w in emoticons]
    #
    lmtzr = WordNetLemmatizer()
    lemmatized_words = [lmtzr.lemmatize(w) for w in meaningful_words]
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( lemmatized_words )) 

num_tweets = sample['full_text'].size
#num of rows in comments column
clean_tweets=[]
for i in range(0, num_tweets):
    if((i+1)%1000 == 0):
        print('tweet %d of %d\n' % ( i + 1, num_tweets))
    clean_tweets.append(cleanTweet(sample['full_text'][i]))

#this makes a text file of your clean tweets, used later in word cloud. 
with open('tweets_text.txt', 'w') as f:
    for item in clean_tweets:
        
        f.write("%s\n" % item)


###############################################################################
###############################################################################
###############################################################################
    # PART THREE: SOME BASIC DESCRIPTIVE ANALYSIS OF YOUR TWEETS (CORPUS)
###############################################################################
###############################################################################
###############################################################################
    
#here we are going to run a little script that tells us the top words
    #you can set the number of top words, and save the output. 
import collections
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from os import path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

from wordcloud import WordCloud, STOPWORDS

file = open('tweets_text.txt', encoding="utf8")
a= file.read()
# Instantiate a dictionary, and for every word in the file, 
# Add to the dictionary if it doesn't exist. If it does, increase the count.
wordcount = {}
# To eliminate duplicates, remember to split by punctuation, and use case demiliters.
stops = set(stopwords.words('english')).union(mystops)
for word in a.lower().split():
    if word not in stops:
        if word not in wordcount:
            wordcount[word] = 1
        else:
            wordcount[word] += 1
# Print most common word
n_print = int(input("How many most common words to print: "))
print("\nOK. The {} most common words are as follows\n".format(n_print))
word_counter = collections.Counter(wordcount)
for word, count in word_counter.most_common(n_print):
    print(word, ": ", count)
# Close the file
file.close()

#adjust above and save top words below
d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()
# Create a data frame of the most common words 
# Draw a bar chart
lst = word_counter.most_common(n_print)
df = pd.DataFrame(lst, columns = ['Word', 'Count'])
df.plot.bar(x='Word',y='Count')
plt.savefig('sampletop25words.png')
plt.show()
df.to_csv("sample_top25words_freq-word.csv", encoding='utf-8')

###############################################################################
###############################################################################
###############################################################################
                #PART FOUR: WORD CLOUDS
###############################################################################
###############################################################################
###############################################################################

#NOW IT IS TIME TO MAKE A WORD CLOUD. 
#WORD CLOUD #1 IS A COOL/FANCY WORD CLOUD USING A MASK IMAGE... 
# get data directory (using getcwd() is needed to support running example in generated IPython notebook)
d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()

# Read the whole text.
text = open(path.join(d, 'tweets_text.txt')).read()

# read the mask image

cone_mask = np.array(Image.open(path.join(d, "cone.png")))

stopwords = set(mystops)

wc = WordCloud(background_color="white", max_words=2000, mask=cone_mask,
               stopwords=stopwords, contour_width=3, contour_color='steelblue')

# generate word cloud
wc.generate(text)

# store to file
wc.to_file(path.join(d, "icecreamtweets.png"))

# show
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.figure()
plt.imshow(cone_mask, cmap=plt.cm.gray, interpolation='bilinear')
plt.axis("off")
plt.show()

#WORLD CLOUD #2 IS PLAIN, BASIC, AND PRINTABLE FOR THESES/DISSERTATIONS
#CREATE A PLAIN WORD CLOUD THAT IS PRINTABLE FOR YOUR COMMITTEE

# Read the whole text.
text = open(path.join(d, 'tweets_text.txt')).read()

stopwords = set(STOPWORDS).union(mystops)

# Generate a word cloud image

import matplotlib.pyplot as plt
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")

def grey_color_func(word, font_size, position,orientation,random_state=None, **kwargs):
    return("hsl(230,100%%, %d%%)" % np.random.randint(49,51))

#create the wordcloud object
wc = WordCloud(width=800, height=400, background_color="white", max_words=2000, collocations=False,
               stopwords=stopwords, contour_width=3, contour_color='steelblue',max_font_size = 50).generate(text)

#change the color setting
#wc.recolor(color_func = grey_color_func).generate(text)

default_colors = wc.to_array()
plt.title("Custom colors")
plt.imshow(wc.recolor(color_func=grey_color_func, random_state=3),
           interpolation="bilinear")
wc.to_file("icecream_plain_wc.png")
plt.axis("off")
plt.figure()
plt.title("Default colors")
plt.imshow(default_colors, interpolation="bilinear")
plt.axis("off")
plt.show()
