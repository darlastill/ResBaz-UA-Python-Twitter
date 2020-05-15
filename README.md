# Twitter Scraping and Basic Descriptive Analysis in Python

This repository was created for UArizona's Research Bazaar workshop on Scraping Twitter in Python. This code was written with the intention of providing an interactive workshop using Python in Spyder. 

## Getting Started

See the .pdf for Getting Started for Scraping Twitter specifically for this workshop. 

### Prerequisites
*Applying for a Twitter Developer Account*
* Apply for and create a [Twitter developer account](https://developer.twitter.com/en)
* Once your account is approved, create an [app](https://developer.twitter.com/en/apps) by logging in with the account above and following the instructions. 
* Here is more information on [Twitter apps](https://developer.twitter.com/en/docs/basics/apps/overview). 
* Once your app is created, locate your individual Consumer API Keys and Access tokens. 

*What you need in Python*

`pip install tweepy`
`pip install wordcloud`

*We are using glob to concatenate the csv files, and I have an older version. You may need to install this if you've never used it before:*

`pip install glob2`

### Installing

* Open your command line (Command Prompt or Powershell for PC users). 
* Type the above commands and press enter. 

`pip install tweepy`

* This can also be done in Spyder. 

`!pip install tweepy`

## Resources on Regular Expressions

We will be using some regular expressions to parse tweets. Here are some quick guides and resources. 

* [Test Your Regular Expressions Here](https://regex101.com/)
* [Quick Tutorial for Regular Expressions in Python](https://pythonprogramming.net/regular-expressions-regex-tutorial-python-3/)

## Built With and Versioning

This code and workshop was written using Python 3.7.4 with the [Anaconda distribution](https://www.anaconda.com/products/individual). 

## Authors

* **Darla Still** 

## Acknowledgments and Sources

* Base code borrowed from the 2019 Twitter scraping workshop at Computational Social Science Mini Conference at UArizona by **Emmi Bevensee**
* Links for additional resources and inspiration for this code: 
* [Scraping Twitter - Towards Data Science](https://towardsdatascience.com/extracting-twitter-data-pre-processing-and-sentiment-analysis-using-python-3-0-7192bd8b47cf)
* [Code for above link](https://www.dropbox.com/s/5enwrz2ggswns56/Telemedicine_twitter_v3.5.py?dl=0)
* [Basic Tweet Preprocessing](https://medium.com/analytics-vidhya/basic-tweet-preprocessing-method-with-python-56b4e53854a1)
* [How to create a Mask Word Cloud](https://amueller.github.io/word_cloud/auto_examples/masked.html)
* [Simple code to count words](https://www.geeksforgeeks.org/python-program-to-count-words-in-a-sentence/)


