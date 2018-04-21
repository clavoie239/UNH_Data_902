###############################################################################################
################################## TWITTER SENTIMENT ANALYSIS #################################
###############################################################################################

#This code uses 2000 Tweets with the hashtag #Obamacare from April 11-15, 2018. 
#Sentiment Analysis is performed through visualization of wordclouds, a commonality cloud,
#a comparison cloud, and emotional analysis chart.
#
#Author: Caroline Lavoie
#Date: April 21, 2018
#Email: cl1152@wildcats.unh.edu
#
#
##############################################################################################
##############################################################################################
##############################################################################################

# load libraries
library(tm)
library(qdap)
library(tibble)
library(ggplot2)
library(RWeka)
library(wordcloud)
library(lubridate)
library(lexicon)
library(tidytext)
library(lubridate)
library(gutenbergr)
library(stringr)
library(dplyr)
library(radarchart)
library(readtext)

############################ LOAD AND CLEAN DATA ###################################################
#CSV loaded with 2000 tweets from twitter with #Obamacare. These tweets were pulled through the
#python file C:/Users/Caroline Lavoie/Jupyter/Text_Mining/tweets_for_902.ipynb

health = read.csv('C:/Users/Caroline Lavoie/Jupyter/Text_Mining/obamacare.csv', header = FALSE)

#name the columns in the dataset
names(health) <- c("Date", "Text")


# remove non alpha numeric characters from the text portion of the tweets
health$Text <- iconv(health$Text, from = "UTF-8", to = "ASCII", sub = "")


#make a corpus from a vector source 
review_corpus <- VCorpus(VectorSource(health$Text))

##Cleaning corpus - pre_processing. This function removes any URLs, replaces abbreviations,
#makes all letters lowercase, remove punctuation, numbers, and any stopwords. I also added in the 
#custom stop word of "brt". This returns a function to apply to the corpus of words.
clean_corpus <- function(cleaned_corpus){
  removeURL <- content_transformer(function(x) gsub("(f|ht)tp(s?)://\\S+", "", x, perl=T))
  cleaned_corpus <- tm_map(cleaned_corpus, removeURL)
  cleaned_corpus <- tm_map(cleaned_corpus, content_transformer(replace_abbreviation))
  cleaned_corpus <- tm_map(cleaned_corpus, content_transformer(tolower))
  cleaned_corpus <- tm_map(cleaned_corpus, removePunctuation)
  cleaned_corpus <- tm_map(cleaned_corpus, removeNumbers)
  cleaned_corpus <- tm_map(cleaned_corpus, removeWords, stopwords("english"))
  custom_stop_words <- c("brt")
  cleaned_corpus <- tm_map(cleaned_corpus, removeWords, custom_stop_words) 
  cleaned_corpus <- tm_map(cleaned_corpus, stripWhitespace)
  return(cleaned_corpus)
}
#Run the function on review_Corpus and assign it to Cleaned_review_corpus.
cleaned_review_corpus <- clean_corpus(review_corpus)

#################################################################################
################################## WORDCLOUDS ###################################
#################################################################################

############################## UNIGRAM Wordcloud ################################

#create a term document matrix  from the cleaned_review_corpus that will count the
#number of times a specific word appears in each document (tweet). Then transform 
#into a matrix which has terms on the left and documents across the top

# TDM/DTM
TDM_reviews <- TermDocumentMatrix(cleaned_review_corpus)
TDM_reviews_m <- as.matrix(TDM_reviews)

# Term Frequency
term_frequency <- rowSums(TDM_reviews_m)

# Assign the word frequencies to a dataframe
word_freqs <- data.frame(term = names(term_frequency), num = term_frequency)

# Create a wordcloud for the values in word_freqs
wordcloud(word_freqs$term, word_freqs$num,min.freq=10,max.words=500,colors=brewer.pal(8, "Paired"))

############################## BIGRAM Wordcloud #######################################

#Define a function to create ngrams of 2 for the term document matrix. Then repeat the
#steps to create a wordcloud

tokenizer <- function(x)
  NGramTokenizer(x,Weka_control(min=2,max=2))

bigram_tdm <- TermDocumentMatrix(cleaned_review_corpus,control = list(tokenize=tokenizer))
bigram_tdm_m <- as.matrix(bigram_tdm)

# Term Frequency
term_frequency <- rowSums(bigram_tdm_m)
# Sort term_frequency in descending order
term_frequency <- sort(term_frequency,dec=TRUE)

# Create word_freqs
word_freqs <- data.frame(term = names(term_frequency), num = term_frequency)
# Create a wordcloud for the values in word_freqs
wordcloud(word_freqs$term, word_freqs$num,min.freq=5,max.words=500,colors=brewer.pal(8, "Paired"))

############################## TRIGRAM Wordcloud #######################################

#Define a function to create ngrams of 3 for the term document matrix. Then repeat the
#steps to create a wordcloud

tokenizer <- function(x)
  NGramTokenizer(x,Weka_control(min=3,max=3))

trigram_tdm <- TermDocumentMatrix(cleaned_review_corpus,control = list(tokenize=tokenizer))
trigram_tdm_m <- as.matrix(trigram_tdm)

# Term Frequency
term_frequency <- rowSums(trigram_tdm_m)
# Sort term_frequency in descending order
term_frequency <- sort(term_frequency,dec=TRUE)

# Create word_freqs
word_freqs <- data.frame(term = names(term_frequency), num = term_frequency)
# Create a wordcloud for the values in word_freqs
wordcloud(word_freqs$term, word_freqs$num,min.freq=10,max.words=500,colors=brewer.pal(8, "Paired"))


############################## TF-IDF Wordcloud #######################################

#create a term document matrix using tf-idf as the weight

tfidf_tdm <- TermDocumentMatrix(cleaned_review_corpus,control=list(weighting=weightTfIdf))
tfidf_tdm_m <- as.matrix(tfidf_tdm)

# Term Frequency
term_frequency <- rowSums(tfidf_tdm_m)
# Sort term_frequency in descending order
term_frequency <- sort(term_frequency,dec=TRUE)

# Create word_freqs
word_freqs <- data.frame(term = names(term_frequency), num = term_frequency)
# Create a wordcloud for the values in word_freqs
wordcloud(word_freqs$term, word_freqs$num,min.freq=5,max.words=500,colors=brewer.pal(8, "Paired"))


#################################################################################
################################## SENTIMENT ANALYSIS ###########################
#################################################################################

#Create a sentiment analysis for each of the 2000 tweets using the bing lexicon.
#Join the bing lexicon to the words from the tweets and assign 1 (positive) or -1
#(negative) to each. Determine the overall sentiment score for each tweet. Finally,
#visualize the sentiment for each individual tweet using ggplot.


tidy_mytext <- tidy(TermDocumentMatrix(cleaned_review_corpus))
bing_lex <- get_sentiments("bing")
mytext_bing <- inner_join(tidy_mytext, bing_lex, by = c("term" = "word"))
mytext_bing$sentiment_n <- ifelse(mytext_bing$sentiment=="negative", -1, 1)
mytext_bing$sentiment_score <- mytext_bing$count*mytext_bing$sentiment_n
aggdata <- aggregate(mytext_bing$sentiment_score, list(index = mytext_bing$document), sum)
sapply(aggdata,typeof)
aggdata$index <- as.numeric(aggdata$index)
ggplot(aggdata, aes(index, x,fill = x > 0)) +
  geom_bar(alpha = 0.5, stat = "identity", show.legend = FALSE) +
  labs(x = "Tweets", y="Sentiment Score", title="Sentiment for 2000 #Obamacare Tweets April 11-15, 2018")

#################################################################################
################### COMPARISON AND COMMONALITY CLOUDS ###########################
#################################################################################

#Using the sentiment score for each tweet, divide the tweets into positive and
#negative. The words from these two sets of tweets are then used to find commonalities
#through the commonality cloud and differences through the comparison cloud.

# split the sentiment scores by positive and negative tweet values
positive <- aggdata[ which(aggdata$x>0),]
negative <- aggdata[ which(aggdata$x<0),]

# create an index column in mytext to be able to subset by tweet number
health$index = c(1:2000)

# Subset the tweets by positive and negative sentiments from the score via index value
ptweets <- health[ positive$index,]
ntweets <- health[ negative$index,]
positive_tweets <- paste(unlist(ptweets$Text),collapse="")
negative_text <- paste(unlist(ntweets$Text),collapse="")

speech <- c(positive_tweets,negative_text)

# remove non alpha numeric characters 
speech <- iconv(speech, from = "UTF-8", to = "ASCII", sub = "")

# making a corpus of a vector source 
speech_corpus <- VCorpus(VectorSource(speech))

# pre_processing 
clean_corpus <- function(corpus){
  cleaned_corpus <- tm_map(corpus, content_transformer(replace_abbreviation))
  cleaned_corpus <- tm_map(cleaned_corpus, removePunctuation)
  cleaned_corpus <- tm_map(cleaned_corpus, removeNumbers)
  cleaned_corpus <- tm_map(cleaned_corpus, removeWords, stopwords("english"))
  custom_stop_words <- c("the", "and")
  cleaned_corpus <- tm_map(cleaned_corpus, removeWords, custom_stop_words)
  cleaned_corpus <- tm_map(cleaned_corpus, stripWhitespace)
  return(cleaned_corpus)
}

cleaned_speech_corpus <- clean_corpus(speech_corpus)

# TDM/DTM
TDM_speech <- TermDocumentMatrix(cleaned_speech_corpus,control =list(wordLengths=c(0,Inf)))
TDM_speech_m <- as.matrix(TDM_speech)

########################### Commonality Cloud ###################################
#this code creates a word cloud based on words that are common among both the
#positive and the negative tweets

commonality.cloud(TDM_speech_m,colors=brewer.pal(8, "Dark2"),max.words =300) 

######################### Comparison Cloud #####################################
#this code creates a word cloud based on the words that are different between the
#positive and negative tweets

TDM_speech <- TermDocumentMatrix(cleaned_speech_corpus)
colnames(TDM_speech) <- c("Positive","Negative")
TDM_speech_m <- as.matrix(TDM_speech)
comparison.cloud(TDM_speech_m,colors=brewer.pal(8, "Dark2"),max.words = 100, scale=c(1,.5))

#################################################################################
################################## EMOTIONAL ANALYSIS ###########################
#################################################################################

#Using the NRC lexicon, this code determines the overall emotional feel for the 
#2000 tweets combined. This shows more than just positive and negative. It includes
#8 different emotions.

tidy_mytext <- tidy(TermDocumentMatrix(cleaned_review_corpus))
nrc_lex <- get_sentiments("nrc")
mytext_nrc <-inner_join(tidy_mytext,nrc_lex, by = c("term"="word"))
mytext_nrc_noposneg <- mytext_nrc[!(mytext_nrc$sentiment %in% c("positive","negative")),]
aggdata <- aggregate(mytext_nrc_noposneg$count,list(index=mytext_nrc_noposneg$sentiment),sum)
chartJSRadar(aggdata)


