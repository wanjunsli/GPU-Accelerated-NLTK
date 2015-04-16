# Using examples from the NLTK documenation
# and from Shankar Ambady, Microsoft New England Research and Dev Center

import time
from nltk import word_tokenize, NaiveBayesClassifier
from nltk import classify
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
import random
import os, glob, re



def get_features(text):
    lem = WordNetLemmatizer()
    commonwords = stopwords.words('english') 
    features = {}
    tokens = [] 

    for word in word_tokenize(text):
	tokens.append(lem.lemmatize(word.lower()))

    for word in tokens:
	if word not in commonwords:
	    features[word] = True

    return features

def main():
    time0 = time.time()
    
    # create two lists, one for ham emails and one for spam 
    ham_emails = []
    spam_emails= []
    total_emails = []
    features = []

    # read in emails
    for infile in glob.glob('ham2/*.txt'):
	    infl = open(infile, "r")
	    ham_emails.append(unicode(infl.read(), errors='ignore'))
	    infl.close()


    for infile in glob.glob('spam2/*.txt'):
	    infl = open(infile, "r")
	    spam_emails.append(unicode(infl.read(), errors='ignore'))
	    infl.close()

    time1 = time.time()
    
    print "Read in files, time elasped = " + str(time1 - time0)
    print "\n"

    time5 = time.time()
    
    # combine lists + shuffle
    for email in ham_emails:
    	total_emails.append((email, 'ham'))

    for email in spam_emails:
    	total_emails.append((email, 'spam'))


    random.shuffle(total_emails)
    
    for (email, cat) in total_emails:
    	strfeats = get_features(email)
    	features.append((strfeats, cat))
    time6 = time.time()
    print "Procesed text, time elasped = " + str(time6 - time5)
    print "\n"

    size = int(len(features)*0.7)

    training = features[size:]
    test = features[:size]
    
    time2 = time.time()
    classifier = NaiveBayesClassifier.train(training)
    
    time3 = time.time()
    
    print "Trained classifer, time elasped = " + str(time3 - time2)
    print "\n"

    print classifier.labels()

    time4 = time.time()
    print classify.accuracy(classifier, test)
    
    print "Classied test, time elasped = " + str(time4 - time3)
    classifier.show_most_informative_features(20)


main()
