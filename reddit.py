# This code is written by @TamasPalfi as an REU participant of the SLANG lab at UMass Amherst. The code is inspired by
# work from the Stanford Learned Hands project (https://learned-hands.github.io/project-hub/index.html) in collecting
# an online legaladvice dataset. This dataset is applied to get prevalence estimation (CC, PCC, and freq-e
# [our software pckg]) for each of the 20 label categories.
#
# Goal:  The main goal is to extend prior work to use the model created from stanford.py to be applied on all of
# /r/legaladvice posts from 2015-2018 and compute prevalence estimations for each month for each of the labels. The
# resulting goal is to have a display of graphical distributions showing the change in prevalence estimations over
# monthly time frames in a four year period which will be done in plotting.py.
#
# Pre: Two files are passed in from linux private server names Hobbes. The first is the training file with all of the
#      standford data used in stanford.py.  The second is a path to a directory containing all of the reddit legaladvice
#      data from each month of 2015-2018.
# Post: Two files will be written to.  Labels.txt will be joblib dumped to to contain an array of the labels.  Results.
#       json will contain the results masterlist - what is described and obtained in ovr_classif function.

#import dependencies
import numpy as np
import pandas
from sklearn import *
import re
import nltk
from sklearn.feature_extraction import DictVectorizer
#from sklearn.metrics import f1_score
import freq_e
import os
import sys
import json
import joblib

# start of added functions

# Function to split training data into "good" data --> by "good" here we mean:  all data examples with at least one
# category 'labeled' ( 1 ).  The reasoning is that those data examples are the most relevant for they provide useful
# classifier data in comparison to an example that suffered from poor annotation with no labels or a lot of NaNs. For
# the data we do take as 'good', we convert all NaNs to 0's so as to actually be able to perform ML fitting.
#
# Pre: start with 3,549 data examples/rows
# Post: a reduced amount of daa
def get_good_data(data):
    # need to create loop to iterate over the rows
    for row in data.itertuples():
        # set up boolean value for tracking (1)
        is_good = False
        # get index of this row in case have to remove it
        index = row[0]
        # need to loop over categories/labels to make sure at least one yes ( 1 )
        for x in row:
            #check for if a (1)/yes for a label is present
            if x == 1.0:
                is_good = True
                break #exits inner loop and goes to next data example
        #check to see if row has no (1)'s  ("NOT is_good") --> if does --> remove row from dataframe
        if not is_good:
            #bad row - remove it
            data = data.drop(index)
    #now with all the "good" rows being the only ones remaining - can replace all NaNs with 0
    data =  data.fillna(0)
    return data

#function to read the input training data and put it into an easily usable format for ML models aka pre-processing
def read_training_data(training_file):
    #use pandas' easily usable and already created library for reading CSV file
    data = pandas.read_csv(training_file)  #dataframe object
    #drop off the unique id column --> has no statistical relevance to our goal, used for id/security purposes.
    data = data.drop(columns=["_id"])
    #gets the column/label names such as "Housing, Transportation" for later usage
    labels = data.columns.values[1:21]
    # function to parse dataset to get "good" examples
    data = get_good_data(data)
    #convert dataframe object to numpy array
    data = data.values  #.values attr converts df to array
    return data, labels

#function to pre-process the test data into a readily usable format for ML methods
#Pre: each test file is a .txt file containing all the posts from /r/legaladvice for a certain month.  Its format is a
#     list of dictionary objects with each object representing a post. The data we care for explicitly is the "selftext"
#     component which is actual text of the post.
#Processing: We get rid of the following items in our processing search: (i) posts with no key value "subreddit" (ii)
#            posts not from sub='legaladvice' (iii) posts with an empty body of text (iv) those with text being
#            deleted (i.e "[deleted]")
#Post: we want the output to be a numpy array where each element of the array is just a post's body of text
def read_test_data(month_file):
    #need array to hold the posts text for this month
    posts_month = []
    cnt_begin = 0
    #iterate through each dictionary object to do (i)pre-processing step to get "good" data (ii) add to list
    for line in open(month_file,'r'): #month file is the path to the file name here, this opens the file
        cnt_begin += 1
        #get the actual json loading of the file
        post = json.loads(line)
        #variable to represent the text of that post
        txt = post["selftext"]
        #set up boolean variable to be used to track whether or not to add this post to the list - Does it pass pre -
        # processing?
        is_good = True  #we assume that the post originally would be of viable format
        # (i) pre-processing
        # (a) posts with no key value "subreddit"
        if ('subreddit' in post) is False:
            #support post to main post - skip it.
            continue
        # (b) check that the post is subreddit "legaladvice"
        if post["subreddit"] != "legaladvice":
            #post is from an unrelated subreddit - skip it
            continue
        # (c) check for if text of post is empty()
        if txt == "":
            #post is empty of text so don't include
            is_good = False
        # (d) check for if post's text was deleted
        if txt == "[deleted]":
            #post's text was deleted so dont include
            is_good = False
        #(ii) check for if whether to add or not
        if is_good:
            #the post passed (i) so add its selftext to the list
            posts_month.append(txt)

    return posts_month

#function to perform the bag of words part for the model - returns a list of dictionaries with count of all words in data
#Pseudocode based on https://www.geeksforgeeks.org/bag-of-words-bow-model-in-nlp/
def bow(data):
    #print(data)

    #(i) pre-process text first to get rid of punctuation and capitals
    #variable to hold list of all sentences in post
    all_sent = []
    #loop through each post
    for post in data:
        #tokenize the text first
        #variable will be a list of all sentences in this document
        post_tok = nltk.sent_tokenize(str(post))
        #add to master sentence list
        all_sent.append(post_tok)  #TODO make extend to make one list , have repeeated '[' and ']' character in final model
    # perform pre-processing - each post_sent here is a list of sentences for each post
    post_index = 0
    for post_sent in all_sent:                 #TODO POSSIBLE ERROR Still have some punctuation in BOW model
        sent_index = 0
        #loop through each sentence in list
        for sent in post_sent:
            #pre-process text to make it same case (lower) and remove the punctuation
            all_sent[post_index][sent_index] = sent.lower()
            all_sent[post_index][sent_index] = re.sub(r'\W', ' ', sent)
            all_sent[post_index][sent_index] = re.sub(r'\s+', ' ', sent)
            #increment sent_index
            sent_index += 1
        #increment doc_index
        post_index += 1

    #(ii) obtain freq count for each word and store in dictionary to be converted later for BOW model  #TODO Speed up with Counter() Module
    # list variable to hold word-count key-value pairs for dict for each document
    word2cnt = []
    #loop through each document
    for doc_sent in all_sent:
        doc_dict = {}
        #loop through each sent in document sentence list
        for sent in doc_sent:
            #tokenize the sentence to get a list of words only
            words = nltk.word_tokenize(sent)
            #iterate through words to see if in dictionary or not.
            for word in words:
                #check if NOT in dict already
                if word not in doc_dict:
                    #word not in --> add it wiht cnt = 1
                    doc_dict[word] = 1
                else:
                    # word is already present in dic --> increment count
                    doc_dict[word] += 1
        #add this documents dict to the list of all dictionaries
        word2cnt.append(doc_dict)

    # ADD THIS if want to limit to n words in the dictionary so as not have huge number of words
    #import heapq
    #freq_words = heapq.nlargest(100, word2count, key=word2count.get)

    #return word2cnt representation
    return word2cnt #list of dictionaries

#function to perform the CC version of prevalence estimation on binary. Takes in an array of label prediction results
def cc(pred_res):
    #create a count variable to hold all the 1/yes's for this label
    cnt_yes = 0
    #iterate through all of the predicted documents to get freq. counts
    for doc in pred_res:
        if doc == 1:
            #document is labeled as yes/1 for this label so include in count
            cnt_yes += 1
    #now with total 'yes' freq. counts -> compute the prevalence estimation
    #print(cnt_yes)
    prev_est = cnt_yes/len(pred_res)
    print("CC Prev Est: ")
    print(prev_est)
    #return
    return prev_est

#function to perform the PCC version of prev. est. on binary.  Takes in array of pred prob arrays for each document.
def pcc(prob_pred_res):
    #create a count variable to hold total sum of 1/yes probabilities
    cnt = 0.0
    #iterate through each of the class predicted prob. for each document and sum up the prob. of positive class
    for doc_pred_res in prob_pred_res:
        #get the probabiltiy of the positive class -this is the second class so arr[1] to index
        prob = doc_pred_res[1]
        #add it to the count
        cnt += prob
    #now with sum of prob. of yes/1 we can compute the prev. est.
    prev_est = cnt/len(prob_pred_res)
    print("PCC Prev Est: ")
    print(prev_est)
    return prev_est

#function to do one vs rest classifying on the data for each of the 20 labels - the result will be a huge list that has
# within it a sublist for each labels results.  These sublists each have three lists inside them as well with each of
# these inner lists storing the prev. pred results for a specific metric (CC, PCC, freq_e) over the 48 months.
def ovr_classif(training_data,list_of_test_files, labels): #labels is part of TODO FEATURETEST

    #perform BOW - returns a list of dictionaries of word-count pairs for each post
    word_dic_train = bow(training_data[:,0])
    #create DictVectorizer to convert dict object to array for ML fitting
    v = DictVectorizer()
    train_bow = v.fit_transform(word_dic_train).toarray()


    #create results master list to store ind. label results for each prediction type
    results_master = []
    #master list to store results of freq-e CI
    freq_e_master = []
    for i in range(1,21):
        # create path to directory of pred outputs
        path = 'preds/'  # part of TODO FEATRUETEST

        # handle edge case for label "Native American Issues & Tribe Law"
        if i == 16:
            #skip for indian case
            continue

        print(i) #TODO remove

        #get label name and add to pathS
        label  = labels[i-1] #TODO THESE LINES are part of FEATURETEST
        path += str(label)

        #use this index to get the label data we are testing for in this iteration - cast to 'int' to fix fitting issue
        train_y = training_data[:,i].astype('int')
        #get training class label prior distribution --> used later for freq-e estimates
        label_prior = np.mean(train_y)

        #create One vs Rest classifier using pre-made sklearn class using the same estimators used by Stanford Learned Group
        # - these estimators can be changed later for optimization purposes                                                #TODO: optimize estimators
        model = linear_model.LogisticRegression(C=1000000000.0, class_weight='balanced', dual=False,
            fit_intercept=False, intercept_scaling=1, max_iter=100,
            multi_class='warn', n_jobs=None, penalty='l2', random_state=None,
            solver='lbfgs', tol=0.0001, verbose=0, warm_start=False)

        #model = multiclass.OneVsRestClassifier(estimator=[('logistic', linear_model.LogisticRegression(C=1000000000.0, class_weight='balanced', dual=False,
           # fit_intercept=False, intercept_scaling=1, max_iter=100,
            #multi_class='warn', n_jobs=None, penalty='l2', random_state=None,                                              #TODO: One vs. Rest api say pass in an estimator object but attributes say 'estimators' --> only could use LogReg Estimator for now
            #solver='warn', tol=0.0001, verbose=0, warm_start=False))]) #, ('GaussianNB', sklearn.naive_bayes.GaussianNB(priors=None, var_smoothing=1e-09))],

        # now with our model created we can: (i) fit (ii) predict (iii) compute metrics
        # fit on train for label y
        model.fit(train_bow, train_y)
        # create list for freq_e results for this classifier - will hold three lists in it: cc, pcc, and freq-e results
        freq_e_single = []
        #create sublists for each metric
        cc_list = []
        pcc_list = []
        freq_e_list = []
        full_freq_e_list = []
        #loop through each month file
        for month in list_of_test_files:
            month_name = month[48:]
            print(month_name)
            # pre-process month file/data for ML methods
            # month file is list of dictionaries
            test = read_test_data(month)   #want another numpy array of just the self_text (text of post)
            #get the bow features for the test data in the month file
            word_dic_test = bow(test)
            #transform dic_test to the already fitted DictVectorizer object
            test_bow = v.transform(word_dic_test).toarray()

            #predict on test
            pred = model.predict(test_bow)
            pred_prob = model.predict_proba(test_bow)

            pred_prob_list = pred_prob.tolist()
            print(pred_prob_list) #TODO REMOVE ASAP
            print(len(pred_prob_list))

            #acc = model.score(test_bow,test_y)
            #f1 = f1_score(test_y, pred)
            #print(labels[i])
            #print("Acc: ")
            #print(acc)
            #print("F1: ")
            #print(f1)

            #Perform prevalence estimation on this specific label
            cc_pred = cc(pred)
            pcc_pred = pcc(pred_prob)
            # perform our method of prev. est. which is Freq-e --> use label prior from training data here
            freq_e_res = freq_e.infer_freq_from_predictions(pred_prob[:,1], label_prior) #pred_prob[:,1] is the % prob of
            print("Freq-e: ")                                                            #1/yes for that post fitting label
            print(freq_e_res)

            #calc absolute difference between true prevalence and predicted prev from freq-e
            #get the actual freq_e point prediction
            freq_e_pred = freq_e_res['point']
            #abs_diff = abs(freq_e_pred-true_prev)
            #print("Diff in prev: ")
            #print(abs_diff)

            #add the point predictions to appropriate list
            cc_list.append(cc_pred)
            pcc_list.append(pcc_pred)
            freq_e_list.append(freq_e_pred)
            full_freq_e_list.append(freq_e_res)

            #add the month to the path and dump pred results to that file
            month_path = path +  '/' + str(month_name) 
            #format dict object for json dump
            pred_final = {'pred': pred_prob_list}
            # dump results to file
            #with open(month_path, 'w') as x:
                #json.dump(pred_final, x)

        #append the pred lists to the single list for this label   
        freq_e_single.append(cc_list)
        freq_e_single.append(pcc_list)
        freq_e_single.append(freq_e_list)
        #add this label's result list to master list
        results_master.append(freq_e_single)
        freq_e_master.append(full_freq_e_list)

        #print(freq_e_single) #TODO REMOVE
        print(full_freq_e_list)

    print(freq_e_master)
    freq_e_master_dic = {'freq_e_master': freq_e_master}
    #put to file
    #with open('freq_e.json', 'w') as y:
        #json.dump(freq_e_master_dic, y)

    #set up dict struct to allow for json dump
    res = {'results': results_master}
    return res    

# Goal: The purpose of this code
# is to read all of the files (e.g RS_2015-01.txt) in the 'legaladvice' directory on the Hobbes private server and to
# output the file path names to a list.  This list will include all of the testing data needed.
#
# Pre: The path of the directory containing all of the text files for each month
# Post: A list is returned which will then be passed to ovr classif function to test on model
def dir_reader(path_to_dir):
    # get a list of all the files in the directory which is RS_2015-01.txt to RS_2018-12.txt
    files = sorted(os.listdir(path_to_dir))
    #create a new list to store the updated file paths to
    list_files = []
    for file in files:
        file_path = path_to_dir + "/" + file
        list_files.append(file_path)
    return list_files

# function that should be called to run the main program
# Pre: trainiing file is stanford data,
#      path_to_dir is the path to the directory containing all the month files aka the test data
def main(training_file, path_to_dir):

    print("Main: ")

    # get the training data & labels in correct format via pre-processing
    training_data, labels = read_training_data(training_file) #numpy arrays
    # get the file names of the test data by calling approp. func
    list_of_test_files = dir_reader(path_to_dir)

    #dump the labels to be the first line of results.txt
    joblib.dump(labels,'labels.txt')

    #call method to do One Vs Rest classification for each of 20 labels and output plots of freq_e estimates
    res = ovr_classif(training_data,list_of_test_files, labels) #labels here is part of TODO FEATUERETEST

    #print(res)

    #dump results to file
    #with open('results.json', 'w') as f:
        #json.dump(res, f)
    print("Done")    


########################################################################

#call statements for program functions
if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2])

########################################################################
#citations

#Main code contributor: @Tamas Palfi

#Stanford: https://learned-hands.github.io/project-hub/index.html

#BoW: Pseudocode based on https://www.geeksforgeeks.org/bag-of-words-bow-model-in-nlp/

#Freq-e software
#@inproceedings{keith2018uncertainty,
#  title={Uncertainty-aware generative models for inferring document class prevalence},
#  author={Keith, Katherine and O'Connor, Brendan},
#  booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
#  year={2018}
#}
