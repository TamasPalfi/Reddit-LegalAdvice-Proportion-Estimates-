# This code is written by @TamasPalfi as an REU participant of the SLANG lab at UMass Amherst. The code is inspired by
# work from the Stanford Learned Hands project (https://learned-hands.github.io/project-hub/index.html) in collecting
# an online legaladvice dataset and taking their dataset to create a ML model to compute accuracy, F1 scores,
# prevalence estimation (CC, PCC, and freq-e [our software pckg]) for each of the 20 label categories.
#
# Goal of Stanford Group: To create a better ML model to represent this dataset fully.  TODO: I haven't really optimized
# #TODO (cont) - for  better model yet so before report anything to stanford should do that.  Optimizations include:
# TODO (cont) - (i) BOW puncuation error - including '[' ']' due to lists format (ii) F1 = 0 error? (iv) splits - kfold?
#
# Our Goal: To use their dataset to create a ML model that will allow use to obtain prevalence estimations for each
#           label given the dataset.  Want to compare our package freq-e for prev. est in comparison to baselines of CC,
#           PCC.
#
# NOTE: THis is a multi-label problem here.

##############################################
#import dependencies
import numpy as np
import pandas
from sklearn import *
import re
import nltk
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import f1_score, roc_auc_score
import freq_e

##############################################
#start of our added program functions

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

#function to perform getting splits in data to form training set and validation/test set
def splits(data):
    #use pre-made sklearn model_selection method to get train/test splits       #TODO: optimize with KFold CV
    train, test = model_selection.train_test_split(data, test_size = 0.2)
    return train, test

#function to perform the bag of words part for the model -returns a list of dictionaries with count of all words in data
def bow(data):
    #print(data)

    #(i) pre-process text first to get rid of punctuation and capitals
    #variable to hold list of all sentences in post
    all_sent = []
    #loop through each post
    for post in data:
        #tokenize the text first
        #variable will be a list of all sentences in this post
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
        #increment post_index
        post_index += 1

    #print(all_sent)

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

#function to do one vs rest classifying on the data for each of the 20 labels (will have 20 classifier data pred outputs
#  as a result).
#
# Pre: Takes in the training dataset from the Stanford Group
# Post: returns a list of results for each label (19 only for 'Native American & Tribal Law' is removed due to no +
#       labels.  Outer list size is 19 and inner is 6 with the following format: [acc, f1, roc_auc, cc, pcc, freq-e]
def ovr_classif(training_data, labels):
    #to create a successful model we have to split the large dataset into training and validation/testing sets - call
    #helper function to do so
    train_split, test_split = splits(training_data)
    #with the full data (text & labels) successfully split --> we need to get just text data for BoW model
    x_train = train_split[:,0]
    x_test = test_split[:,0]

    #perform BOW - returns a list of dictionaries of word-count pairs for each post
    word_dic_train = bow(x_train)
    word_dic_test = bow(x_test)

    #create DictVectorizer to convert dict object to array for ML fitting
    v = DictVectorizer()
    train_bow = v.fit_transform(word_dic_train).toarray()
    test_bow = v.transform(word_dic_test).toarray()

    #create results master list to store ind. label results for each prediction type
    results_master = []
    for i in range(1,21):
        #get labels
        #make new counter specific to the labels
        j = i -1
        label = str(labels[j])
        # handle edge case for label "Native American Issues & Tribe Law"
        if i == 16:
            #skip for indian case
            continue

        print(i)
        print(label)

        #use this index to get the label data we are testing for in this iteration - cast to 'int' to fix fitting issue
        train = train_split[:,i].astype('int')
        #get training class label prior distribution --> used later for freq-e estimates
        label_prior = np.mean(train)
        #get true test labels
        test = test_split[:,i].astype('int')

        #get true/'gold standard' labels true prevalence
        true_prev = np.mean(test)
        print("True Test Prev: ")
        print(true_prev)

        #create One vs Rest classifier using pre-made sklearn class using the same estimators used by Stanford Learned Group
        # - these estimators can be changed later for optimization purposes                                                #TODO: optimize estimators
        model = linear_model.LogisticRegression(C=1000000000.0, class_weight='balanced', dual=False,
            fit_intercept=False, intercept_scaling=1, max_iter=100,
            multi_class='warn', n_jobs=None, penalty='l2', random_state=None,
            solver='lbfgs', tol=0.0001, verbose=0, warm_start=False)

        #model = multiclass.OneVsRestClassifier(estimator=[('logistic', linear_model.LogisticRegression(C=1000000000.0, class_weight='balanced', dual=False,
           # fit_intercept=False, intercept_scaling=1, max_iter=100,
            #multi_class='warn', n_jobs=None, penalty='l2', random_state=None,                                              #TODO: One vs. Rest api say pass in an estimator object but attributes say 'estimators' --> only could use LogReg Estimator for now

        # now with our model created we can: (i) fit (ii) predict (iii) compute metrics
        # fit on train for label y
        model.fit(train_bow, train)
        # create list for results for this classifier - will hold 6 stats in it: [acc, f1, roc_auc, cc, pcc, freq-e]
        res_single = []
        #predict on test
        pred = model.predict(test_bow) #labels
        pred_prob = model.predict_proba(test_bow) #probabilites of + and - class
        #get the pred prob list for positive class only --> used for roc_auc and freq-e
        pos_pred = pred_prob[:,1]

        acc = model.score(test_bow,test)
        f1 = f1_score(test, pred)
        roc_auc = roc_auc_score(test, pos_pred)
        print("Acc: ")
        print("%0.3f" %acc)
        print("F1: ")
        print("%0.3f" %f1)
        print("Roc_auc: ")
        print("%0.3f" %roc_auc)

        #Perform prevalence estimation on this specific label
        cc_pred = cc(pred)
        pcc_pred = pcc(pred_prob)
        #perform our method of prev. est. which is Freq-e --> use label prior from training data here
        freq_e_res = freq_e.infer_freq_from_predictions(pos_pred, label_prior)

        #calc absolute difference between true prevalence and predicted prev from freq-e
        #get the actual freq_e point prediction
        freq_e_pred = freq_e_res['point']
        abs_diff = abs(freq_e_pred-true_prev)

        print("Freq-e Pred: ")
        print("%0.3f" %freq_e_pred)
        print("Diff in prev: ")
        print(abs_diff)

        # Note:all of these are exact prob. -
        #     not the rounded result.
        res_single.append(acc)
        res_single.append(f1)
        res_single.append(roc_auc)
        res_single.append(cc_pred)
        res_single.append(pcc_pred)
        res_single.append(freq_e_pred)

        #add this label's result list to master list
        results_master.append(res_single)
        print(res_single) #TODO: better as a dict actually saying what each one is.
    return results_master

#function to get the training/true prevalence on the whole stanford data set (1,820 ex) (not just the 80/20% splits)
def true_prior(training_data, labels):
    for i in range(1,21):
        #get labels
        #make new counter specific to the labels
        j = i -1
        label = str(labels[j])
        # handle edge case for label "Native American Issues & Tribe Law"
        if i == 16:
            #skip for indian case
            continue

        print(i)
        print(label)

        # use this index to get the label data we are testing for in this iteration - cast to 'int' to fix fitting issue
        data = training_data[:, i].astype('int')
        # get training class label prior distribution --> used later for freq-e estimates
        label_prior = np.mean(data) #on all 1,820 examples

        print("True Prior: ")
        print("%0.3f" %label_prior)

# function that should be called to run the main program
# Pre: training file is the csv file containing all of the stanford data
def main(training_file):
    print("Main: ")

    # get the training data & labels in correct format via pre-processing
    training_data, labels = read_training_data(training_file) #numpy arrays
    #call method to do One Vs Rest classification for each of 20 labels
    res = ovr_classif(training_data, labels)
    #true_prior(training_data,labels)

    print("Done")

##############################################
#program calls

if __name__ == '__main__':
    main('2019-04-23_best-guess_binary.csv')


##############################################
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
