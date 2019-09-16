#This code is written by @TamasPalfi as an REU participant of the SLANG lab at UMass Amherst.
#
# Goal: use the results from reddit.py and plotting.py to infer stats and other patterns
# about the /r/legaladvice data that we have modeled and predicted with.

#import dependencies
import joblib
import json
from statistics import mean


#start of my added functions

#function to take in the results and get an ordering of which labels sees the greatest increase (+ change) in prev. est
# from the first month (Jan 2015) to last month (Dec 2018) to the greatest decrease (- change) in prev est.
def ordering(labels, results):
    #counter variable for labels
    i = 0
    #create dict object to hold the ordered results
    res = {}
    for label_pred in results:
        # get this category/label we are plotting for
        label = labels[i]

        print(label)

        # get the pred for each metric over the 48 months
        y_cc = label_pred[0]
        y_pcc = label_pred[1]
        y_freq = label_pred[2]
        #get the split list from Oct 2015 to Dec 2018
        y_cc_split = y_cc[9:]
        y_pcc_split = y_pcc[9:]
        y_freq_e_split = y_freq[9:]
        #get the averages of all of these splits
        cc_avg = mean(y_cc_split)
        pcc_avg = mean(y_pcc_split)
        freq_e_avg = mean(y_freq_e_split)

        print("CC avg:")
        print("%0.3f" %cc_avg)
        print("PCC avg: ")
        print("%0.3f" %pcc_avg)
        print("freq_e avg: ")
        print("%0.3f" %freq_e_avg)

        #get the difference between Oct 2015 and last month for freq_e
        freq_e_diff = y_freq_e_split[38] - y_freq_e_split[0]
        #add this difference to ordered_res dict as 'label': freq_e_diff key-value pair
        res[label] = freq_e_diff #Will order plots by freq-e though so put that here
        i += 1
    #sort the items in the res dict by lowest to highest for value
    ordered_res = sorted(res.items(), key = lambda kv: (kv[1], kv[0]))
    return ordered_res




#main method - takes in the results, labels arrays and runs functions to get stats, metrics
def main(labels_file, results_file):
    print("start")
    # decompress the file for the labels
    labels = joblib.load(labels_file)
    # remove native american label - no results for it
    lbls = []
    for label in labels:
        if label == "Native American Issues and Tribal Law":
            continue
        else:
            lbls.append(label)
    # load the results file
    with open(results_file) as f:
        res = json.load(f)
    #get results_master list from results dict
    results = res['results']
    ordered_res = ordering(lbls,results)

    print(ordered_res)
    print("end")


###############################################
#call statements for program functions

if __name__ == '__main__':
    main('labels.txt', 'results.json')

###############################################
#citations

#@TamasPalfi








