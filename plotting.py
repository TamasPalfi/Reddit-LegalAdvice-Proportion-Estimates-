# This code is written by @TamasPalfi as an REU participant of the SLANG lab at UMass Amherst.
#
# Goal: The resulting goal is to have a display of graphical distributions showing the change in prevalence estimation
#  w/ respect to three measures (CC, PCC, & freq-e) over monthly time frames in a four year period.
#
# Pre: A compressed numpy array of labels and a json file of the results master list.
# Post: graphs will be shown on screen and subsequently saved to file.

#import dependicies
import matplotlib.pyplot as plt
import joblib
import json
import numpy as np

#start of added functions

#this function takes the results megalist &
#creates plots showing the CC, PCC, & freq-e prediction changes for each label over the 48 months
def plotting(labels,results,intervals):
    #create list for x_axis:
    x_ax = ["Jan-15", "Feb-15", "Mar-15", "Apr-15", "May-15", "June-15", "July-15", "Aug-15", "Sep-15", "Oct-15",
            "Nov-15", "Dec-15","Jan-16", "Feb-16", "Mar-16", "Apr-16", "May-16", "June-16", "July-16", "Aug-16",
            "Sep-16", "Oct-16", "Nov-16", "Dec-16","Jan-17", "Feb-17", "Mar-17", "Apr-17", "May-17", "June-17",
            "July-17", "Aug-17", "Sep-17", "Oct-17", "Nov-17", "Dec-17","Jan-18", "Feb-18", "Mar-18", "Apr-18",
            "May-18", "June-18", "July-18", "Aug-18", "Sep-18", "Oct-18", "Nov-18", "Dec-18"]

    #counter variable for labels
    i = 0
    #each list is already set up in the correct format meaning that it is ordered by freq-e predictions by month
    # we simply just have to plot the results of each freq_e list for each label. We iterate through each label...
    for label_pred in results:
        print(i)
        #get the label interval list
        label_interval = intervals[i]
        #convert CI to floats to be able to plot #TODO remove not needed - waste of computational speed.
        list = []
        for interval in label_interval:
            inner_list = []
            for bound in interval:
                inner_list.append(float(bound))
            list.append(inner_list)
        #create separate lists for the non-symmetric error bar bounds
        lower = []
        upper = []
        full = []
        for interval_bounds in list:
            #get the upper and lower bound then append to the respective lists
            low = float(interval_bounds[0])
            high = float(interval_bounds[1])
            lower.append(low)
            upper.append(high)
        #add the lower and upper bounds lists to one list for (2,n) shape for error bar plotting.
        full.append(lower)
        full.append(upper)

        print(label_interval)
        print(full)

        #get the pred for the y-axis
        y_cc = label_pred[0]
        y_pcc = label_pred[1]
        y_freq = label_pred[2]
        #plot graph with x and y's
        plt.plot(x_ax, y_cc, color = 'blue', label = 'cc')
        plt.plot(x_ax, y_pcc, color='green', label = 'pcc')


        plt.errorbar(x_ax, y_freq, yerr= full, color='red', label='freq-e')


        #j = 0
        #for freq_e_point in y_freq:
            #interval = label_interval[j]
            #plt.errorbar(x_ax, freq_e_point, xerr=interval, color='red', label='freq-e')
            #j += 1

        #get this category/label we are plotting for
        label = labels[i]

        print(str(label)) #TODO remove

        #set y-axis intervals
        plt.xticks(rotation = 90)
        plt.yticks(np.arange(0,0.5,0.05))
        #create legend - loc=1 is top right corner 
        plt.legend(loc = 1)
        # set up titles
        title = "Prevalence Estimation Predictions over 2015-18 for " + str(label)
        plt.title(title)
        plt.xlabel("Month & Year")
        plt.ylabel("Prevalence Estimation Predictions")
        #show the plots
        plt.show()
        #increment i to move on to next label
        i += 1




#main method - takes in the results array and runs the plotting function
def main(labels_file, results_file, freq_e_ci_file):
    print("start")

    #decompress the file for the labels
    labels = joblib.load(labels_file)
    lbls = []
    #remove native american law
    for label in labels:
        if label == "Native American Issues and Tribal Law":
            continue
        else:
            lbls.append(label)
    #load the results file
    with open(results_file) as f:
        res = json.load(f)
    # get results_master list from results dict
    results = res['results']
    #get freq_e interval
    with open(freq_e_ci_file) as y:
        freq_e_ci_dict = json.load(y)
    #perform processing to get correct format of CIs
    freq_e_ci_begin = freq_e_ci_dict['freq_e_master']
    freq_e_end = []
    for freq_e_labels in freq_e_ci_begin:
        freq_e_CI_mid = []
        for freq_e_month in freq_e_labels:
            #create another temp list to store results
            adj_bounds = []
            #get only interval part of freq-e pred
            freq_e_ci = freq_e_month['conf_interval']
            #get the upper and lower bound
            lower_pt = freq_e_ci[0]
            upper_pt = freq_e_ci[1]
            #get the freq-e point estimate
            point = freq_e_month['point']
            #get the adjusted bounds
            lower_adj = point - lower_pt
            upper_adj = upper_pt - point
            #add to temp list
            adj_bounds.append(lower_adj)
            adj_bounds.append(upper_adj)
            freq_e_CI_mid.append(adj_bounds)
        freq_e_end.append(freq_e_CI_mid)
    #run the plotting function
    plotting(lbls, results, freq_e_end)

    print("end")

#####################################
#call statements for program functions

if __name__ == '__main__':
    main('labels.txt', 'results.json', 'freq_e.json')


######################################
#citations

#@TamasPalfi
