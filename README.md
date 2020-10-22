# Reddit-LegalAdvice-Class-Proportion-Estimates
Project that showcases my most extensive research on a large dataset as such is one of the largest problems that I have worked on to date.

Before I get into the details of the project, this work was based off on an extension of research from Assistant Professor Brendan O'Connor and graduate student Katherine O'Kieth.  The main research introduced a new method developed by them called Freq-e to infer class prevalence estimation (e.g. Sentiment analysis) on group of items/documents.  This new method was compared in accuracy to older categorical proportion algorithms such a CC, PCC, and ACC, and introduced CI to the methods.  My part included working on a large-scale side project that used this new method compared to baselines on a very large dataset of reddit data.  As such I used their freq-e software package which can be found at their project website [here](http://slanglab.cs.umass.edu/doc_prevalence/).  Please cite them properly if their software is ever used in another project.  

Also, this project was used in a collaboration with the Stanford Learned Hands group/project.  They were seeking a better ML model to represent their ML game, and we in turn had use of a great training dataset on their classifier of online legal advice posts. More info about their project can be found at their [website](https://learned-hands.github.io/project-hub/).  Due to their help in having a legal advice classifier with training data already prepared, we cite them here and thank them for working together. 

Many common ML packages were used to help with this project such as sklearn, etc...

# The Project
The main goal of this project was to build a classifier based off on the 'Stanford' LearnedHands game data and then use that classifier to extend it to the full of reddit legal advice subreddit.  A legal advice posts online formally depicts a problem in someone's life that they are not sure if they need legal involvement with.  From there, the method we created in our past paper (freq-e) would be used to compare for accuracy (and other metrics) against other class proportion baselines such as CC (classify & count), PCC (probalistic classify & count), and ACC (adjusted classify & count).

**NOTE:** To see the most accurate documentation of the project details and results please see the Results Summary word file.  There are all of my notes and results including metric analysis and plots.  Another file to check out is the input data metrics excel file which has some analysis on the amount of data involved this project.

The ML classifier used in this project wasn't actaully too complex, and that is one change that could be integrated is to test around more options to improve the classifier.  The classifier used was a One vs Rest classifier using Logistic regression.  Detail on the hyperparams can be found in the code, but it should be noted that I tried to keep it consitent with the orginal choice from Stanford Learned Hands so as to make it similar.  

The Stanford training data was used to train the classifier.  There were 20 categories (one had to be dropped due to insufficient data) involved in their ML game to annotate the data, so there were 19 classed reported on and used for the OvR classifier.  The amount of data trained on was 1,820 annoted examples.  Check out **stanford.py** and the **Learned Hands website**.

The actual data that the classifier was used on was the reddit legal advice data from a public source project that saves reddit posts for a certain month.  I downloaded **all of the reddit posts** from **January 2015 to December 2018** and then ran command line scripts to get all of the /r/LegalAdvice posts that passed the pre-processing (e.g. had a post text, not a bot, etc.).  The size of the data set before was **516,641 posts** and after the pre-processing that number went down to **392,116 posts** which is what was tested on.

The **code** that should be looked at if are curious is **reddit.py**.  That is where the classifier built and trained is used to run on all of the reddit data compiled to get results of predictions on each class for each post.  Note, that this is **multilabel** problem as in such each post could be labeled to multiple categories.  It usually involved saving the results of each month and class to a prediction folder but that part I haven't had a chance to upload to this repo yet.  Other **code** to look at is **plotting.py** which creates the plots from the results and **anlaysis.py** which does analysis on the results.

IF you have any questions or concerns, please contact me @ tamas.palfi34@gmail.com.  Enjoy!

# Technologies Used:
  - **Python** - language used for coding
  - **freq-e** - to do prevalence estimation with focus on implicit likelihood
  - Training Data/Stanford Model:
     - **pandas** : to read the input CSV file
     - **sklearn** : 
        -**model selection** - to get the train and test split
        -**DictVectorizer** - convert dict from BoW to array for ML fitting
        -**linear_model.LogisticRegression** - to make OvR classifier
        -**metrics: roc_auc, f1_score** - analysis
     - **nltk**: used for BoW - tokenize to get sentences for each post, and word_tokenize to get words for each sentence
     - **re**: to get rid of puncuation in BoW
     - **numpy**: used .mean to find class prior distribution and true prevalence of labels
     
  - Obtaining Reddit Data Set:
    -**linux**
      - many basic operations - **cd**, **ls**, **mkdir**, **cp**, etc..
    -**bash/shell scripting**
        -**grep**
        -**piping**
        -**cat**
    -**Vim** - navigation of file system
        
  - Creating & Applying Model to Reddit Dataset:
     -**pandas** : to read the input CSV file
     -**json & joblib**:  load and dump data from/to a file
     - **nltk**: used for BoW - tokenize to get sentences for each post, and word_tokenize to get words for each sentence
     - **re**: to get rid of puncuation in BoW
     - **numpy**: used .mean to find class prior distribution and true prevalence of labels
      - **sklearn** : 
        -**DictVectorizer** - convert dict from BoW to array for ML fitting
        -**linear_model.LogisticRegression** - to make OvR classifier
        
  - Plotting:
    -**joblib & json** - used to load files
    -**matplotlib** - python's plotting library
  
    
  
        
      


