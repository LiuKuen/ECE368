import os.path
import numpy as np
#import matplotlib.pyplot as plt
import util
import collections

def get_all_unique_words(files):
    ret = set()
    for f in files:
        words = util.get_words_in_file(f)
        #print(set(words))
        ret = ret.union(set(words))
        #print(ret)
    return ret

def get_estimates(unqiue_words, files):
    ret = dict()

    counter = util.get_counts(files)
    total_words = 0
    for word in counter:
        total_words += counter[word]

    for word in unqiue_words:
        ret[word] = (counter[word] + 1) / (total_words + len(unqiue_words))

    return ret

def learn_distributions(file_lists_by_category):
    """
    Estimate the parameters p_d, and q_d from the training set
    
    Input
    -----
    file_lists_by_category: A two-element list. The first element is a list of 
    spam files, and the second element is a list of ham files.

    Output
    ------
    probabilities_by_category: A two-element tuple. The first element is a dict 
    whose keys are words, and whose values are the smoothed estimates of p_d;
    the second element is a dict whose keys are words, and whose values are the 
    smoothed estimates of q_d 
    """
    ### TODO: Write your code here
    
    spam_files = file_lists_by_category[0]
    ham_files = file_lists_by_category[1]

    spam_unique_words = get_all_unique_words(spam_files)
    ham_unique_words = get_all_unique_words(ham_files)
    unqiue_words = spam_unique_words.union(ham_unique_words)

    p_d_estimates = get_estimates(unqiue_words, spam_files)
    q_d_estimates = get_estimates(unqiue_words, ham_files)

    probabilities_by_category = (p_d_estimates, q_d_estimates)
    return probabilities_by_category

def get_log_pxy(probabilities_by_category, y, words):
    #sadly y = 0 is p_d's and y = 1 is q_d's
    category = probabilities_by_category[y]

    ret = 0
    for word in category:
        #print(ret)
        if word in words:
            ret += np.log(category[word])
        else:
            ret += np.log((1-category[word]))

    return ret

def classify_new_email(filename,probabilities_by_category,prior_by_category):
    """
    Use Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    filename: name of the file to be classified
    
    probabilities_by_category: output of function learn_distributions
    
    prior_by_category: A two-element list as [\pi, 1-\pi], where \pi is the 
    parameter in the prior class distribution

    Output
    ------
    classify_result: A two-element tuple. The first element is a string whose value
    is either 'spam' or 'ham' depending on the classification result, and the 
    second element is a two-element list as [log p(y=1|x), log p(y=0|x)], 
    representing the log posterior probabilities
    """
    ### TODO: Write your code here
    prior_0 = prior_by_category[0]
    prior_1 = prior_by_category[1]

    words = util.get_words_in_file(filename)
    unique_words = set(words)

    p_xy0 = get_log_pxy(probabilities_by_category, 0, words) + np.log(prior_0)
    p_xy1 = get_log_pxy(probabilities_by_category, 1, words) + np.log(prior_1)

    #print(p_xy0, p_xy1)
    #p_xy0 contains p_d, p_xy1 contains q_d therefore we need to swap
    res = 'spam' if p_xy0 > p_xy1 else 'ham'
    classify_result = (res, (p_xy0, p_xy1))

    return classify_result

if __name__ == '__main__':
    
    # folder for training and testing 
    spam_folder = "data/spam"
    ham_folder = "data/ham"
    test_folder = "data/testing"

    # generate the file lists for training
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))
        
    # Learn the distributions    
    probabilities_by_category = learn_distributions(file_lists)
    
    # prior class distribution
    priors_by_category = [0.5, 0.5]
    
    # Store the classification results
    performance_measures = np.zeros([2,2])
    # explanation of performance_measures:
    # columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    # to be more clear, performance_measures = [[p1 p2]
    #                                           [p3 p4]]
    # p1 = Number of emails whose true label is 'spam' and classified as 'spam' 
    # p2 = Number of emails whose true label is 'spam' and classified as 'ham' 
    # p3 = Number of emails whose true label is 'ham' and classified as 'spam' 
    # p4 = Number of emails whose true label is 'ham' and classified as 'ham' 

    # Classify emails from testing set and measure the performance
    for filename in (util.get_files_in_folder(test_folder)):
        # Classify
        label,log_posterior = classify_new_email(filename,
                                                 probabilities_by_category,
                                                 priors_by_category)
        
        # Measure performance (the filename indicates the true label)
        base = os.path.basename(filename)
        true_index = ('ham' in base) 
        guessed_index = (label == 'ham')
        performance_measures[int(true_index), int(guessed_index)] += 1

    template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0],totals[0],correct[1],totals[1]))
    print(performance_measures[0,1])
    print(performance_measures[1,0])
    
    
    ### TODO: Write your code here to modify the decision rule such that
    ### Type 1 and Type 2 errors can be traded off, plot the trade-off curve
   

 