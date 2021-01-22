import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class NB_model():
    def __init__(self): 
        self.pi = {} # to store prior probability of each class 
        self.Pr_dict = None
        self.num_vocab = None
        self.num_classes = None
    
    def fit(self, train_data, train_label, vocab, if_use_smooth=True):
        # get prior probabilities
        self.num_vocab = len(vocab['index'].tolist())
        self.get_prior_prob(train_label)
        # ================== YOUR CODE HERE ==========================
        # Calculate probability of each word based on class 
        # Hint: Store each probability value in matrix or dict: self.Pr_dict[classID][wordID] or Pr_dict[wordID][classID])
        # Remember that there are possible NaN or 0 in Pr_dict matrix/dict. Use smooth method

        if if_use_smooth:
            Pr_dict = np.ones((self.num_classes, self.num_vocab))
            div = self.num_vocab * np.ones((self.num_classes, 1))
        else:
            Pr_dict = np.zeros((self.num_classes, self.num_vocab))
            div = np.zeros((self.num_classes, 1))
    
        for i, (docIdx, wordIdx, count, classIdx) in train_data.iterrows():
            Pr_dict[classIdx - 1][wordIdx - 1] += count
            div[classIdx - 1] += count

        self.Pr_dict = Pr_dict / div

        # self.Pr_dict = np.ones((self.num_classes, self.num_vocab)) if if_use_smooth else np.zeros((self.num_classes, self.num_vocab))
        # denom = np.ones(self.num_classes) * self.num_vocab if if_use_smooth else np.zeros(self.num_classes)
        # vals = train_data.values
        # for row in vals:
        #     self.Pr_dict[row[3]-1][row[1]-1] += row[2]
        #     denom[row[3]-1] += row[2]
        # self.Pr_dict = self.Pr_dict / denom[:,None]
        # ============================================================
        print("Training completed!")
    
    def predict(self, test_data):
        test_dict = test_data.to_dict() # change dataframe to dict
        new_dict = {}
        prediction = []
        
        for idx in range(len(test_dict['docIdx'])):
            docIdx = test_dict['docIdx'][idx]
            wordIdx = test_dict['wordIdx'][idx]
            count = test_dict['count'][idx]
            try: 
                new_dict[docIdx][wordIdx] = count 
            except:
                new_dict[test_dict['docIdx'][idx]] = {}
                new_dict[docIdx][wordIdx] = count
                ''
        for docIdx in range(1, len(new_dict)+1):
            score_dict = {}
            #Creating a probability row for each class
            for classIdx in range(1,self.num_classes+1):
                score_dict[classIdx] = 0
                # ================== YOUR CODE HERE ==========================
                ### Implement the score_dict for all classes for each document
                ### Remember to use log addtion rather than probability multiplication
                ### Remember to add prior probability, i.e. self.pi
                log_likelihood = np.log(self.pi[classIdx])
                log_likelihood += sum(
                        count * np.log(self.Pr_dict[classIdx-1][wordIdx-1]) 
                        for wordIdx, count in new_dict[docIdx].items()
                    )

                score_dict[classIdx] = log_likelihood
                # ============================================================
            max_score = max(score_dict, key=score_dict.get)
            prediction.append(max_score)
        return prediction
                    
    
    def get_prior_prob(self,train_label, verbose=True):
        unique_class = list(set(train_label))
        self.num_classes = len(unique_class)
        total = len(train_label)
        for c in unique_class:
            # ================== YOUR CODE HERE ==========================
            ### calculate prior probability of each class ####
            ### Hint: store prior probability of each class in self.pi
            count = 0
            for label in train_label:
                if c == label:
                    count += 1
            self.pi[c] = count / total
            # ============================================================
        if verbose:
            print("Prior Probability of each class:")
            print("\n".join("{}: {}".format(k, v) for k, v in self.pi.items()))
