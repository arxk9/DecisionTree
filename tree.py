import pandas as pd
import math
import random
import operator

class Node:
    def __init__(self, isLeaf):
        self.children = {}
        self.feature = ''
        self.value = ''
        self.isLeaf = isLeaf
        self.ig = -1
        self.bias = ''
    def display(self, edge = '', level=0):
        if self.isLeaf:
            print('\t' * level + edge + "--->" + self.value)
        else:
            print('\t' * level + edge + "--->" + self.feature + ": Information Gain = " + str(self.ig) + " Bias = " + self.bias)
        for path, node in self.children.items():
            node.display(path, level+1)
            
    def pathLengthSum(self, depth=1):
        if self.isLeaf:
            return depth, 1
        return sum([child.pathLengthSum(depth+1)[0] for child in list(self.children.values())]), sum([child.pathLengthSum(depth+1)[1] for child in list(self.children.values())])
    
    def avgPathLength(self):
        temp = self.pathLengthSum()
        return temp[0]/temp[1]
        
            
def entropy(prob_list):
    return -sum([prob*math.log(prob, 2) for prob in prob_list if prob != 0])

def getProbs(data_frame, col = None, thing = None):
    if col is None:
        outcomes = data_frame.iloc[:,-1]
    else:
        subframe = refine(data_frame, col, thing)
        outcomes = subframe.iloc[:,-1]
    dict = {}
    for out in outcomes:
        if out not in dict.keys():
            dict[out] = 1
        else:
            dict[out] += 1
    return [yoy/len(outcomes) for yoy in list(dict.values())], dict

def infogain(data_frame, col):
    return entropy(getProbs(data_frame)[0])-sum([list(data_frame[col]).count(thing)/len(data_frame[col])*entropy(getProbs(data_frame, col, thing)[0]) for thing in set(data_frame[col])])

def info(data_frame, col):
    return sum([list(data_frame[col]).count(thing)/len(data_frame[col])*entropy(getProbs(data_frame, col, thing)[0]) for thing in set(data_frame[col])])

def find_best_feature(data_frame):
    temp = list(data_frame)[:-1]
    f = max(temp, key = lambda x: infogain(data_frame,x))
    ig = infogain(data_frame, f)
    return (f, ig)

def refine(data_frame, col, thing):
    filter = (data_frame[col] == thing)
    return data_frame[filter]

def learn(data_frame):
    if entropy(getProbs(data_frame)[0]) == 0:
        temp = Node(True)
        temp.value = set(data_frame.iloc[:,-1]).pop()
        return temp
    f, ig = find_best_feature(data_frame)
    n = Node(False)
    n.feature = f
    n.ig = ig
    probs = getProbs(data_frame)[1]
    n.bias = max(probs.keys(), key = lambda x:probs[x])
    for thing in set(data_frame[f]):
        subframe = refine(data_frame, f, thing)
        #print("Splitting %s on %s, split is %0.0f : %0.0f" % (f, thing, list(subframe.iloc[:,-1]).count('republican'), list(subframe.iloc[:,-1]).count('democrat')))
        if len(subframe) == 0:
            continue
        else:
            child = learn(subframe)
            n.children[thing] = child
    return n

def classify(tree, features):
    if tree.isLeaf:
        return tree.value
    else:
        if features[tree.feature] == '?':
            return tree.bias
        return classify(tree.children[features[tree.feature]], features)

    
waits = pd.read_csv('restaurants.csv')
tennis_days = pd.read_csv('tennis.csv')
del tennis_days['Day']
votes = pd.read_csv('house-votes-84.csv')
del votes['Label']


votes_train = votes[:350]
for col in votes_train:
    filter = votes_train[col] != '?'
    votes_train = votes_train[filter]
    
votes_test = votes[350:]


tree = learn(votes_train)
tree.display()
print("Average Path Length:", tree.avgPathLength())



plt.plot(x,y)
plt.axis([0,150,0,100])
plt.xlabel('Training Size')
plt.ylabel('Accuracy (%)')
plt.show()
