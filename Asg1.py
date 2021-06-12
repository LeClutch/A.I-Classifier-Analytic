from __future__ import division
from codecs import open
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_validate, cross_val_predict, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score , recall_score, f1_score, confusion_matrix

#Task #0-A
def read_documents(doc_file):
 docs = []
 labels = []
 with open(doc_file, encoding='utf-8') as f:
    for line in f:
        s = " "
        words = line.strip().split()
        docs.append(s.join(words[3:]))
        labels.append(words[1])
 return docs, labels




all_docs_preprocess, all_labels_preprocess = read_documents('all_sentiment_shuffled.txt')

all_labels = []

for item in all_labels_preprocess:

    if item == "neg":
        all_labels.append('1')
    elif item == "pos":
        all_labels.append('0')

print(all_labels[0:5])
all_docs = all_docs_preprocess #COMMENT THIS LINE IF YOU WANT TO USE THE PROCESSING LOOP



#Task #0-B
split_point = int(0.80*len(all_docs))
train_docs = all_docs[:split_point]
train_labels = all_labels[:split_point]
eval_docs = all_docs[split_point:]
eval_labels = all_labels[split_point:]

# print(all_labels.count('1'))
# print(all_labels.count('0'))

##################################################################################
##################################################################################
# Task #1

df = pd.DataFrame({'Label':['neg', 'pos'], 'Sentiment Polarity Label ':[all_labels.count('1'), all_labels.count('0')]})
ax = df.plot.bar(x='Label', rot=0)
plt.ylim([0, len(all_docs)])
plt.ylabel('Comments/Reviews')
ax.set_title('Distribution of the number of the instances in each class')
plt.show()


##################################################################################


# Task #2
vectorizer = TfidfVectorizer(stop_words='english')
train_docs_vectors =vectorizer.fit_transform(train_docs)
eval_docs_vectors =vectorizer.transform(eval_docs)

print(train_docs_vectors.toarray())

# Task #2-A Naive Bayes
################################# [Naive Baye's] #################################
naiveBML = MultinomialNB()
naiveBML.fit(train_docs_vectors,train_labels)

# print(naiveBML, "\n")
# print(eval_docs[0], "\n")

labelsPredic = naiveBML.predict(eval_docs_vectors)

# print(train_docs[11]) #print line 12
# print(listlabelspredicted[10])
# print(eval_labels[10])
# print("\n")
# print(listlabelspredicted[11])
# print(eval_labels[11])
# print("\n")
# print(listlabelspredicted[12])
# print(eval_labels[12])
# print("\n")

#Task 4
listlabelspredicted = labelsPredic.tolist()
i=0
misclassifiedNB = []
for x in eval_labels:
    if x != listlabelspredicted[i]:
        misclassifiedNB.append(i)
    i = i + 1
print(misclassifiedNB)


file = open("NB-dataset.txt", "w")

#Task 3-A
#method 1
counter = len(all_docs) - len(eval_docs)+1
results = []
for item in labelsPredic:
    comb = counter,item
    results.append(comb)
    counter+=1

print(results)

y=0
count=0
for y in listlabelspredicted:
    file.write(str(count))
    file.write(" , ")
    file.write(y)
    file.write("\n")
    count +=1


nbf1score = f1_score(eval_labels, labelsPredic, average=None, labels=['0', '1']) #F1 SCORE
nbRS = recall_score(eval_labels, labelsPredic, average= None,labels=['0', '1'])  # RECALL SCORE
nbPS = precision_score(eval_labels, labelsPredic, average= None,labels=['0', '1']) # PRECISION SCORE

#CALCULATES AND PRINTS THE NUMBER OF DOCUMENTS CLASSIFIED CORRECTLY
nb_correct = (eval_labels == labelsPredic).sum()

file.write(f'{nb_correct} documents classified correctly')
file.write("\n")

file.write('Accuracy Score: ')
file.write("\n")
file.write(str(accuracy_score(eval_labels, labelsPredic)))# 1st way of getting accuracy
# print(naiveBML.score(eval_docs_vectors,eval_labels)) #2nd way of getting accuracy

file.write("\n")
file.write('F1 Score: pos ; neg ')
file.write("\n")
file.write(str(nbf1score))
file.write("\n")
file.write('Recall Score: pos ; neg ')
file.write("\n")
file.write(str(nbRS))
file.write("\n")
file.write('Precision Score: pos ; neg ')
file.write("\n")
file.write(str(nbPS))


labels = ['0','1']
cm = confusion_matrix(eval_labels, labelsPredic, labels=labels)
labels = ['pos','neg']
df_cm = pd.DataFrame(cm, index=labels, columns=labels)
sn.heatmap(df_cm, annot=True, fmt='d')
file.write("\n")
file.write('Confusion Matrix: ')
file.write("\n")
file.write(str(cm))
file.write("\n")
file.close()
plt.show()



##################################################################################
############################ [Baseline Decision Tree] ############################

classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(train_docs_vectors, train_labels)

classifier.predict(eval_docs_vectors)


labelsPredic = classifier.predict(eval_docs_vectors)

#Task 4
listlabelspredicted = labelsPredic.tolist()
i=0
misclassifiedBaseDT = []
for x in eval_labels:
    if x != listlabelspredicted[i]:
        misclassifiedBaseDT.append(i)
    i = i + 1
print(misclassifiedBaseDT)
file = open("BaseDT-dataset.txt", "w")

#Task 3-A
#method 1
counter = len(all_docs) - len(eval_docs)+1
results = []
for item in labelsPredic:
    comb = counter,item
    results.append(comb)
    counter+=1

print(results)

y=0
count=0
for y in listlabelspredicted:
    file.write(str(count))
    file.write(" , ")
    file.write(y)
    file.write("\n")
    count +=1

nbf1score = f1_score(eval_labels, labelsPredic, average=None, labels=['0', '1']) #F1 SCORE
nbRS = recall_score(eval_labels, labelsPredic, average= None,labels=['0', '1'])  # RECALL SCORE
nbPS = precision_score(eval_labels, labelsPredic, average= None,labels=['0', '1']) # PRECISION SCORE

#CALCULATES AND PRINTS THE NUMBER OF DOCUMENTS CLASSIFIED CORRECTLY
nb_correct = (eval_labels == labelsPredic).sum()

file.write(f'{nb_correct} documents classified correctly')
file.write("\n")

file.write('Accuracy Score: ')
file.write("\n")
file.write(str(accuracy_score(eval_labels, labelsPredic)))# 1st way of getting accuracy
# print(classifier.score(eval_docs_vectors,eval_labels)) #2nd way of getting accuracy

file.write("\n")
file.write('F1 Score: pos ; neg ')
file.write("\n")
file.write(str(nbf1score))
file.write("\n")
file.write('Recall Score: pos ; neg ')
file.write("\n")
file.write(str(nbRS))
file.write("\n")
file.write('Precision Score: pos ; neg ')
file.write("\n")
file.write(str(nbPS))


labels = ['0','1']
cm = confusion_matrix(eval_labels, labelsPredic, labels=labels)
labels = ['pos','neg']
df_cm =  pd.DataFrame(cm, index=labels, columns=labels)
sn.heatmap(df_cm, annot=True, fmt='d')
file.write("\n")
file.write('Confusion Matrix: ')
file.write("\n")
file.write(str(cm))
file.write("\n")
file.close()
plt.show()


# Task #2-C Best Decision Tree
############################## [Best Decision Tree] ##############################

classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(train_docs_vectors, train_labels)

###################### [Improvement Algorithm With Pruning] ######################

# Reference: https://stackoverflow.com/questions/49428469/pruning-decision-trees
from sklearn.tree._tree import TREE_LEAF
def prune_index(inner_tree, index, threshold):
    if inner_tree.value[index].min() < threshold:
        # turn node into a leaf by "unlinking" its children
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF
    # if there are children, visit them as well
    if inner_tree.children_left[index] != TREE_LEAF:
        prune_index(inner_tree, inner_tree.children_left[index], threshold)
        prune_index(inner_tree, inner_tree.children_right[index], threshold)

print(sum(classifier.tree_.children_left < 0))
# start pruning from the root
prune_index(classifier.tree_, 0, 150)
sum(classifier.tree_.children_left < 0)

##################################################################################

labelsPredic = classifier.predict(eval_docs_vectors)
#Task 4
listlabelspredicted = labelsPredic.tolist()
i=0
misclassifiedBestDT = []
for x in eval_labels:
    if x != listlabelspredicted[i]:
        misclassifiedBestDT.append(i)
    i = i + 1
print(misclassifiedBestDT)

file = open("BestDT-dataset.txt", "w")

#Task 3-A
y=0
count=0
for y in listlabelspredicted:
    file.write(str(count))
    file.write(" , ")
    file.write(y)
    file.write("\n")
    count +=1


nbf1score = f1_score(eval_labels, labelsPredic, average=None, labels=['0', '1']) #F1 SCORE
nbRS = recall_score(eval_labels, labelsPredic, average= None,labels=['0', '1'])  # RECALL SCORE
nbPS = precision_score(eval_labels, labelsPredic, average= None,labels=['0', '1']) # PRECISION SCORE

#CALCULATES AND PRINTS THE NUMBER OF DOCUMENTS CLASSIFIED CORRECTLY
nb_correct = (eval_labels == labelsPredic).sum()
file.write(f'{nb_correct} documents classified correctly')
file.write("\n")

file.write('Accuracy Score: ')
file.write("\n")
file.write(str(accuracy_score(eval_labels, labelsPredic)))# 1st way of getting accuracy
# print(classifier.score(eval_docs_vectors,eval_labels)) #2nd way of getting accuracy

file.write("\n")
file.write('F1 Score: pos ; neg ')
file.write("\n")
file.write(str(nbf1score))
file.write("\n")
file.write('Recall Score: pos ; neg ')
file.write("\n")
file.write(str(nbRS))
file.write("\n")
file.write('Precision Score: pos ; neg ')
file.write("\n")
file.write(str(nbPS))

labels = ['0','1']
cm = confusion_matrix(eval_labels, labelsPredic, labels=labels)
labels = ['pos','neg']
df_cm = pd.DataFrame(cm, index=labels, columns=labels)
sn.heatmap(df_cm, annot=True, fmt='d')
file.write("\n")
file.write('Confusion Matrix: ')
file.write("\n")
file.write(str(cm))
file.write("\n")
file.close()
plt.show()



