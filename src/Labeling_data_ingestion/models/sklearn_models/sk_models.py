from sklearn.linear_model import LogisticRegression, SGDClassifier
from Labeling_data_ingestion.train.train_sklearn import train
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes  import ComplementNB
from sklearn.svm import LinearSVC



def logistic_model():
    
    return  LogisticRegression(
    l1_ratio=0,
    C=10.0,
    solver="saga",
    max_iter=1000,
    class_weight="balanced",
    random_state=42
    )



def MultiNomialNB_model():
    
    return MultinomialNB(
    alpha = 0.65
    )



def ComplementNB_model():

    return ComplementNB(alpha=0.005)



def SGDClassifier_model():

    return SGDClassifier(loss="log_loss",
    class_weight="balanced",
    max_iter=1000,
    tol=1e-3,
    alpha=0.0001,
    random_state=42
    )

def linearSVC_model():

    return LinearSVC(
    penalty="l2",
    loss="squared_hinge",
    dual=False,
    tol=1e-3,
    C=10.0,
    class_weight="balanced",
    max_iter=1000,
    random_state=42
    )

def SGDClassifier_hinge():
    return SGDClassifier(
    loss ="hinge",  
    alpha=0.0001,
    max_iter=1000,
    class_weight="balanced",
    random_state=42
    )


result = train(SGDClassifier_hinge(), 0.15)
print(result[2])

#Logistic regression

#[[113  60]
#[ 15  67]]


#              precision    recall  f1-score   support

#           0       0.88      0.65      0.75       173
#           1       0.53      0.82      0.64        82

#    accuracy                           0.71       255
#   macro avg       0.71      0.74      0.70       255
#weighted avg       0.77      0.71      0.72       255

#0.7058823529411765


#MultiNomialNB_model

#[[122  51]
# [ 18  64]]


#              precision    recall  f1-score   support

#           0       0.87      0.71      0.78       173
#           1       0.56      0.78      0.65        82

#    accuracy                           0.73       255
#   macro avg       0.71      0.74      0.71       255
#weighted avg       0.77      0.73      0.74       255

#0.7294117647058823



#ComplementNB_model

#[[115  58]
# [ 20  62]]


#              precision    recall  f1-score   support

#           0       0.85      0.66      0.75       173
##           1       0.52      0.76      0.61        82

#    accuracy                           0.69       255
#   macro avg       0.68      0.71      0.68       255
#weighted avg       0.74      0.69      0.70       255

#0.6941176470588235



#SGDClassifier

#[[128  45]
 #[ 21  61]]


  #            precision    recall  f1-score   support

#           0       0.86      0.74      0.80       173
#           1       0.58      0.74      0.65        82

#    accuracy                           0.74       255
#   macro avg       0.72      0.74      0.72       255
#weighted avg       0.77      0.74      0.75       255


#SGDClassifier_Hinge


#              precision    recall  f1-score   support

#           0       0.90      0.80      0.84       173
#           1       0.65      0.80      0.72        82

#    accuracy                           0.80       255
#   macro avg       0.77      0.80      0.78       255
#weighted avg       0.82      0.80      0.80       255

#0.8