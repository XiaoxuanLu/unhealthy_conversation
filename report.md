# Report

My narrative of doing the assignment is including in the model.ipynb. 
I mainly use ensemble learning model to predict healthy comments.
I only use attributes to classify comment. I combine the attributes with
their confidence level making them into a new dataset ranging from 0-1. 
I first combine some classifiers into my voting ensemble model, and I use hard voting
and soft voting to see which one performs better. I compare the voting classifier
with the classifier in itself, finding that voting is not the best. Then 
I train bagging decision trees, out of bag evaluation, random forest, and 
Extra trees, and find bagging decision trees and random forest perform great. Then
I use voting again to combine the three bagging models, getting 
a new voting model, which does not perform as good as my first hard voting model.
Then I use boosting algorithm, Adaboost, gradient boost, and XGB boost,
XGB boost performs best, and I use hard voting combine all boosting models. I find 
voting inside the three does not make voting much better. It might because voting algorithm
performs better if the classifiers inside have really different algorithms, thus I wonder 
I could choose some best classifier to include in the voting classifier.
Thus I finally create a final hard voting model with SVM, random forest, bagging decision trees, 
and XGB boost, which increase my predictions. 


