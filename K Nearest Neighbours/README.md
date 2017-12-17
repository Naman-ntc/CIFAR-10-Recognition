# K Nearest Neigbours Classifier

KNNclassifier class uses efficient scipy cdist2 function as well as KDTree approach to find the K nearest Neighbours. After finding them using scipy mode we obtain the predictions for each of our test cases. 

Refer to knn_test.py file on usage.
```python
model = KNNclassifier()
model.add_data(data,label)  # data is mxn dimensional np array while label is mx1 dimensional np array
# Use model.give_prediction(test_data,k) to get prediction for some k
# Use model.give_prediction_and_accuracy(test_data,labels,k) to get dictionary containing both
# prediction_labels and accuracy 
```

Here are the various results obtained on object recognition using K Nearest Neighbours

For k = 1 the accuracy mean is 0.28044432
For k = 2 the accuracy mean is 0.28044414
For k = 3 the accuracy mean is 0.28042001
For k = 4 the accuracy mean is 0.28045532
For k = 5 the accuracy mean is 0.28150000
For k = 8 the accuracy mean is 0.28150000
For k = 10 the accuracy mean is 0.28150000
For k = 15 the accuracy mean is 0.28150000
For k = 20 the accuracy mean is 0.28150000
For k = 25 the accuracy mean is 0.28044443
For k = 50 the accuracy mean is 0.28044443
For k = 75 the accuracy mean is 0.28044443
For k = 100 the accuracy mean is 0.28044443
For k = 200 the accuracy mean is 0.28044443
For k = 250 the accuracy mean is 0.28044443
For k = 500 the accuracy mean is 0.28044443
For k = 1000 the accuracy mean is 0.28044443
For k = 5000 the accuracy mean is 0.28044443
