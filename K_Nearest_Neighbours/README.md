# K Nearest Neigbours Classifier
KNN is an non parametric lazy learning algorithm. Given any point during testing time we find its k nearest neighbour in feature space and provide results based on the output

### Usage

```python
model = KNNclassifier()
model.add_data(data,label)  # data is mxn dimensional np array while label is mx1 dimensional np array
# Use model.give_prediction(test_data,k) to get prediction for some k
# Use model.give_prediction_and_accuracy(test_data,labels,k) to get dictionary containing both
# prediction_labels and accuracy 
```


### Implementation
KNNclassifier class uses efficient scipy cdist2 function as well as KDTree approach to find the K nearest Neighbours. After finding them using scipy mode we obtain the predictions for each of our test cases. 


### Results
Here are the various results obtained on object recognition using K Nearest Neighbours

![alt text](results.txt)