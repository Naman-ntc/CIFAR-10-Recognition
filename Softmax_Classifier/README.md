# Softmax Classifier
Softmax regression (or multinomial logistic regression) is a generalization of logistic regression to the case where we want to handle multiple classes.

### Usage

```python
model = Softmaxclassifier()
model.add_data(data,label,val_data,val_label)  
# data is mxn dimensional np array while label is 1 dimensional np array
model.InitializePars()
model.set_lr(lr)
model.set_reg(rs)
model.set_bs(bs)
# Use model.give_prediction(test_data,k) to get prediction for some k
# Use model.give_prediction_and_accuracy(test_data,labels,k) to get dictionary containing both
# prediction_labels and accuracy 
```


### Implementation
It is an inherited class of Generalclassifier. The function InitializePars helps in changing the standard deviations and mean of softmax parameters.
It performs gradient descent using rate decay over the softmax loss. Decay rate can be manually set up.

### References
* [Brilliant Tutorial](http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/)
* [Another Nice Tutorial](https://www.pyimagesearch.com/2016/09/12/softmax-classifiers-explained/)

### Results
Best Validation Accuracy 0.3345 for lr :0.00004000 rs :100000.0000 bs :600.000000 

And Testing Accuracy is 0.3741

### Loss Curves
![Loss Curve](Losses_over_time_Softmax.png)