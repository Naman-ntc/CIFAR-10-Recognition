# Feedforward Neural Networks
Neural networks are a great tool for learning non linear data. We use activation functions to introduce non linearity

### Usage

```python
model = SVMclassifier()
model.add_data(data,label,val_data,val_label)  
# data is mxn dimensional np array while label is 1 dimensional np array
model.InitializePars()
model.set_lr(lr)
model.set_reg(rs)
model.set_bs(bs)
model.GradientDescent(num_epoches)
# Use model.give_prediction(test_data,k) to get prediction for some k
# Use model.give_prediction_and_accuracy(test_data,labels,k) to get dictionary containing both
# prediction_labels and accuracy 
```

### Implementation
It is an inherited class of Generalclassifier. The function InitializePars helps in providing dimensions of hidden layers as well as in changing the standard deviations and mean of neural net parameters.
It performs gradient descent using rate decay over the L2 loss. Decay rate can be manually set up.