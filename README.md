# assignment2 : Results of plot_classification

Training examples and ground truth labels
![Figure_1](https://user-images.githubusercontent.com/37702725/132055125-c855713e-6821-4da2-b0f0-205807f3a572.png)

Testing Samples and predictions
![Figure_2](https://user-images.githubusercontent.com/37702725/132055215-43d12a0c-cf20-4076-910c-3e350acd7cab.png)

Confusion Matrix


![Figure_3](https://user-images.githubusercontent.com/37702725/132055247-a81013d0-8dc4-44f8-8ef2-0189f846ba57.png)

# assignment3 : How metrics changes with paramters

Following are the results for different hyperparameters. The results shows how metric(s) vary with the hyperparameter. We used different values of gamma. First column represent gamma value, second value represent accuracy on test data and third column represents F1 score on test data.


![image](https://user-images.githubusercontent.com/37702725/132888764-fdca50b0-d305-4014-bf09-e501da1af724.png)

With different value of gamma we represent how training accuracy and test accuracy varies.  First column represent gamma value, second value represent accuracy on train data and third column represents accuracy on test data.


![image](https://user-images.githubusercontent.com/37702725/132897448-5f3ac184-f59b-4ea5-bcdc-88d6b3cea899.png)

Below graph shows how training and testing accuracy varies with different values of gamma.


![graph](https://user-images.githubusercontent.com/37702725/132897684-6aa155e4-9e5e-4ff1-8511-b4b4f2d9ed7d.png)

The gamma parameter defines how far the influence of a single training example reaches, with low values meaning ‘far’ and high values meaning ‘close’. The gamma parameters can be seen as the inverse of the radius of influence of samples selected by the model as support vectors.The behavior of the model is very sensitive to the gamma parameter. If gamma is too large, the radius of the area of influence of the support vectors only includes the support vector itself and no amount of regularization with C will be able to prevent overfitting.When gamma is very small, the model is too constrained and cannot capture the complexity or “shape” of the data. The region of influence of any selected support vector would include the whole training set.
