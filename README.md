## Flower-Recognition-Model
Flower recognition with CNN using Tensorflow & Keras


##The goal of the project

We have a dataset of around 4000 flower and the goal is to make a CNN that is able to recognize the different types of flower

##The different CNN

**no_dropout.py**
In order to make the best neural network, we have done few step and each time we tried to increase the efficientie of our models.
First we tried a simple CNN classifier with 4 pair of Conv2D and MaxPooling 2D.

**train_dropout.py**
We noticed that we have an overfitting problem, so we added dropouts to try to reduce the overfitting. It work pretty well.

**finetunning.py**
We though that we can increase the accuracy of our model using fine tunning with VGG16 by freezing the last 5 convolution blocks,
this model take much time to train, but the result are way better ! Around ~15% more accurate !

>**Note:** Each model is saved in the **model** directory, indeed the plots are also save in the **plots** directory.

##The loader.py file

The loader.py allow us to try the models with testing data, you just have to change in the file the path of the model that you want to try.
The output will be a confusion matrix plot and a classification report of the result of the predictions
