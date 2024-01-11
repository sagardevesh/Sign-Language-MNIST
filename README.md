# Sign-Language-MNIST
This project focusses on evaluating the accuracy of CNN models in detecting hand gestures in the Sign Language MNIST dataset.

Here is the link to the [Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) dataset.

### Visualisation of the dataset

The first task was to import the train and test datasets which were basically images of hand gestures used to portray English alphabets. For the descriptive analysis of the dataset, I plotted a histogram, which showed the count of the images of each alphabet portrayed by hand gestures. The numbers 1-24 represent the English alphabets, except the letters j and z because they require motion movements. Below is the figure of the histogram obtained.

![image](https://github.com/sagardevesh/Sign-Language-MNIST/assets/25725480/b25d1830-2cb4-42e6-b95b-92e9b9f12b94)

Another way of portraying the count of each alphabet is the count plot. I obtained the below figure by using the count plot feature of seaborn.

![image](https://github.com/sagardevesh/Sign-Language-MNIST/assets/25725480/3af8bb47-a22e-4c68-8a4a-8e586e2cbedc)

For an even better visual representation of the count of each alphabet, I used the plotly module of python to display the count in the form of a pie chart below. The numbers 0-25 represent the alphabets of the English language. Below is the figure obtained. [1]

![image](https://github.com/sagardevesh/Sign-Language-MNIST/assets/25725480/4d4d2696-c3cd-4040-bf12-e4094bef39cb)


### (i) Description of the dataset:

The dataset we are using for this assignment is the Sign Language MNIST dataset, which has a collection of 28x28 images of hands depicting English alphabets. The hand images are depicting only 24 alphabets out of the 26. Letters ‘j’ and ‘z’ are not included in the dataset, as depicting these two letters involve motion. Using Convolution Neural Network, our objective would be to accurately detect and classify each American Sign Language alphabet images.

The format of the dataset is in close resemblance with the original MNIST dataset. The training and test data are approximately half the size of the original MNIST dataset. It basically consists of 27, 455 training images and 7,712 test images. Each image is converted into grayscale and has a 28x28 structure.

### (ii) Pre-processing
As first part of the pre-processing step, I separated the labels from rest of the dataset. This was done for both train and test data. Next, I plotted and converted the images into gray-scale for both train and test datasets. Below are the figures for images in original format and in grayscale format.

![image](https://github.com/sagardevesh/Sign-Language-MNIST/assets/25725480/6e48c081-ffaf-4c45-b2c2-104cd2701cf4)
RGB format 

![image](https://github.com/sagardevesh/Sign-Language-MNIST/assets/25725480/0865192d-bbc8-48f3-a660-17eb9bcb12c8)
Grayscale format

Gray scaling is done so that the model does not take longer when images are fed into them. In gray-scaling, a certain pixel value will be one-dimensional instead of three-dimensional, and it will just vary from 0 to 255. This will result in some information loss in terms of actual colours, but will eventually result in model running a lot faster.

Once the image was converted into gray scale, I implemented histogram equalisation to adjust the contrast of an image by modifying the intensity distribution of the histogram. Below is the image comparison of before and after histogram equalisation was applied. [7]

![image](https://github.com/sagardevesh/Sign-Language-MNIST/assets/25725480/bbf7fdce-2fe4-48af-8a36-b438499e6697)
**Before histogram equalisation **

![image](https://github.com/sagardevesh/Sign-Language-MNIST/assets/25725480/97e209ea-e135-477c-b82c-c632f28226ef)
**After histogram equalisation**

Next, I used Label Binarizer module to one hot encode the labels for both train and test data. Label Binarizer is an Scikit Learn class that accepts categorical data as input and returns an Numpy array. As a next part of pre-processing, I performed normalisation for train and test input data by dividing each value in both the datasets by 255. Normalisation is used to standardise the data. This means to bring different sources of data inside the same range.

### (iii) Splitting of the dataset:
We are aware that the original dataset had already come in partitions of train and test data. The train dataset consisted of 27, 455 images and test dataset consisted of 7,712 images. Our next task was to further divide the train dataset into train and validation sets. I performed this split in 80-20 ratio, where train set now consisted of 80% of the images, whereas the validation set consisted of 20% of the images.

While training the model, I will use the new train dataset to train the model, and the validation dataset to validate the results. Once the model was trained, I took the test set and made the model predict the labels for each test image.

## Task 2: Description

In this task, I started implementing CNN models with the pre-processed data obtained in the previous task. I took the batch size to be 128, number of classes as 24 and epochs as 10. The optimizer used was Adam optimizer. When I trained the model and tested it on test data, it gave an accuracy of 90.5%. In the next step, I changed the architecture and added batch normalisation which gave me slightly different results. Next, I removed a dropout layer to see the effects. The detailed results of the above operations are explained in (iii) question below.

### (i) What is convolution?
A convolution refers to the mathematical combination of any 2 functions to produce a third function. Basically, it refers to the task of merging 2 sets of information. In case of Convolutional Neural Networks, convolution is applied to the input data to the to filter the information and produce a feature map. This filter is also called a kernel, or feature detector, and its dimensions can be, let’s say, 2x2 [3]. In our case, I have taken the kernel size to be 3x3 for all the models.

### (ii) Selection of an evaluation metric and explaination about why I selected that:-
I have used accuracy as a metric to evaluate and compare different CNN models. Accuracy is a very useful and an effective metric when all the classes are of equal importance. It is calculated as the ratio between the number of correct predictions to the total number of predictions [4].
. Furthermore, accuracy is a metric which is easy to calculate, easy to interpret, and is a single number to summarize the model’s performance.

### (iii) Describe the obtained results:
For this task, I ran 3 models after making the necessary changes as per the task requirement. The optimizer used in all the 3 models in this task was the Adam optimizer. All the models were used for 3 epochs. In the first model, I initialised the model, and added two convolutional layers. The first convolutional layer has a filter size of 32, while the second convolutional layer has a filter size of 64. 

Then I added a Max pooling layer to reduce the dimensions of the feature maps. This layer basically reduces the number of parameters to learn and reduces computation in the neural network. This model also has a dropout layer which is ideally used to prevent overfitting of the model. As I used accuracy as the evaluation metric, the accuracy for this model obtained on test data was 90.51%.

In the second model, I added batch normalisation. Batch normalisation is a normalization technique done between the layers of a neural network instead of in raw data. It is applied on small batches instead of the raw data [5]. It is used to speed up model training and make learning easier. Because the normalisation is computed over mini batches, model’s data distribution sees each time it has some noise. This acts as a regularizer and helps prevent overfitting. In general, batch normalized models achieve a higher accuracy on train and test data, and I observed a similar effect on my model. The second model’s accuracy after batch normalisation went up to 91.39%, as compared to previous model’s accuracy of 90.51%.

In the third model, I removed one of the dropout layers from the model. In a neural network model, a dropout layer is a mask that nullifies the effect of some neurons towards the next layer. This helps in preventing overfitting of the model. In our case, I removed a dropout layer after the second convolution layer. Once I removed the dropout layer and trained the model again, I got an accuracy of 94.91%, which is significantly more than the accuracy obtained in the previous model. This implies that the removal of the dropout layer has led to the model overfitting, and hence the increased accuracy.

### (iv) Analysis on the results, such as overfitting/underfitting. Which model provide the best performance? (Using the test set)
As discussed above, we saw an example of overfitting in the third model. Dropout layer helps in preventing overfitting of the model. Therefore, as soon as we removed the dropout layer from the model, it led to its overfitting.

Overall, we can say that the second model has provided the best performance. We obtained the second model by adding batch normalisation to the first model. This increased the accuracy of the second model from 90.51% to 91.39%. Even though the third model provided the best accuracy of 94.91%, we won’t consider that to be the best model because the increased accuracy was a result of overfitting, caused due to the removal of one dropout layer. All the above results were obtained after testing the models on the test data.

### (v) Conclusion and insights on these models
We can conclude that the use of batch normalisation and dropout layers play a very significant role in determining the performance of a model on test data. In our case, adding batch normalisation increased the performance of the model. Dropout layer helped in preventing overfitting in our models. These are some factors other than the optimizers that affect the model performance. In all these models so far, the optimizer used was the Adam optimizer. In the upcoming models, I will change the optimizers and see what each optimizer brings to the table in terms of model performance. For the models to take less time, we had also performed gray scaling of the images that were to be input into these models. Gray scaling does lead to loss of some information, but it is usually a trade-off between obtaining an acceptable model accuracy and not loosing out on too much information during pre-processing of the dataset.

## Task 3: In this task, you have to do experiments changing the optimizer and its parameters using the best model you found in task 2.
**Description:**

For this task in hand, I went ahead and changed the optimizer in the best model obtained in the previous task. As discussed earlier, I chose model 2 to be the best model out of the 3 with an accuracy of 91.39%. For this model, I made changes to the optimizer. I used 3 different optimizers one by one in model 2, and noticed the effects of each of them on the model, which is explained below.

### (i) Description the obtained results
Firstly, I used the Stochastic gradient descent optimizer in model 2. After training the model, when I tested it on test data, I got an accuracy of 90.28%. I used the default learning rate for SGD optimizer, which is 0.01. For all the above and subsequent models, I have used the default learning rate of that respective optimizer.

Next, I used the Adagrad optimizer to test the model. I used the default learning rate of the Adagrad optimizer, which is 0.01. With the Adagrad optimizer, I got the model accuracy of 82.93%, which is significantly lower than the SGD optimizer.
Lastly, I trained the model with the Adamax optimizer. With this optimizer, I got the highest accuracy of 90.88%, although it is marginally higher than the model accuracy we obtained when we used the SGD optimizer.

### (ii) Analysis, conclusions and insights in terms of the optimization
Just to reiterate the performance obtained with the default optimizer Adam, we got an accuracy of 91.39%, which is the best among the 4 optimizers. Traditionally, Adam performs the best among all the optimizers. When we compare it with SGD, Adam performs better, because SGD is more locally unstable and is more likely to converge to the minima at the flat or asymmetric basins which often have better generalization performance over other type minima. However, usually SGD performs better than Adam and all the other optimizers in image classification tasks, but our case seems to be an exception as the performance of Adam optimizer is marginally better than that of SGD optimizer. However, for tasks other than image classification, Adam is considered to be the best optimizer, because it trains the neural network model in less time and more efficiently.

## Task 4:
### Description:
In task 4, I did further experiments to see how the model accuracy can be improved. I tried with adding a convolution layer, and changing the epochs and the batch size, the results of which are explained in (ii). There could have been a lot more experiments like changing the train-test split ratio, and doing some further image modifications, which may have resulted in an increased accuracy. However, for this assignment, I have tried changing the architecture and the batch size/epochs.

### (i) Approaches that might improve the model:-

Most of the models that I have used has 2 convolutional layers, or 3 convolutional layers at most. Sometimes, a model with just 2 convolutional layers can be too shallow to learn and differentiate between the different classes. Therefore, increasing the number of
convolutional layers might increase the model performance and help the network identify more meaningful patterns. However, making networks overly deep and complex may also turn out to be a pitfall [6].

Another approach that can improve the model performance is by decreasing the learning rate. Sometimes a model tends to overfit very quickly within a few epochs. Lowering the learning rate might not eliminate overfitting completely, but will surely slower the process of the model reaching towards overfitting. This would definitely help us find an epoch with better overall performance before overfitting takes place.

One approach we can take in the initial stages is to obtain the best train test or train validation (in our case) split to get the best performance out of the model. I took a train-validation split of 80-20 split throughout, and that effectively gave good model performance.

### (ii) Results

For this task, I tried to make some changes in the architecture of the model. I added a third convolution layer after max pooling. The optimizer used was Adam with default learning rate. After making this change in the architecture, the model gave an increased accuracy of 94.75%, as compared to previous model with just 2 convolution layers, which gave an accuracy of 91.39%. So, adding an extra convolution layer did improve the model performance!

Next change I made to the original model was to increase the number of epochs to 15 and the batch size to 1024. Originally the number of epochs was 10 and the batch size was 128. Increasing the epochs and batch size to the above numbers also increased the model accuracy from 91.39% to 92.67%. Therefore, this is another change which improved the model performance.

### (iii) Conclusions and insights:

We noticed the effect of changing several parameters on the model’s accuracy. Each parameter plays a vital role in determining how a model performs eventually. We saw a lot of examples and conducted several experiments such as changing the architecture of the neural network, changing batch size and the epochs, and changing the optimizers. Use of different optimizers gave different results. Adam optimizer in general provides the best results, but for image classification tasks, Stochastic gradient descent optimizer has proven itself to be the best optimizer. Overall, it must be kept in mind that there is always a trade-off between accuracy of the model and the time the model takes to run.

## Summary:
Firstly, we pre-processed the data that was to be fed into the CNN model. Then we went ahead and created several models and made modifications by adding batch normalization and removing dropout layers, the results of which we discussed in this report earlier. Subsequently, we made changes in the optimizer parameter and used optimizers like SGD, Adagrad and Adamax and saw the results using each of them. 

We also experimented further by adding an extra convolutional layer and then also tinkered with the batch size and the number of epochs which gave positive results. There is of course, always a trade-off between several aspects like accuracy and the time that a model takes to run. Converting images to grayscale led to information loss but it also led to models running significantly faster. All in all, it is a matter of finding the right combination of parameters and the right architecture to obtain the maximum accuracy from a model.

## REFERECNES:
[1] liujeff. (2022, August 8). Hand Sign Classification MNIST. Kaggle: Your Machine Learning and Data Science Community. https://www.kaggle.com/code/liujeff/hand-sign-classification-mnist

[2] lordkun. (2022, April 3). Go Deeper with ResNet | Sign language (.99 acc). Kaggle: Your Machine Learning and Data Science Community. https://www.kaggle.com/code/lordkun/go-deeper-with-resnet-sign-language-99-acc/notebook

[3] Gavrilova, Y. (2021, August 3). What Are Convolutional Neural Networks? Serokell Software Development Company. https://serokell.io/blog/introduction-to-convolutional-neural-networks

[4] Gad, A. F. (2020, October 12). Accuracy, Precision, and Recall in Deep Learning | Paperspace Blog. Paperspace Blog. https://blog.paperspace.com/deep-learning-metrics-precision-recall-accuracy/

[5] Batch Normalization in Convolutional Neural Networks | Baeldung on Computer Science. (n.d.). Baeldung on Computer Science. https://www.baeldung.com/cs/batch-normalization-cnn#:~:text=Batch%20Norm%20is%20a%20normalization,learning%20rates,%20making%20learning%20easier.

[6] How to improve the performance of CNN Model for a specific Dataset? Getting Low Accuracy on both training and Testing Dataset. (n.d.). Stack Overflow. https://stackoverflow.com/questions/70554413/how-to-improve-the-performance-of-cnn-model-for-a-specific-dataset-getting-low

[7] Ali, A.-R. (2018, January 30). Histogram Equalization in Python. Code Envato Tuts+. https://code.tutsplus.com/tutorials/histogram-equalization-in-python--cms-30202

[8] How to Visualize a Deep Learning Neural Network Model in Keras - MachineLearningMastery.com. (n.d.). MachineLearningMastery.com. https://machinelearningmastery.com/visualize-deep-learning-neural-network-model-keras/

[9] exter minar. (2021, April 22). 35 Gesture Recognition Using Sign Language MNIST [Video]. YouTube. https://www.youtube.com/watch?v=6Bn0PY_ouBY

[10] Building a Convolutional Neural Network | Build CNN using Keras. (n.d.). Analytics Vidhya. https://www.analyticsvidhya.com/blog/2021/06/building-a-convolutional-neural-network-using-tensorflow-keras/

[11] Unable to import SGD and Adam from 'keras.optimizers'. (n.d.). Stack Overflow. https://stackoverflow.com/questions/67604780/unable-to-import-sgd-and-adam-from-keras-optimizers

****************************************************************************************************************

## Steps to run on local machine/jupyter notebook:
To run this assignment on your local machine, you need the following software installed.

*********************************
Anaconda Environment

Python
*********************************

To install Python, download it from the following link and install it on your local machine:

https://www.python.org/downloads/

To install Anaconda, download it from the following link and install it on your local machine:

https://www.anaconda.com/products/distribution

After installing the required software, clone the repo, and run the following command for opening the code in jupyter notebook.

jupyter notebook Sign_language_MNIST_detection.ipynb

This command will open up a browser window with jupyter notebook loading the project code.

You need to upload the dataset from the git repo to your jupyter notebook environment.
