# FaceRecognition-System
# Library:
1-> cmake
2-> DLib
3->facereognation 
4-> numpy
5-> opencv

# Steps for facial recognition:
step 1: finding the faces and we are using the hog method in the backhand which is 
Hogs Method:
The histogram of oriented gradients (HOG) is a feature descriptor used in computer vision and 
image processing for the purpose of object detection. The technique counts occurrences of 
gradient orientation in localized portions of an image.

Step 2: posing and projecting Faces it uses Dlib library
Dlib: This library is use for implementing a variety of machine learning algorithms, including
classification, regression,clustering, data transformation, and structured prediction.
Dlib make facial landmarks and using it make it centrilized the face then after centralizing 
then it send the face to neural network which is already trained and it gives encoded feature

step 3: Encodeing faces
its a measurements of distance between eyes nose and etc and it generate about 128 measurements
by using this measurements we can differentiat between different people


step Finding the person's name from teh encoding to do that we can use any machine learning meathods
to classifing here we use SVM classifier it can take the measurements and classify that this measurement 
is for the person that store in our data base or note..
