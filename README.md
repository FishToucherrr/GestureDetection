# Gesture Detection

Shilong Dong 19307130264,  Xushu Dai 19307130122

Project Repo: https://github.com/FishToucherrr/GestureDetection

accompanies the article [Motion gesture detection using Tensorflow on Android](http://blog.lemberg.co.uk/motion-gesture-detection-using-tensorflow-android)

forked from [Fudan-HCI/motion_gestures_detection](https://github.com/Fudan-HCI/motion_gestures_detection)


## Supported Gestures

> Five gestures in total

1. MoveLeft -> AliPay
2. MoveRight -> AliPay
3. UpDown -> Wechat
4. ForwardBack -> addressBook
5. Circle -> Camera

## 1. Project Introduction

​		This project aims to exploit multiple human gestures to supply self-preferenced interaction methods when using mobile phones. The whole project is built and supported by the Android Studio platform, and the gesture detection is based on machine learning methods which can be embedded into the Android application statically.

## 2. Neural Network Design

​		The neural network used for gesture motion detection inherits from the referenced project, which consists of two convolution layers and an intermedium max pooling layers to extract the feature of the input signals. After being flattened, the output of the convolution layer is fed into an activation layer with tanh serving as activate function, and then passes through a fully connect layer with Softmax to get the prediction of the gesture type.

​		Unlike the referenced project, we fully exploit the accelerator via recording and using its three axes data. The adequate physical data enables us to design and distinguish three more novel gestures (besides Moveleft and Moveright), which are Updown, Forwardback and Circle. Updown requires the users to shake their phones vertically, Forwardback requires the users to move their phone from a far positon to a closer one to their body, and Circle requires them to draw a circle in either clockwise or anticlockwise direction. 

​		The neural network has been trained three epoches with an accuracy over 99.9%.

## 3. Android Application Modification

​		We modify the detection and classification module of the origin project. Firstly we increase the dimension of recording data to three, and add our newly-desinged gesture labels. Secondly we replace the original frozen tensorflow model by our 5-category classification one.

​		After several trials we found that Simply modifying the MotionDetectionlibfile didn't make any impact on the exported APK file, for what the demo imports is the already packed jar file. We attempted to import source code of the lib project to build it instead of exporting a new jar package to cooperate with the demo.

​		We further exploit the gesture recognization application by enabling different gestures to wake  up different user-defined applications on the mobile phone. For example, when the gesture Updown is recognized, the Wechat will be waken up in a flash. Other gestures stand for various shorcuts: Forwardback stands for dialing, Circle represents opening the camera, while Moveleft and Moveright both wake up the Alipay based on our design.

## 4. Group Members' Contribution

​		This is a teamwork project, so we have a reasonable division and cooperation scheme. 

​		Dong designed the newly added gestures and was responsible for training the newly designed neural networks. He also contributed to one third of the training data for the three novel gestures and drafted the project report.

​		Dai modified the demo Android project to adjust to our new gesture recognization, and added the user defined shorcuts into the Motion Gesture Detect Application. He supplied the rest of the training data and implementd the project report.
