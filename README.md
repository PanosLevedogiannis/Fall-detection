# Fall-detection
code for my project in cyberphysical


To run our code we have to install the following packages :
pip install opencv-python numpy tensorflow tensorflow-hub
Packages for train code:
 pip install scikit-learn
 pip install pandas


First in datacollectionwithvideo.py we run it and create a csv file named lstm_v1.csv
Then in train_model.py we train the model and create a .h5 file named lstm_v1.h5
To see the results we run videodetection.py and put a video for example the falls1 from falls video, if our model detect fall its working .
