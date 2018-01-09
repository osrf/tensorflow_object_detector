# Tensorflow Object Detector with ROS

Requirements:
Tensorflow and ROS

Steps:

To run Default SSD (Single Shot Detection) algorithm:
1) This package uses the proposed standard Vision messages, so clone/download the vision messages package and place it in the catkin workspace. Vision messages package: https://github.com/Kukanani/vision_msgs.git
2) Clone/download this repository and place it your catkin workspace, RUN `catkin_make`
3) Source the tensorflow environment
4) Run `roslaunch tf_object_detect object_detector.launch`. (NOTE: This launch file also launches the openni2.launch file for the camera. If you are using any other camera, please change the camera topic in the launch file before launching the file)

If you want to try any other model:
1) Download any Object Detection Models from the Tensorflow Object detection API and place it in `data/models/`. 
You can find the models in tensorflow Object Detection Model Zoo : https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md. Extract the `tar.gz` file.  

2) Edit the MODEL_NAME and LABEL_NAME in detect_ros.py, By default it is `ssd_mobilenet_v1_coco_11_06_2017` with `mscoco_label_map.pbtxt` respectively. 

