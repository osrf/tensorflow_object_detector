# tf_object_detect

Tensorflow Object Detector with ROS

Requirements:
-Tensorflow
-ROS

Steps:

To run Default SSD (Single Shot Detection) algorithm:
1) In your catkin workspace, RUN `catkin_make install`
2) Source the tensorflow environment
3) Run `roslaunch tf_object_detect object_detect.launch`

If you want to try any other model:
1) Download any Object Detection Models from the Tensorflow Object detection API and place it in `data/models/`. 
You can find the models in tensorflow Object Detection Model Zoo : https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md. Extract the `tar.gz` file.  

2) Edit the MODEL_NAME and LABEL_NAME in detect_ros.py, By default it is `ssd_mobilenet_v1_coco_11_06_2017` with `mscoco_label_map.pbtxt` respectively. 

