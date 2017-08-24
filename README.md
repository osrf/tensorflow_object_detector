# tf_object_detect
Tensorflow Object Detector

Steps:

1) Download the Object Detection Model from the Tensorflow Object detection API and place it in `data/models/`. 
You can find the models in tensorflow Object Detection Model Zoo : https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md. Extract the `tar.gz` file.  

2) Edit the MODEL_NAME and LABEL_NAME in detect_ros.py, By default it is `ssd_mobilenet_v1_coco_11_06_2017` with `mscoco_label_map.pbtxt` respectively. 

4) In your workspace Do, `catkin_make install`

3) Source the tensorflow environment

4) Run `roslaunch tf_object_detect object_detect.launch`

