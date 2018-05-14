#!/usr/bin/env python
## Author: Rohit
## Date: July, 25, 2017
# Purpose: Ros node to detect objects using tensorflow

# Modified by Thanuja Ambegoda, Enway GmbH - May 13th, 2018
# Added config.yaml to paraeterize model name, model path, label file path...

import os
import sys
import cv2
import numpy as np
try:
    import tensorflow as tf
except ImportError:
    print("unable to import TensorFlow. Is it installed?")
    print("  sudo apt install python-pip")
    print("  sudo pip install tensorflow")
    sys.exit(1)

# ROS related imports
import rospy
import rospkg
from std_msgs.msg import String , Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

# Object detection module imports
import object_detection
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

rospy.init_node('tensorflow_object_detector_node')
debug_image_topic = rospy.get_param('~debug_image_pub_name')
object_pub_topic = rospy.get_param('~object_pub_name')
model_name = rospy.get_param('~model_name')
gpu_fraction = rospy.get_param('~gpu_fraction')
rospack = rospkg.RosPack()
pkg_path = rospack.get_path('tensorflow_object_detector');
model_root = rospy.get_param('~model_root')
model_path = os.path.join(model_root, 'models', model_name)
frozen_graph_file_name = rospy.get_param('~frozen_graph_file_name')
path_to_ckpt = os.path.join(model_path, frozen_graph_file_name)
label_map_file = rospy.get_param('~label_map_file_name')
path_to_label_maps = os.path.join(model_root, 'labels', label_map_file)
num_classes = rospy.get_param('~num_classes')

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`,
# we know that this corresponds to `airplane`.  Here we use internal utility functions,
# but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(path_to_label_maps)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Setting the GPU options to use fraction of gpu that has been set
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction

# Detection
with detection_graph.as_default():
  with tf.Session(graph=detection_graph,config=config) as sess:
    class detector:

      def __init__(self):
        self.image_pub = rospy.Publisher(debug_image_topic, Image, queue_size=1)
        self.object_pub = rospy.Publisher(object_pub_topic, Detection2DArray, queue_size=1)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("image", Image, self.image_cb, queue_size=1, buff_size=2**24)

      def image_cb(self, data):
        objArray = Detection2DArray()
        try:
          cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
          print(e)
        image=cv2.cvtColor(cv_image,cv2.COLOR_BGR2RGB)

        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = np.asarray(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        objects=vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=2)

        objArray.detections =[]
        objArray.header=data.header
        object_count=1

        for i in range(len(objects)):
          object_count+=1
          objArray.detections.append(self.object_predict(objects[i],data.header,image_np,cv_image))

        self.object_pub.publish(objArray)

        img=cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_out = Image()
        try:
          image_out = self.bridge.cv2_to_imgmsg(img,"bgr8")
        except CvBridgeError as e:
          print(e)
        image_out.header = data.header
        self.image_pub.publish(image_out)

      def object_predict(self,object_data, header, image_np,image):
        image_height,image_width,channels = image.shape
        obj=Detection2D()
        obj_hypothesis= ObjectHypothesisWithPose()

        object_id=object_data[0]
        object_score=object_data[1]
        dimensions=object_data[2]

        obj.header=header
        obj_hypothesis.id = object_id
        obj_hypothesis.score = object_score
        obj.results.append(obj_hypothesis)
        obj.bbox.size_y = int((dimensions[2]-dimensions[0])*image_height)
        obj.bbox.size_x = int((dimensions[3]-dimensions[1] )*image_width)
        obj.bbox.center.x = int((dimensions[1] + dimensions [3])*image_height/2)
        obj.bbox.center.y = int((dimensions[0] + dimensions[2])*image_width/2)

        return obj

def main(args):
  obj=detector()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("ShutDown")
  cv2.destroyAllWindows()

if __name__=='__main__':
  main(sys.argv)
