#!/usr/bin/env python
## Author: Rohit
## Date: July, 25, 2017
# Purpose: Ros node to detect objects using tensorflow

import os
import sys
import cv2
import tarfile
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
from std_msgs.msg import String , Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

# Object detection module imports
import object_detection
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import urllib2

DEFAULT_MODEL = os.path.join(os.path.expanduser("~"), "data", "models", "ssd_mobilenet_v1_coco_2018_01_28", "frozen_inference_graph.pb")
DEFAULT_LABELS = os.path.join(os.path.dirname(sys.path[0]), "data", "labels", "mscoco_label_map.pbtxt")

def download_model(url, location):

    file_name = url.split('/')[-1]
    u = urllib2.urlopen(url)
    if not os.path.exists(location):
        os.makedirs(location)

    file_path = os.path.join(location, os.path.basename(url))
    f = open(file_path, 'wb')
    meta = u.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    print "Downloading: %s Bytes: %s" % (file_name, file_size)

    file_size_dl = 0
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
        status = status + chr(8)*(len(status)+1)
        print status,

    f.close()

    return file_path

def download_data():
    ## Downloading COCO Trained Model
    if not os.path.exists(DEFAULT_MODEL):
        model_path = os.path.join(os.path.expanduser("~"), "data", "models")
        final_path = download_model("http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz", model_path)
        print final_path
        tar = tarfile.open(final_path)
        tar.extractall(path=os.path.dirname(final_path))
        tar.close()

def get_model_params():
    model_path = rospy.get_param("~model_path")
    labels_path = rospy.get_param("~labels_path")


    if (not model_path and not labels_path):
        rospy.logwarn("No params passed, using default model")
        download_data()
        return (DEFAULT_MODEL, DEFAULT_LABELS)

    elif (os.path.exists(os.path.join(model_path, "frozen_inference_graph.pb")) and os.path.exists(labels_path)):
        rospy.loginfo("Using Passed parameters")
        return (os.path.join(model_path, "frozen_inference_graph.pb"), labels_path)

    else:
        raise Exception("Either Incomplete arguments were passed or the paths do not exist. To use the default model do not pass any parameters. NOTE: Please use absolute paths in params")

# Detection

class Detector:

    def __init__(self):

        ######### Set model here ############
        path_to_ckpt, path_to_labels = get_model_params()

        num_classes = rospy.get_param("~num_classes")

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
        label_map = label_map_util.load_labelmap(path_to_labels)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

        # Setting the GPU options to use fraction of gpu that has been set
        # config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = GPU_FRACTION
        ops = detection_graph.get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        self.tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                self.tensor_dict[key] = detection_graph.get_tensor_by_name(
                    tensor_name)

        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        self.sess = tf.Session(graph=detection_graph)

        rospy.loginfo("Initializing")

        dummy_tensor = np.zeros((1,1,1,3), dtype=np.int32)
        self.sess.run(self.tensor_dict,
                               feed_dict={self.image_tensor: dummy_tensor})

        self.image_pub = rospy.Publisher("debug_image",Image, queue_size=1)
        self.object_pub = rospy.Publisher("objects", Detection2DArray, queue_size=1)
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

        output_dict = self.sess.run(self.tensor_dict,
                                    feed_dict={self.image_tensor: image_np_expanded})

        objects=vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(output_dict["detection_boxes"]),
            np.squeeze(output_dict["detection_classes"]).astype(np.int32),
            np.squeeze(output_dict["detection_scores"]),
            self.category_index,
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
        obj.bbox.center.x = int((dimensions[1] + dimensions [3])*image_width/2)
        obj.bbox.center.y = int((dimensions[0] + dimensions[2])*image_height/2)

        return obj

def main(args):
    rospy.init_node('detector_node')
    obj=Detector()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("ShutDown")
    cv2.destroyAllWindows()

if __name__=='__main__':
    main(sys.argv)
