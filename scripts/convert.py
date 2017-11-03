
import os
import sys
import tensorflow as tf
import pandas as pd
from PIL import Image

from object_detection.utils import dataset_util

csv_path = os.path.join(os.path.dirname(sys.path[0]),"pedestrian_data", "data.csv")
img_path = os.path.join(os.path.dirname(sys.path[0]),"pedestrian_data", "images")

def get_csv_data(file):
    data = pd.read_csv(file)
    image_names, center_x, center_y, size_x, size_y = [], [], [], [], []
    image_names = list(data['images'])
    center_x = list(data['center_x'])
    center_y = list(data['center_y'])
    size_x = list(data['size_x'])
    size_y = list(data['size_y'])
    return image_names, center_x, center_y, size_x, size_y


def create_tf_example(encoded_image_data, filename, height, width, xmins, xmaxs, ymins, ymaxs ):
    image_format = b'jpg'
    tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

if __name__ == "__main__":
    class_name  = "pedestrian"
    image_names, center_x, center_y, size_x, size_y = get_csv_data(csv_path)
    for index in range(len(image_names)):
        filename = image_names[index]
        im=Image.open(os.path.join(img_path, image_names[index]))
        width = int(im.size[0])
        height = int(im.size[1])
        dw = 1./width
        dh = 1./height
        # Normalizing the coordinates
        x_mins = (center_x[index]-size_x[index]/2)*dw
        x_maxs = (center_x[index]+size_x[index]/2)*dw
        y_mins = (center_y[index]-size_y[index]/2)*dh
        y_maxs = (center_y[index]+size_y[index]/2)*dh
        classes_text = [class_name]
        classes = [1]
        tf_example = create_tf_example(im, filename, height, width, x_mins, x_maxs, y_mins, y_maxs)
        writer.write(tf_example.SerializeToString())
    writer.close()
