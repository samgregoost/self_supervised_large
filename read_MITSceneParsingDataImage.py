__author__ = 'charlie'
import numpy as np
import os
import random
from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile
import glob

import TensorflowUtils as utils

# DATA_URL = 'http://sceneparsing.csail.mit.edu/data/ADEChallengeData2016.zip'
DATA_URL = 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'

def recursive_glob(rootdir='.', suffix=''):
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if filename.endswith(suffix)]

def read_dataset(data_dir):
    pickle_filename = "MITSceneParsing.pickle"
    pickle_filepath = os.path.join('/scratch1/ram095/nips20/logs_colorimglarge', pickle_filename)
    if not os.path.exists(pickle_filepath):
       # utils.maybe_download_and_extract(data_dir, DATA_URL, is_zipfile=True)
       # SceneParsing_folder = os.path.splitext(DATA_URL.split("/")[-1])[0]
        result = create_image_lists(data_dir)
        print ("Pickling ...")
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    else:
        print ("Found pickle file!")

    with open(pickle_filepath, 'rb') as f:
        result = pickle.load(f)
        training_records = result['training']
        validation_records = result['validation']
        del result

    return training_records, validation_records


def create_image_lists(image_dir):
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    directories = ['train', 'val']
    image_list = {}

    for directory_ in directories:
        if directory_ == "train":
            directory = "training"
        else:
            directory = "validation" 
        file_list = []
        image_list[directory] = []
      #  print(os.path.join(image_dir, "images", directory, '*.' + 'jpg'))
      #  file_glob_jpg = os.path.join(image_dir, "images", directory, '*.' + 'jpg')
       # file_glob_JPG = os.path.join(image_dir, "images", directory, '*.' + 'JPG')
      #  file_glob_png = os.path.join(image_dir, "images", directory, '*.' + 'png')
     #   file_glob_JPEG = os.path.join(image_dir, "images", directory, '*.' + 'JPEG')
       # file_glob_png = os.path.join(image_dir, "images", directory, '*.' + 'png')
        file_list.extend(recursive_glob(os.path.join(image_dir,directory_),".jpg" ))
        file_list.extend(recursive_glob(os.path.join(image_dir,directory_),".JPG" ))
        file_list.extend(recursive_glob(os.path.join(image_dir,directory_),".png" ))
        file_list.extend(recursive_glob(os.path.join(image_dir,directory_),".JPEG" ))
        if not file_list:
            print('No files found')
        else:
            for f in file_list:
                filename = os.path.splitext(f.split("/")[-1])[0]
              #  annotation_file_JPG = os.path.join(image_dir, "images", directory, filename + '.JPG')
               # annotation_file_png = os.path.join(image_dir, "images", directory, filename + '.png')
              #  annotation_file_jpg = os.path.join(image_dir, "images", directory, filename + '.jpg')
             #   annotation_file_JPEG = os.path.join(image_dir, "images", directory, filename + '.JPEG')
               # if os.path.exists(annotation_file_jpg):
                record = {'image': f, 'annotation': f, 'filename': filename}
                image_list[directory].append(record)
                #elif  os.path.exists(annotation_file_png):
                #    record = {'image': f, 'annotation': annotation_file_png, 'filename': filename}
                #    image_list[directory].append(record)
                #elif  os.path.exists(annotation_file_JPG):
                #    record = {'image': f, 'annotation': annotation_file_JPG, 'filename': filename}
                #    image_list[directory].append(record)
                #elif  os.path.exists(annotation_file_JPEG):
                #    record = {'image': f, 'annotation': annotation_file_JPEG, 'filename': filename}
                #    image_list[directory].append(record)
                #else:
                 #   print("Annotation file not found for %s - Skipping" % filename)

        random.shuffle(image_list[directory])
        no_of_images = len(image_list[directory])
        print ('No. of %s files: %d' % (directory, no_of_images))

    
    return image_list
