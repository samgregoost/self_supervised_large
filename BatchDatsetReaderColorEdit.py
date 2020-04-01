"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
import numpy as np
import scipy.misc as misc
from skimage import io, color
from sklearn.utils import shuffle

class BatchDatset:
    files = []
    images = []
    image_files = []
    annotation_files = []
    annotations = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, records_list, image_options={}):
        """
        Intialize a generic file reader with batching for list of files
        :param records_list: list of file records to read -
        sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}
        :param image_options: A dictionary of options for modifying the output image
        Available options:
        resize = True/ False
        resize_size = #size of output image - does bilinear resize
        color=True/False
        """
        print("Initializing Batch Dataset Reader...")
        print(image_options)
        self.files = records_list
        self.image_options = image_options
        self._read_images()

    def _read_images(self):
        self.image_files  = [filename['image'] for filename in self.files]
        self.annotation_files =  [filename['annotation'] for filename in self.files]
 

        
        print (len(self.image_files))
        print (len(self.annotation_files))

    def _transform(self, filename, mode):
       # image = io.imread(filename).astype(np.uint8)
        image = misc.imread(filename, mode = 'RGB').astype(np.uint8)
       # image = np.interp(image, (0, 255), (0, 0.1))
       # if self.__channels:  # make sure images are of shape(h,w,3)
#            print("SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS")
        #    image = np.array([image for i in range(3)])
        
      #      print(np.max(image))
       #     print(np.min(image))
        #    print(np.count_nonzero(image))
      

        if self.image_options.get("resize", False) and self.image_options["resize"]:
            resize_size = int(self.image_options["resize_size"])
            resize_image = misc.imresize(image,
                                         [resize_size, resize_size], interp='bicubic')
        else:
            resize_image = image
        
        if mode == 'images':  # make sure images are of shape(h,w,3)
#            print("SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS")
            resize_size = int(self.image_options["resize_size"])
            image_ = color.rgb2lab(resize_image)
            image = np.reshape(image_[:,:,0],(resize_size,resize_size,1))
            image = image/50.0 - 1
        #    print("$$$$$$$$$$$")
        #    print(np.max(image))
        #    print(np.min(image))
        else:
            image_ = color.rgb2lab(resize_image)
            image = image_[:,:,1:]
            image = (image + 128.0)/128.0 - 1

          
        
        #print("$$$$$$$$$$$")
        #print(np.max(image))
        #print(np.min(image))
       # image = image/127.5 - 1.0
        return np.array(image)

    def get_records(self):
        return self.images, self.annotations

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > len(self.image_files):
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            self.image_files, self.annotation_files = shuffle(self.image_files, self.annotation_files)
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
       
        current_image_batch = self.image_files[start:end]
        current_annotation_batch = self.annotation_files[start:end]
        
        list1 = []
        list2 = []
        try:
            for filename in current_image_batch:
                list1.append(self._transform(filename,'images'))
                list2.append(self._transform(filename,'annotations')) 
         #   image_batch = np.array([self._transform(filename,'images') for filename in current_image_batch])
          #  annotation_batch = np.array([self._transform(filename,'annotations') for filename in current_annotation_batch])        
        except:
            print(filename + "_error")
        
        image_batch = np.array(list1)
        annotation_batch = np.array(list2) 
      #  print([current_image_batch])
      #  print([filename for filename in current_annotation_batch])    
        return image_batch , annotation_batch

    def get_random_batch(self, batch_size):
        list1 = []
        list2 = []
        indexes = np.random.randint(0, len(self.image_files), size=[batch_size]).tolist()
        for i in indexes:
            filename = self.image_files[i]
            list1.append(self._transform(filename,'images'))
            list2.append(self._transform(filename,'annotations'))

        image_batch = np.array(list1)
        annotation_batch = np.array(list2)
        return image_batch, annotation_batch
