import numpy as np
import pickle
import glob
import cv2

import os

dictt = {'a':'11','b':'12','c':'13','d':'14','e':'15','f':'16','g':'17','h':'18',
'i':'19','j':'20','k':'21','l':'22','m':'23','n':'24','o':'25','p':'26','q':'27',
'r':'28','s':'29','t':'30','u':'31','v':'32','w':'33','x':'34','y':'35','z':'36'
}
def create_dataset(images_dir,size):
    count = 0
    
    cpt = sum([len(files) for r, d, files in os.walk(images_dir)])
    size1 = (cpt,) + size
    size2 = (cpt,1)

    dataset_array = np.zeros(size1,dtype='float64')
    labels_array = np.zeros(size2,dtype='float64')



    subdirs = [x[0] for x in os.walk(images_dir)]
    del subdirs[0]

    for subdir in subdirs:
        label = subdir.split('/')[-1:][0]
        if label.isdigit():
            label = int(label)
        else:
            label = dictt[label.lower()]
            label = int(label)
        
        images_list = glob.glob1(subdir + "/","*.png")

        # append 4th dimesion == Total num of Images --> shape = [NUM_IMAGES][W][H][D]

        # create a numpy array (float64) of zeros of size evaluated.
        # Float becasue of normlization of images
        for im in images_list:
            image = cv2.imread(subdir + '/'+ im)
            resized_image = cv2.resize(image,(size1[1],size1[2]))
            normalize_image = np.zeros(resized_image.shape,dtype='float64')
            cv2.normalize(resized_image, normalize_image, alpha=0, beta=1,\
                                            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)

            dataset_array[count][:][:][:] = normalize_image
            labels_array[count][0] = label
            count = count + 1

    return dataset_array,labels_array
    

if __name__ == "__main__":
    images_dir = 'chars-dataset/'
    size = (8,8,3)
    dataset_array,labels_array = create_dataset(images_dir,size)
    #print dataset_array

    file = open('data-chars.pkl', 'wb')
    # Pickle dictionary using protocol 0.
    pickle.dump([dataset_array, labels_array], file)
    file.close()

    with open('data-chars.pkl') as f:
        obj0, obj1 = pickle.load(f)
        print obj0
