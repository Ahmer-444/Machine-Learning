import numpy as np
import pickle
import glob
import cv2



def create_dataset(images_dir,size):
    count = 0
    images_list = glob.glob1(images_dir,"*.jpg")

    # append 4th dimesion == Total num of Images --> shape = [NUM_IMAGES][W][H][D]
    size = (len(images_list),) + size

    # create a numpy array (float64) of zeros of size evaluated.
    # Float becasue of normlization of images
    dataset_array = np.zeros(size,dtype='float64')

    for i in images_list:
        image = cv2.imread(images_dir+i)
        resized_image = cv2.resize(image,(size[1],size[2]))
        normalize_image = np.zeros(resized_image.shape,dtype='float64')
        cv2.normalize(resized_image, normalize_image, alpha=0, beta=1,\
                                        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)

        dataset_array[count][:][:][:] = normalize_image
        count = count + 1


    return dataset_array
    

if __name__ == "__main__":
    images_dir = 'sidefeet/'
    size = (64,64,3)
    dataset_array = create_dataset(images_dir,size)

    file = open('data.pkl', 'wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(dataset_array, file)
    file.close()

    

    '''
    B = dataset_array[50][:][:][:] * 255
    print B.shape
    B = B.astype(int)
    cv2.imwrite('numpy_img.jpg',B)
    '''
    
    
