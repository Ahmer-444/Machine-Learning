import pickle
import cv2

def read_dataset(pkl_dataset_path):
    file = open(pkl_dataset_path, 'rb')
    dataset = pickle.load(file)
    file.close()

    return dataset

if __name__ == "__main__":
    pkl_dataset_path = 'data.pkl'
    dataset = read_dataset(pkl_dataset_path)

    '''
    B = dataset[770][:][:][:] * 255
    B = B.astype(int)
    cv2.imwrite('numpy_img.jpg',B)
    '''
