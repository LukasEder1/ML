import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import FeatureDetection as FD
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from scipy.spatial import distance

def kmeans(k, descriptor_list):
    """
    performs kmeans clustering
    used for bag of visual words
    
    Parameters
    ----------
    k : int
        number of clusters
    descriptor_list : list
        list of descriptors from sift/surf
    """
    kmeans = KMeans(n_clusters = k, n_init=10, verbose = 1)  
    kmeans.fit(descriptor_list)
    visual_words = kmeans.cluster_centers_ 
    return visual_words


# taken from:
# https://github.com/AybukeYALCINER/gabor_sift_bovw/blob/master/assignment1.py
def find_index(image, center):
    """
    find index of closest central point to each surf/sift descriptor
    """
    count = 0
    ind = 0
    for i in range(len(center)):
        if(i == 0):
            
            count = distance.euclidean(image, center[i]) 
           #count = L1_dist(image, center[i])
        else:
            dist = distance.euclidean(image, center[i]) 
            #dist = L1_dist(image, center[i])
            if(dist < count):
                ind = i
                count = dist
    return ind
    


def pick_categories(path, number_of_categories):
    """
    function used to determine, which categories will be used
    Also sets a fixed size of images to be used per category
    -> to ensure a fair distribution of images
    
    Parameters
    ----------
    path : String
        absolute path to the folder, where the different category
        folders are
    number_of_categories : int
        number of categories to be used
        
    Returns
    -------
    list
        the chosen categories
    int
        number of images per category
    """
    all_categories = os.listdir(path)
    # from all categories choose 'number_of_categories' many
    chosen_categories = random.sample(all_categories, number_of_categories)
    
    # output, which categories are being used
    print("Categories:")
    for i, category in enumerate(chosen_categories):
        category_name = category.split(".")
        print(f"category number {(i)} is {category_name[1]}")
    print("\n")
    
    # to have a balanced amount of images choose the category with the least amount of images as upper threshold
    number_of_images = min([len(os.listdir(os.path.join(path, category))) for category in chosen_categories])
    return chosen_categories, number_of_images
    


def image_class(bovw, centers):
    """
    Parameters
    ----------
    bovw : dict
        keys: categories, values: sift/surf descriptors
    centers : list
        visual words
        
    Returns
    -------
    dict
        keys:categories, values: histogram for each image

    """
    dict_feature = {}
    for key,value in bovw.items():
        category = []
        for img in value:
            histogram = np.zeros(len(centers))
            for each_feature in img:
                ind = find_index(each_feature, centers)
                histogram[ind] += 1
            category.append(histogram)
        dict_feature[key] = category
    return dict_feature



def load_and_format_data(directory, categories, detection_type, number_of_images, resize_size = (50,50), n_keypoints = 50, 
                         pixels_per_cell = (8,8), cells_per_block=(2,2)):
    """
    Parameters
    ----------
    directory : string
        absolute path to the folder
    categories : list
        categories to be used
    detection_type: string
        algorithm to be used either hog, sift, surf
    number_of_images: int
        maximum number of images per category
    resize_size: tuple
        new resolution of images
    n_keypoints: int
        number of keypoints for sift/surf
    pixels_per_cell: tuple
        pixels per cell for hog
     cells_per_block: tuple
        cells per block for hog   
    """
    
    # given to pickle
    data = []
    keypoint_vectors = {}
    descriptor_list = []
    for category in categories:
        # features for this very category
        features = []
        path = os.path.join(directory, category)
        label = categories.index(category)
        # loop over only the first "number_of_images" images
        for img in os.listdir(path)[:number_of_images]: 
            img_path = os.path.join(path, img)
            loaded_img = cv2.imread(img_path, 0)
            try:  # cv2.resize needs this try block
                loaded_img = cv2.resize(loaded_img, resize_size)
                if detection_type == "sift":
                    sift_kp, sift_des = FD.SIFT(loaded_img, n_keypoints)
                    descriptor_list.extend(sift_des)
                    features.append(sift_des)

                if detection_type == "surf":
                    surf_kp, surf_des = FD.SURF(loaded_img, n = 100, n_keypoints=n_keypoints)
                    descriptor_list.extend(surf_des)
                    features.append(surf_des)
                    
                if detection_type == "hog":
                    feature_matrix = FD.HOG(loaded_img, pixels_per_cell, cells_per_block)
                    data.append([scale_array(feature_matrix.flatten().astype(float), 0, 255), label])
        
        
        
            except Exception as e:
                print(e)
                
        if detection_type == "sift" or detection_type == "surf":
            keypoint_vectors[category] = features
    
    if detection_type == "sift" or detection_type == "surf":
        visual_words = kmeans(50, descriptor_list)
        data = image_class(keypoint_vectors, visual_words) 
                    
            
    pick_in = open('data_formatted.data', "wb")
    pickle.dump(data, pick_in)
    pick_in.close()
    
    
    # parameters to be used for different detection types (found using grid search)
    parameters = {'hog': {'C':0.001, 'kernel':'poly', 'gamma':'auto'},
                  'surf':{'C':10, 'kernel':'rbf', 'gamma':0.0001},
                  'sift':{'C':10, 'kernel':'rbf', 'gamma':1e-05}}
    
    pick_parameter = open('parameters.data', "wb") #save parameters
    pickle.dump(parameters[detection_type], pick_parameter)
    pick_parameter.close()
    

def gridsearch(data, parameters):
    """
    simple function to perform a gridsearch
    result: prints out best parameter combination
     
    Parameters
    ----------
    data : list
        test data
    parameters : dict
        parameters to be used in gridsearch
    """
    pick_in = open(data, "rb")  # 'data_test.data' , be careful not to leak training data here
    data = pickle.load(pick_in)
    pick_in.close()
    model = SVC()
    features = []
    labels = []
    
    for feature, label in data:
        features.append(feature)
        labels.append(label)
    
    xtest = features
    ytest = labels
    clf = GridSearchCV(model, parameters, scoring='accuracy', cv=5, n_jobs=-1, verbose=3)
    clf.fit(xtest, ytest)
    print(f"Best parameters: {clf.best_params_}")


def create_and_save_model(datapath, test_percentage = 0.2):
    """
    create and save svc model with prev. defined parameters
     
    Parameters
    ----------
    datapath : string
        used for pickle to load train/test data ('data_formatted.data')
    test_percentage : float
        percentage of data used for testing
    """
    
    pick_in = open(datapath, "rb")
    data = pickle.load(pick_in)
    pick_in.close()
    pick_parameter = open('parameters.data', "rb")
    parameters = pickle.load(pick_parameter)
    pick_parameter.close()
    #random.shuffle(keys)
    #shuffled_data = [(key, data[key]) for key in keys]
    
    features = []
    labels = []
    
    # sift/surf return dictonary, while hog returns list
    # convert both in same format
    if type(data) == dict:
        farray = []
        for label, label_features in data.items():
            for feature in label_features:
                farray.append([feature, label])
        data = farray
    
    random.shuffle(data)

    for feature, label in data:
        features.append(feature)
        labels.append(label)
        
    
        
    xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=test_percentage)
    
    # unpack parameters
    model = SVC(**parameters)
    model.fit(xtrain, ytrain)
    
    pick = open('model.data', "wb") #save model
    pickle.dump(model, pick)
    pick.close()

    test_data = list(zip(xtest,ytest))

    pick1 = open('data_test.data', "wb") #save test data, so that we don't mix up training and test data
    pickle.dump(test_data, pick1)
    pick1.close()

    print("n_test: ", len(xtest))
    print("n_train: ", len(xtrain))

def predict(model,test_data, show_confusion_matrix=False):
    """
    Make predictions on test_data using prev. model
     
    Parameters
    ----------
    model : string
        used for pickle to load model ('model_formatted.data')
    test_data : string
        used for pickle to load test data ('data_test.data')
    show_confusion_matrix: boolean
        flag used to decide whether or not a confusion matrix should be drawn
    """

    pick_model = open(model, "rb")  # 'model.data'
    model = pickle.load(pick_model)
    pick_model.close()

    pick_in = open(test_data, "rb")  # 'data_test.data' , be careful not to leak training data here
    data = pickle.load(pick_in)
    pick_in.close()

    features = []
    labels = []

    for feature, label in data:
        features.append(feature)
        labels.append(label)

    xtest = features
    ytest = labels
    
    accuracy = model.score(xtest, ytest)
    print("accuracy: ", accuracy)
      
    if show_confusion_matrix:
        prediction = model.predict(xtest)
        cm_array = confusion_matrix(ytest, prediction)
        sns.heatmap(cm_array, annot=True, cmap='Blues')
        plt.show()
    

def scale_array(arr, new_min, new_max):
    scale_arr = arr - np.min(arr) # offset to 0
    scale_arr = scale_arr - new_min  # offset to new_min
    scaled_arr = scale_arr * (new_max / (np.max(scale_arr) - np.min(scale_arr))) # stretch to new_max
    return scaled_arr


if __name__ == "__main__":
    dir = "path to directory of directories of images"
    
    categories, number_of_images = pick_categories(dir, 5)
    #load_and_format_data(dir, categories, "sift",number_of_images, resize_size=(500, 500), n_keypoints=500)  # use sift descritpors for model
   #load_and_format_data(dir, categories, "surf",number_of_images, resize_size=(500, 500), n_keypoints=500)  # use surf descritpors for model
    load_and_format_data(dir, categories, "hog",number_of_images, resize_size=(250, 500), pixels_per_cell = (8,8), cells_per_block=(2,2))  # use hog descritpors for model
    create_and_save_model('data_formatted.data', 0.2)
    
    # possible gridsearch usage -> prints out best parameters
    #parameters_hog = {'kernel': ['linear', 'poly'], 'C': [0.001, 0.1, 1, 10], 'degree': list(range(1,6))}
    #gridsearch('data_test.data', parameters_hog)
    
    # show_confusion_matrix to True
    predict('model.data', 'data_test.data', True)



