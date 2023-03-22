from sklearn.svm import LinearSVC
from skimage import feature
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import os
import cv2
import numpy as np

def prepare_data(data_folder_path):
    images = []
    labels = []

    # get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)

    for dir_name in dirs:        
        # ignore folders that don't start with 's'
        if not dir_name.startswith("s"):
            continue;
            
        # extract the label from each folder
        label = int(dir_name.replace("s", ""))
        
        # get full path
        subject_dir_path = data_folder_path + "/" + dir_name
        
        # get all image names inside complete path
        subject_images_names = os.listdir(subject_dir_path)
        
        # for each image, detect face and add face to list of faces
        for image_name in subject_images_names:
            
            # ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;
            
            # build image path
            image_path = subject_dir_path + "/" + image_name

            #read image
            image = cv2.imread(image_path)
            image = cv2.resize(image, (128, 256))        
            images.append(image)
            labels.append(label)
            
    return images, labels

def prepare_train(data_folder_path):
    hog_descriptions = []
    hog_images = []
    labels = []

    # get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)
        
    #let's go through each directory and read images within it
    for dir_name in dirs:
        if not dir_name.startswith("s"):
            continue;
        
        label = int(dir_name.replace("s", ""))
        
        # get full path
        subject_dir_path = data_folder_path + "/" + dir_name
        
        # get all image names inside complete path
        subject_images_names = os.listdir(subject_dir_path)
        
        # for each image, detect face and add face to list of faces
        for image_name in subject_images_names:
            
            #ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;
            
            # build image path
            image_path = subject_dir_path + "/" + image_name

            # read image
            image = cv2.imread(image_path)
            image = cv2.resize(image, (128, 256))        
            hog_desc, hog_image = feature.hog(image, orientations=9, pixels_per_cell=(8, 8),
                                        cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys', visualize=True, channel_axis=-1)
            hog_descriptions.append(hog_desc)
            hog_images.append(hog_image)
            labels.append(label)
                        
    return hog_descriptions, hog_images, labels

def apply_hog_on_images(images):
    hog_images = []
    hog_descriptions = []
    for image in images:
        hog_desc, hog_img = feature.hog(image, orientations=9, pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys', channel_axis=-1, visualize=True)
        hog_descriptions.append(hog_desc)
        hog_images.append(hog_img)
    return hog_descriptions 

labels_name = ["", "Robert Downey Jr.", "Elvis Presley"]
train_images, train_labels = prepare_data("training-data")
hog_descriptions = apply_hog_on_images(train_images)

# train Linear SVC 
print('Training on train images...')
svm_model = LinearSVC(random_state=42, tol=1e-5)

# check the shape of your label array
svm_model.fit(hog_descriptions, train_labels)

# test images
test_images, test_labels = prepare_data("test-data")
pred_labels = []

test_hog_desc, test_hog_img, true_labels = prepare_train("training-data")

# make the predictions
for hog in test_hog_desc:
    pred = svm_model.predict(hog.reshape(1, -1))[0]
    pred_labels.append(pred)

# calculate accuracy score
accuracy = accuracy_score(true_labels, pred_labels)
print("Accuracy:", accuracy)

# calculate confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
print("Confusion Matrix:\n", cm)

# calculate f1 score
f1 = f1_score(true_labels, pred_labels, average='micro')
print("F1 score:", f1)

""" 
hog_image = test_hog_img.astype('float64')
# show the HOG image
cv2.imshow('HOG Image', hog_image)

# put the predicted text on the test image
cv2.putText(test_hog_img, prediction_labels[pred], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
cv2.imwrite(f"outputs/hog_2.jpg", hog_image*255.) # multiply by 255. to bring to OpenCV pixel range
cv2.imwrite(f"outputs/pred_2.jpg", test_hog_img) 
"""
cv2.waitKey(0)
cv2.destroyAllWindows()
