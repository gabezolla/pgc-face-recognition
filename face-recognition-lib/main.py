import face_recognition
from sklearn import svm
import os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

def prepare_data(data_folder_path):
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
            face = face_recognition.load_image_file(image_path)
            face_bounding_boxes = face_recognition.face_locations(face)

            # If training image contains exactly one face
            if len(face_bounding_boxes) == 1:
                face_enc = face_recognition.face_encodings(face)[0]
                # Add face encoding for current image with corresponding label (name) to the training data
                encodings.append(face_enc)
                labels.append(label)          
            
    return encodings, labels

def prepare_test(data_folder_path):
    true_labels = []
    pred_labels = []
    undetectedFaces = 0
    
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
            face = face_recognition.load_image_file(image_path)
            
            # Find all the faces in the test image using the default HOG-based model
            if len(face_recognition.face_locations(face)) == 0:
                undetectedFaces += 1
                continue;
            
            test_image_enc = face_recognition.face_encodings(face)[0]
            pred_labels.append(clf.predict([test_image_enc]))
            true_labels.append(label)

    return pred_labels, true_labels, undetectedFaces

# The training data would be all the face encodings from all the known images and the labels are their names
encodings = []
labels = []


encodings, labels = prepare_data("training-data")

# Create and train the SVC classifier
clf = svm.SVC(gamma='scale')
clf.fit(encodings, labels)

pred_labels, true_labels, undetectedFaces = prepare_test("test-data")

# detection percentage
detectionPercentage = len(true_labels)*100/(len(true_labels) + undetectedFaces)
print("Detection percentage:", detectionPercentage)

# calculate accuracy score
accuracy = accuracy_score(true_labels, pred_labels)
print("Accuracy:", accuracy)

# calculate confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
print("Confusion Matrix:\n", cm)

# calculate f1 score
f1 = f1_score(true_labels, pred_labels, average='micro')
print("F1 score:", f1)
