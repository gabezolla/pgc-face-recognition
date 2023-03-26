import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import statistics
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

#function to detect face using OpenCV
def detect_face(img):
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #load OpenCV face detector, I am using LBP which is fast
    #there is also a more accurate but slow Haar classifier
    face_cascade = cv2.CascadeClassifier('open-cv-face-recognizer/haarcascade_frontalface_default.xml')

    #let's detect multiscale (some images may be closer to camera than others) images
    #result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    
    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None

    #under the assumption that there will be only one face,
    #extract the face area
    (x, y, w, h) = faces[0]
    
    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]

#this function will read all persons' training images, detect face from each image
#and will return two lists of exactly same size, one list 
# of faces and another list of labels for each face
def prepare_training_data(data_folder_path):
    #------STEP-1--------
    #get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)
    totalFaces = 0
    undetectedFaces = 0
    
    #list to hold all subject faces
    faces = []
    #list to hold labels for all subjects
    labels = []
    
    #let's go through each directory and read images within it
    for dir_name in dirs:
        
        #our subject directories start with letter 's' so
        #ignore any non-relevant directories if any
        if not dir_name.startswith("s"):
            continue;
            
        #------STEP-2--------
        #extract label number of subject from dir_name
        #format of dir name = slabel
        #, so removing letter 's' from dir_name will give us label
        label = int(dir_name.replace("s", ""))
        
        #build path of directory containin images for current subject subject
        #sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name
        
        #get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)
        
        #------STEP-3--------
        #go through each image name, read image, 
        #detect face and add face to list of faces
        for image_name in subject_images_names:
                        
            #ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;
            
            #build image path
            #sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name

            #read image
            image = cv2.imread(image_path)
            image = cv2.resize(image, (416,416))

            #display an image window to show the image 
            cv2.imshow("Training on image...", image)
            cv2.waitKey(100)
                        
            #detect face
            face, rect = detect_face(image)
                        
            #------STEP-4--------
            #for the purpose of this tutorial
            #we will ignore faces that are not detected
            if face is not None:
                totalFaces += 1
                #add face to list of faces
                faces.append(face)
                #add label for this face
                labels.append(label)
            else:
                undetectedFaces += 1
            
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    return faces, labels, totalFaces, undetectedFaces

def prepare_test(data_folder_path):
    true_labels = []
    pred_labels = []
    pred_distance = []
    totalFaces = 0
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
            test_img1 = cv2.imread(image_path)
            pred_img, pred_label = predict(test_img1)

            if pred_label is not None:
                pred_labels.append(pred_label[0])
                pred_distance.append(pred_label[1])
                true_labels.append(label)
            else:
                continue;

    return pred_labels, true_labels, totalFaces, undetectedFaces, pred_distance

#function to draw rectangle on image 
#according to given (x, y) coordinates and 
#given width and heigh
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
#function to draw text on give image starting from
#passed (x, y) coordinates. 
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

#this function recognizes the person in image passed
#and draws a rectangle around detected face with name of the 
#subject
def predict(test_img):
    #make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    #detect face from the image
    face, rect = detect_face(img)

    if face is None:
        return None, None

    #predict the image using our face recognizer 
    test_label = face_recognizer.predict(face)
        
    return img, test_label

faces, labels, train_total, train_undetected = prepare_training_data("training-data")
image_size = (416,416)
for face in faces: 
    face = cv2.resize(face, image_size)

#train LBPHFaceRecognizer 
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))

pred_labels, true_labels, test_total, test_undetected, pred_distance = prepare_test("test-data")

# detection percentage
detectionPercentage = (train_total + test_total - train_undetected - test_undetected)*100/(train_total + test_total)
print("Detection percentage:", detectionPercentage)

# distance average
distanceAverage = statistics.mean(pred_distance)
print("Distance average:", distanceAverage)

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
#create a figure of 2 plots (one for each test image)
#f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

#display test image1 result
ax1.imshow(cv2.cvtColor(predicted_img1, cv2.COLOR_BGR2RGB))

#display test image2 result
ax2.imshow(cv2.cvtColor(predicted_img2, cv2.COLOR_BGR2RGB))

#display both images
cv2.imshow("Test one", predicted_img1)
cv2.imshow("Test two", predicted_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()
"""