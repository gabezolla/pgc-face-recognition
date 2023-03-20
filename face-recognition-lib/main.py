import face_recognition
import os

# Set the path to the directory containing the images
image_dir = "images"

# Set the training face
obama_image = face_recognition.load_image_file("obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
known_faces = [obama_face_encoding]

# Initialize an empty array to store the images
images = []
encodings = []
numberOfObamas = 0

# Loop over the files in the directory and load each image
for file in os.listdir(image_dir):
    if file.endswith(".jpg"):
        image_path = os.path.join(image_dir, file)
        image = face_recognition.load_image_file(image_path)
        images.append(image)

        # Compute the face encodings for the image
        encoding = face_recognition.face_encodings(image)[0]
        print(f"This image is {file}")
        print("Is this image Obama? {}".format(face_recognition.compare_faces(known_faces, encoding)))
        if(face_recognition.compare_faces(known_faces, encoding)[0] == True):
            numberOfObamas += 1 

print(f"Obamas counted: {numberOfObamas}")