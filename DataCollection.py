# Muh-dikhai ki rasam
import cv2
import os
import pickle
import numpy as np


def datacollect(data_directory):
    cap = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    count = 0
    name = input("Enter name - ")

    while True:
        _, img = cap.read()
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(grey, 1.1, 6)  # Adjust scaleFactor and minNeighbors for better results

        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cropped = grey[y:y + h, x:x + w]
            cv2.putText(img, "S : Save", (20, 400), 0, 1.5, (255, 0, 0), 2)
            cv2.putText(img, "Q : Quit", (20, 450), 0, 1.5, (255, 0, 0), 2)
            cv2.imshow("Capture Data", img)

            if cv2.waitKey(1) == ord('s'):
                if not os.path.exists(data_directory):
                    os.makedirs(data_directory)
                image = cv2.resize(cropped, (100, 100))
                file_path = os.path.join(data_directory, f"{name}_{count}.jpg")
                cv2.imwrite(file_path, image)
                print(f"Saved: {file_path}")
                count += 1

            break  # Ensure only one detection at a time

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def get_images_and_labels(data_directory):
    image_paths = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith('.jpg')]
    images = []
    labels = []

    for image_path in image_paths:
        # Read the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Extract the label from the image filename
        label = os.path.split(image_path)[-1].split('_')[0]
        # Append the image and label to lists
        images.append(image)
        labels.append(label)

    return images, labels


def prepare_data(images, labels):
    label_dict = {label: idx for idx, label in enumerate(set(labels))}
    labeled_images = []
    labeled_labels = []

    for image, label in zip(images, labels):
        labeled_images.append(image)
        labeled_labels.append(label_dict[label])

    return labeled_images, labeled_labels, label_dict


def train_face_recognizer(data_directory, model_path):
    images, labels = get_images_and_labels(data_directory)
    images, labels, label_dict = prepare_data(images, labels)

    # Convert lists to numpy arrays
    images_np = np.array([np.array(image, 'uint8') for image in images])
    labels_np = np.array(labels)

    # Initialize the face recognizer
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Train the recognizer
    face_recognizer.train(images_np, labels_np)

    # Save the trained model
    face_recognizer.save(model_path)

    return label_dict


# Run this function
def collect_and_train(data_directory, model_path):
    datacollect(data_directory)
    label_dict = train_face_recognizer(data_directory, model_path)

    with open('label_dict.pkl', 'wb') as f:
        pickle.dump(label_dict, f)
    print("Model Trained")
