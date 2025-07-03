Emotion Recognition Using CNN

This project uses a Convolutional Neural Network (CNN) to classify emotions from facial images using the FER2013 dataset. The model classifies images into seven emotion categories: angry, disgust, fear, happy, neutral, sad, and surprise.

Installation

Clone the repository and install required libraries:

bash
Copy
git clone https://github.com/yourusername/emotion-recognition-cnn.git
cd emotion-recognition-cnn
!pip install kaggle tensorflow matplotlib seaborn

Dataset

The dataset can be downloaded from Kaggle FER2013 and contains:

Training: 22968 images

Validation: 5741 images

Test: 7178 images

Model Architecture

The CNN model consists of:

Conv2D and MaxPooling2D layers

BatchNormalization and Dropout layers for regularization

Dense layers for classification (7 classes)

Training

Train the model with:


Copy
history = model.fit(
    train_generator,
    validation_data=val_generator,
    callbacks=[checkpoint],
    epochs=100
)

Evaluation

Evaluate the model on the test data:


Copy
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy:.4f}")

Inference
To predict emotions from an image:


Copy
def predict_emotion(img_path):
    img = image.load_img(img_path, color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    predicted_label = class_labels[predicted_class]
    plt.imshow(img_array[0].squeeze(), cmap='gray')
    plt.title(f"Predicted: {predicted_label.capitalize()}")
    plt.axis('off')
    plt.show()
    
Save and Load Model

Save the trained model:


Copy
model.save('emotion_model.h5')
Load it later with:


Copy
from tensorflow.keras.models import load_model
model = load_model('emotion_model.h5')
