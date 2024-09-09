import tensorflow as tf
import numpy as np

# Load the best saved model for inference
best_model = tf.keras.models.load_model('best_model.keras')

# Example inference on a single image
def predict_image(img_path, model):
    try:
        # Load and preprocess the image (ensure input size matches training)
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(128, 128))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize the image

        # Debugging: Print the preprocessed image array shape and pixel values
        print(f"Image shape: {img_array.shape}")
        print(f"First 5 pixel values: {img_array[0, :5, :5, :]}")

        # Make a prediction
        prediction = model.predict(img_array)
        
        # Decode the prediction (for binary classification)
        predicted_class = (prediction > 0.5).astype("int32")
        return predicted_class, prediction
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

# Example usage for inference
img_path = 'aa.jpg'  # Path to an image for prediction
predicted_class, prediction = predict_image(img_path, best_model)
if predicted_class is not None:
    print(f'Predicted class: {predicted_class}')
    print(f'Prediction confidence: {prediction}')
