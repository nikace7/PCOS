import tensorflow as tf
import numpy as np

# Load the best saved model for inference
best_model = tf.keras.models.load_model('best_model.keras')  # Use .h5 if you saved in that format

# Example inference on a single image
def predict_image(img_path, model):
    try:
        # Load and preprocess the image
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(64, 64))  # Match this with your model's input size
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize the image

        # Make a prediction
        prediction = model.predict(img_array)
        
        # Decode the prediction (for binary classification)
        predicted_class = (prediction > 0.5).astype("int32")  # Adjust based on binary classification threshold
        return predicted_class, prediction
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

# Example usage for inference
img_path = 'q.jpg'  # Provide the path to an image for prediction
predicted_class, prediction = predict_image(img_path, best_model)
if predicted_class is not None:
    print(f'Predicted class: {predicted_class}')
    print(f'Prediction confidence: {prediction}')
