import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved model and vectorizer
model = joblib.load('random_forest_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Preprocess the input text
def preprocess_text(text):
    """
    Preprocess the input text by cleaning and normalizing it.
    :param text: Raw resume text.
    :return: Cleaned text.
    """
    # Lowercase the text
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text

# Function to predict using the trained model
def predict_resume_class(resume_text):
    """
    Predict whether the given resume is flagged or unflagged.
    :param resume_text: Raw resume text.
    :return: Prediction (0 = Unflagged, 1 = Flagged).
    """
    # Preprocess the text
    cleaned_text = preprocess_text(resume_text)
    
    # Vectorize the text
    vectorized_text = vectorizer.transform([cleaned_text])
    
    # Predict using the trained model
    prediction = model.predict(vectorized_text)
    
    return int(prediction[0])  # Return the predicted class



if __name__ == "__main__":
    # Example test input
    test_resume = "Experienced teacher with a strong background in education and leadership."
    
    # Get the prediction
    prediction = predict_resume_class(test_resume)
    
    # Print the result
    print("Prediction (0 = Unflagged, 1 = Flagged):", prediction)
