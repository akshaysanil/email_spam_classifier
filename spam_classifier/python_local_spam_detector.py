import joblib

# Load the model and the vectorizer
model = joblib.load('spam_classifier/spam_detection_model_tf_idf_naive_bayes.pkl')
vectorizer = joblib.load('spam_classifier/tf_idf_vecorizer.pkl')

def predict_spam(text):
    # Preprocess the text data
    text_vectorized = vectorizer.transform([text])
    print(text_vectorized)
    
    # Predict
    prediction = model.predict(text_vectorized)
    print('prediction',prediction)
    result = 'spam' if prediction[0] == 1 else 'not spam'
    
    return result

if __name__ == '__main__':
    while True:
        # Take input from the user
        user_input = input("Enter a message to classify (or type 'exit' to quit): ")
        user_input.lower()
        
        if user_input.lower() == 'exit':
            break
        
        # Get prediction
        result = predict_spam(user_input)
        
        # Output the result
        print(f"The message is: {result}")
