from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import predict_resume_class

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from the request
        input_data = request.json.get('resume', '')

        if not input_data:
            return jsonify({'error': 'No resume text provided'}), 400
        
        # Get prediction
        prediction = predict_resume_class(input_data)

        # Return prediction as JSON
        return jsonify({'prediction': prediction}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
