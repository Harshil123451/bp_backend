from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import joblib
from feature_utils import resample_to_30fps, extract_features
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models
try:
    ridge_model = joblib.load("model/ridge_model_systolic.pkl")
    lasso_model = joblib.load("model/lasso_model_diastolic.pkl")
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info("Received video prediction request")
        if 'video' not in request.files:
            logger.error("No video file in request")
            return jsonify({'error': 'No video file uploaded'}), 400
            
        video = request.files['video']
        if video.filename == '':
            logger.error("Empty filename received")
            return jsonify({'error': 'No selected file'}), 400
            
        filename = secure_filename(video.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        logger.info(f"Saving video to {filepath}")
        video.save(filepath)

        logger.info("Resampling video to 30fps")
        resampled = resample_to_30fps(filepath, filepath.replace(".mp4", "_30fps.mp4"))
        
        logger.info("Extracting features")
        features = extract_features(resampled)
        logger.info(f"Extracted features: {features}")

        X = [[features['best_mean'], features['best_std'], features['estimated_hr'], features['estimated_hr']]]
        logger.info("Predicting blood pressure values")
        sys = ridge_model.predict(X)[0]
        dia = lasso_model.predict(X)[0]

        result = {
            'systolic': round(float(sys), 2),
            'diastolic': round(float(dia), 2)
        }
        logger.info(f"Prediction result: {result}")
        return jsonify(result)

    except Exception as e:
        error_msg = f"Error processing video: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
