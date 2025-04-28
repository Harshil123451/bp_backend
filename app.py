from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import joblib
from feature_utils import resample_to_30fps, extract_features
import traceback
import logging
import platform
import psutil
import time

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

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify server is running"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'uptime': time.time() - psutil.Process().create_time()
    })

@app.route('/test', methods=['GET'])
def test_endpoint():
    """Test endpoint to verify basic functionality"""
    return jsonify({
        'message': 'Server is running',
        'status': 'success',
        'timestamp': time.time()
    })

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded ML models"""
    try:
        ridge_features = ridge_model.n_features_in_ if hasattr(ridge_model, 'n_features_in_') else 'unknown'
        lasso_features = lasso_model.n_features_in_ if hasattr(lasso_model, 'n_features_in_') else 'unknown'
        
        return jsonify({
            'systolic_model': {
                'type': type(ridge_model).__name__,
                'features': ridge_features,
                'status': 'loaded'
            },
            'diastolic_model': {
                'type': type(lasso_model).__name__,
                'features': lasso_features,
                'status': 'loaded'
            }
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/status', methods=['GET'])
def server_status():
    """Get detailed server status information"""
    try:
        return jsonify({
            'status': 'running',
            'system': {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'memory_usage': f"{psutil.Process().memory_percent():.2f}%",
                'cpu_usage': f"{psutil.Process().cpu_percent():.2f}%"
            },
            'models': {
                'systolic_loaded': ridge_model is not None,
                'diastolic_loaded': lasso_model is not None
            },
            'storage': {
                'upload_folder': UPLOAD_FOLDER,
                'folder_exists': os.path.exists(UPLOAD_FOLDER)
            }
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

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
