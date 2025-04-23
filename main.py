from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import xgboost as xgb
import random
import warnings
import os
from waitress import serve  # Production server

# Suppress warnings
warnings.filterwarnings('ignore')


class ReturnShieldAPI:
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for all routes
        self.model = None
        self.categories = ['Clothing', 'Electronics', 'Home', 'Beauty', 'Shoes']
        self._initialize_model()
        self._setup_routes()
        self._add_health_check()

    def _initialize_model(self):
        """Initialize with pre-trained model"""
        print("Initializing model...")
        data = self._generate_training_data()
        self._train_model(data)

    def _add_health_check(self):
        """Add health check endpoint for Render"""

        @self.app.route('/health')
        def health_check():
            return jsonify({"status": "healthy", "service": "ReturnShieldAPI"}), 200

    def _generate_training_data(self):
        """Generate synthetic training data with more realistic patterns"""
        data = []
        for _ in range(1000):
            category = random.choice(self.categories)
            base_risk = {
                'Clothing': 0.35,
                'Shoes': 0.3,
                'Electronics': 0.15,
                'Home': 0.2,
                'Beauty': 0.1
            }[category]

            data.append({
                'product_category': category,
                'purchase_history': np.random.poisson(15),
                'return_history': np.random.beta(2, 5),
                'price': np.random.uniform(10, 200),
                'return_risk': min(max(base_risk + np.random.normal(0, 0.1), 0.05), 0.95)
            })
        return pd.DataFrame(data)

    def _train_model(self, data):
        """Train the risk prediction model with improved parameters"""
        data = pd.get_dummies(data, columns=['product_category'])
        features = ['purchase_history', 'return_history', 'price'] + \
                   [col for col in data.columns if col.startswith('product_category_')]
        X = data[features]
        y = (data['return_risk'] > 0.7).astype(int)

        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        self.model.fit(X, y)

    def _setup_routes(self):
        @self.app.route('/predict-risk', methods=['POST'])
        def predict_risk():
            """Enhanced risk prediction endpoint with better validation"""
            if not request.is_json:
                return jsonify({"error": "Request must be JSON"}), 400

            data = request.get_json()
            required_fields = ['product_category', 'purchase_history', 'return_history', 'price']

            if not all(field in data for field in required_fields):
                missing = [field for field in required_fields if field not in data]
                return jsonify({"error": f"Missing required fields: {missing}"}), 400

            try:
                input_data = {
                    'purchase_history': float(data['purchase_history']),
                    'return_history': float(data['return_history']),
                    'price': float(data['price']),
                    f"product_category_{data['product_category']}": 1
                }

                risk_prob = self._predict_risk(input_data)
                return jsonify({
                    'risk_level': risk_prob,
                    'risk_category': self._get_risk_category(risk_prob),
                    'confidence': round(1 - abs(risk_prob - 0.5) * 2, 2)  # Confidence metric
                })
            except ValueError as e:
                return jsonify({"error": f"Invalid data format: {str(e)}"}), 400
            except Exception as e:
                return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

        @self.app.route('/predict-risk-by-name', methods=['POST'])
        def predict_risk_by_name():
            """Improved product name inference endpoint"""
            if not request.is_json:
                return jsonify({"error": "Request must be JSON"}), 400

            data = request.get_json()
            if 'product_name' not in data:
                return jsonify({"error": "Missing product_name"}), 400

            category = self._infer_category_from_name(data['product_name'])
            if not category:
                return jsonify({"error": "Could not infer category from product name"}), 400

            try:
                synthetic_data = {
                    'product_category': category,
                    'purchase_history': np.random.poisson(15),
                    'return_history': np.random.beta(2, 5),
                    'price': np.random.uniform(10, 200)
                }

                input_data = {
                    'purchase_history': synthetic_data['purchase_history'],
                    'return_history': synthetic_data['return_history'],
                    'price': synthetic_data['price'],
                    f"product_category_{synthetic_data['product_category']}": 1
                }

                risk_prob = self._predict_risk(input_data)
                return jsonify({
                    'product_name': data['product_name'],
                    'inferred_category': category,
                    'risk_level': risk_prob,
                    'risk_category': self._get_risk_category(risk_prob),
                    'note': 'Values generated synthetically based on category'
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500

    def _get_risk_category(self, risk_prob):
        """Categorize risk with clear thresholds"""
        if risk_prob >= 0.7:
            return 'high'
        elif risk_prob >= 0.4:
            return 'medium'
        return 'low'

    def _infer_category_from_name(self, name):
        """Enhanced category inference with more keywords"""
        name = name.lower()
        category_keywords = {
            'Clothing': ['shirt', 't-shirt', 'jeans', 'jacket', 'hoodie', 'dress', 'pants'],
            'Electronics': ['phone', 'laptop', 'headphone', 'camera', 'tablet', 'charger'],
            'Home': ['sofa', 'table', 'lamp', 'mattress', 'chair', 'desk', 'bed'],
            'Beauty': ['cream', 'lotion', 'shampoo', 'perfume', 'makeup', 'serum'],
            'Shoes': ['shoe', 'sneaker', 'boot', 'heel', 'sandals', 'loafer']
        }

        for category, keywords in category_keywords.items():
            if any(keyword in name for keyword in keywords):
                return category
        return None

    def _predict_risk(self, input_data):
        """Robust prediction handling"""
        input_df = pd.DataFrame([input_data])
        expected_features = self.model.get_booster().feature_names

        # Ensure all expected features exist
        for feat in expected_features:
            if feat not in input_df.columns:
                input_df[feat] = 0

        input_df = input_df[expected_features]
        return float(self.model.predict_proba(input_df)[0][1])

    def run(self):
        """Run with production-ready server"""
        port = int(os.environ.get('PORT', 5000))
        print(f"Starting ReturnShield API on port {port}")
        serve(self.app, host='0.0.0.0', port=port)  # Production server


if __name__ == '__main__':
    api = ReturnShieldAPI()
    api.run()