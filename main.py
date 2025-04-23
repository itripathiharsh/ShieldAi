# returnshield_api.py - Pure API version with CORS enabled
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import xgboost as xgb
import random
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')


class ReturnShieldAPI:
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)  # Enable Cross-Origin Resource Sharing
        self.model = None
        self.categories = ['Clothing', 'Electronics', 'Home', 'Beauty', 'Shoes']
        self._initialize_model()
        self._setup_routes()

    def _initialize_model(self):
        """Initialize with pre-trained model (in production you would load a saved model)"""
        print("Initializing model...")
        data = self._generate_training_data()
        self._train_model(data)

    def _generate_training_data(self):
        """Generate synthetic training data"""
        data = []
        for _ in range(1000):
            item = {
                'product_category': random.choice(self.categories),
                'purchase_history': np.random.poisson(15),
                'return_history': np.random.beta(2, 5),
                'price': np.random.uniform(10, 200),
                'return_risk': self._calculate_return_risk()
            }
            data.append(item)
        return pd.DataFrame(data)

    def _calculate_return_risk(self):
        """Calculate synthetic return risk"""
        return np.random.uniform(0.05, 0.95)

    def _train_model(self, data):
        """Train the risk prediction model"""
        data = pd.get_dummies(data, columns=['product_category'])
        features = ['purchase_history', 'return_history', 'price'] + \
                   [col for col in data.columns if col.startswith('product_category_')]
        X = data[features]
        y = (data['return_risk'] > 0.7).astype(int)  # 70% threshold for high risk

        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1
        )
        self.model.fit(X, y)

    def _setup_routes(self):
        @self.app.route('/predict-risk', methods=['POST'])
        def predict_risk():
            """API endpoint for return risk prediction"""
            data = request.json
            required_fields = ['product_category', 'purchase_history', 'return_history', 'price']
            if not all(field in data for field in required_fields):
                return jsonify({"error": "Missing required fields"}), 400

            input_data = {
                'purchase_history': data['purchase_history'],
                'return_history': data['return_history'],
                'price': data['price'],
                f"product_category_{data['product_category']}": 1
            }

            try:
                risk_prob = self._predict_risk(input_data)
                return jsonify({
                    'risk_level': risk_prob,
                    'risk_category': 'high' if risk_prob >= 0.7 else 'medium' if risk_prob >= 0.4 else 'low'
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/sizing-advice', methods=['POST'])
        def sizing_advice():
            """API endpoint for sizing advice"""
            data = request.json
            if 'product_category' not in data or 'question' not in data:
                return jsonify({"error": "Missing product_category or question"}), 400

            advice = self._get_sizing_advice(data['product_category'], data['question'])
            return jsonify({'advice': advice})

        @self.app.route('/predict-risk-by-name', methods=['POST'])
        def predict_risk_by_name():
            """API endpoint to predict risk from product name"""
            data = request.json
            if 'product_name' not in data:
                return jsonify({"error": "Missing product_name"}), 400

            category = self._infer_category_from_name(data['product_name'])
            if not category:
                return jsonify({"error": "Could not infer category from name"}), 400

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

            try:
                risk_prob = self._predict_risk(input_data)
                return jsonify({
                    'product_name': data['product_name'],
                    'inferred_category': category,
                    'risk_level': risk_prob,
                    'risk_category': 'high' if risk_prob >= 0.7 else 'medium' if risk_prob >= 0.4 else 'low'
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500

    def _infer_category_from_name(self, name):
        """Infer category based on product name keywords"""
        name = name.lower()
        if any(keyword in name for keyword in ['shirt', 't-shirt', 'jeans', 'jacket', 'hoodie']):
            return 'Clothing'
        elif any(keyword in name for keyword in ['phone', 'laptop', 'headphones', 'camera']):
            return 'Electronics'
        elif any(keyword in name for keyword in ['sofa', 'table', 'lamp', 'mattress']):
            return 'Home'
        elif any(keyword in name for keyword in ['cream', 'lotion', 'shampoo', 'perfume']):
            return 'Beauty'
        elif any(keyword in name for keyword in ['shoe', 'sneaker', 'boot', 'heel']):
            return 'Shoes'
        return None

    def _predict_risk(self, input_data):
        """Make a risk prediction"""
        input_df = pd.DataFrame([input_data])
        expected_features = self.model.get_booster().feature_names
        for feat in expected_features:
            if feat not in input_df.columns:
                input_df[feat] = 0
        input_df = input_df[expected_features]
        return float(self.model.predict_proba(input_df)[0][1])

    def _get_sizing_advice(self, product_category, question):
        """Get sizing advice (mock implementation)"""
        advice_map = {
            "Clothing": {
                "runs small": "This item tends to run small, consider sizing up",
                "true to size": "Most customers find this true to size",
                "fit": "The fit is generally regular, check the size chart"
            },
            "Shoes": {
                "size up": "If you have wide feet, consider sizing up",
                "narrow": "This style runs narrow in the toe box"
            }
        }

        product_advice = advice_map.get(product_category, {})
        for keyword, response in product_advice.items():
            if keyword in question.lower():
                return response

        return "Please check the product size chart for accurate measurements"

    def run(self, host='0.0.0.0', port=5000):
        """Run the API server"""
        print(f"Starting ReturnShield API on http://{host}:{port}")
        print("Available endpoints:")
        print("POST /predict-risk - Predict return risk")
        print("POST /predict-risk-by-name - Predict return risk using product name only")
        print("POST /sizing-advice - Get sizing advice")
        self.app.run(host=host, port=port)




def run(self):
    port = int(os.environ.get('PORT', 5000))
    self.app.run(host='0.0.0.0', port=port)

