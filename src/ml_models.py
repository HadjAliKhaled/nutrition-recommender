import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class CaloriePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.le_gender = None
        self.le_activity = None
        self.feature_names = ['age', 'height', 'weight', 'gender_encoded', 'activity_encoded']
        self.model_type = None
        
    def train(self, X_train, y_train, X_test, y_test, scaler, le_gender, le_activity):
        """Train multiple models and select the best one"""
        self.scaler = scaler
        self.le_gender = le_gender
        self.le_activity = le_activity
        
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(
                n_estimators=200, 
                max_depth=15, 
                min_samples_split=5,
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                random_state=42
            )
        }
        
        results = {}
        
        for name, model in models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            results[name] = {
                'model': model,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_mae': test_mae,
                'test_rmse': test_rmse
            }
            
            print(f"{name}:")
            print(f"  Train R²: {train_r2:.4f}")
            print(f"  Test R²: {test_r2:.4f}")
            print(f"  Test MAE: {test_mae:.2f} calories")
            print(f"  Test RMSE: {test_rmse:.2f} calories")
            print()
        
        # Select best model based on test R2
        best_model_name = max(results, key=lambda x: results[x]['test_r2'])
        self.model = results[best_model_name]['model']
        self.model_type = best_model_name
        
        print(f"Selected best model: {best_model_name}")
        print(f"Best Test R²: {results[best_model_name]['test_r2']:.4f}")
        print(f"Best Test MAE: {results[best_model_name]['test_mae']:.2f} calories")
        
        return results
    
    def predict(self, user_input):
        """Predict calorie needs for a user"""
        # Encode categorical variables
        gender_encoded = self.le_gender.transform([user_input['gender']])[0]
        activity_encoded = self.le_activity.transform([user_input['activity_level']])[0]
        
        # Create feature array
        features = np.array([[
            user_input['age'],
            user_input['height'],
            user_input['weight'],
            gender_encoded,
            activity_encoded
        ]])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        predicted_calories = self.model.predict(features_scaled)[0]
        
        return predicted_calories
    
    def save_model(self, path):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'le_gender': self.le_gender,
            'le_activity': self.le_activity,
            'model_type': self.model_type
        }
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load a trained model"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.le_gender = model_data['le_gender']
        self.le_activity = model_data['le_activity']
        self.model_type = model_data['model_type']
        print(f"Model loaded from {path}")
        print(f"Model type: {self.model_type}")
    
    def get_feature_importance(self):
        """Get feature importance for tree-based models"""
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_names, self.model.feature_importances_))
        return None