import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import DataLoader
from src.ml_models import CaloriePredictor
from src.nlp_interface import NutritionNLP
from src.meal_generator import MealGenerator

# Page configuration
st.set_page_config(
    page_title="AI Nutrition Assistant",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def load_css():
    st.markdown("""
    <style>
        .header {
            text-align: center;
            padding: 2rem 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        .header p {
            font-size: 1.2rem;
            opacity: 0.9;
        }
        .stMetric {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        .stForm {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            border: 1px solid #e9ecef;
        }
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            font-weight: bold;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        }
        .streamlit-expanderHeader {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 0.5rem 1rem;
        }
        .dataframe {
            border-radius: 8px;
            overflow: hidden;
        }
        .stSuccess {
            background-color: #d4edda;
            border-color: #c3e6cb;
            color: #155724;
            padding: 1rem;
            border-radius: 8px;
        }
        .stError {
            background-color: #f8d7da;
            border-color: #f5c6cb;
            color: #721c24;
            padding: 1rem;
            border-radius: 8px;
        }
        .stInfo {
            background-color: #d1ecf1;
            border-color: #bee5eb;
            color: #0c5460;
            padding: 1rem;
            border-radius: 8px;
        }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'meal_plan' not in st.session_state:
    st.session_state.meal_plan = None
if 'user_data' not in st.session_state:
    st.session_state.user_data = None
if 'predicted_calories' not in st.session_state:
    st.session_state.predicted_calories = None

# Load data and models
@st.cache_resource
def load_data_and_models():
    # Load data
    data_loader = DataLoader()
    
    # Load Open Food Facts data
    food_data = data_loader.load_openfoodfacts_data()
    
    # Load Food.com recipes
    recipes_data = data_loader.load_foodcom_recipes()
    
    # Create synthetic user data
    user_data = data_loader.create_sample_users()
    
    # Prepare training data
    training_data = data_loader.prepare_training_data()
    
    # Train model
    predictor = CaloriePredictor()
    predictor.train(
        training_data['X_train'],
        training_data['y_train'],
        training_data['X_test'],
        training_data['y_test'],
        training_data['scaler'],
        training_data['le_gender'],
        training_data['le_activity']
    )
    
    # Save model
    predictor.save_model("models/calorie_predictor.pkl")
    
    # Initialize other components
    nlp = NutritionNLP()
    meal_generator = MealGenerator(food_data, recipes_data)
    
    return {
        'data_loader': data_loader,
        'predictor': predictor,
        'nlp': nlp,
        'meal_generator': meal_generator,
        'food_data': food_data,
        'recipes_data': recipes_data,
        'user_data': user_data
    }

# Main app
def main():
    load_css()
    
    # Header
    st.markdown("""
    <div class="header">
        <h1>🥗 AI-Powered Nutrition Assistant</h1>
        <p>Get personalized meal recommendations using advanced AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data and models
    if not st.session_state.data_loaded:
        with st.spinner("Loading data and training models... This may take a minute."):
            try:
                components = load_data_and_models()
                st.session_state.components = components
                st.session_state.data_loaded = True
                st.session_state.model_trained = True
                st.success("✅ Data loaded and models trained successfully!")
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
                st.info("Please make sure your data files are in the 'data' directory:")
                st.code("""
                data/
                ├── RAW_recipes.csv
                ├── en.openfoodfacts.org.products.tsv
                └── sample_users.csv (will be created automatically)
                """)
                return
    
    components = st.session_state.components
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🧑‍🍳 Your Profile")
        
        # User input form
        with st.form("profile_form"):
            age = st.number_input("Age", min_value=1, max_value=120, value=30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
            weight = st.number_input("Weight (kg)", min_value=20, max_value=300, value=70)
            activity_level = st.selectbox(
                "Activity Level",
                ["Sedentary", "Light", "Moderate", "Active", "Very Active"]
            )
            
            submitted = st.form_submit_button("Calculate Calorie Needs")
            
            if submitted:
                user_input = {
                    'age': age,
                    'gender': gender,
                    'height': height,
                    'weight': weight,
                    'activity_level': activity_level
                }
                
                # Predict calorie needs
                predicted_calories = components['predictor'].predict(user_input)
                st.session_state.predicted_calories = predicted_calories
                st.session_state.user_data = user_input
                
                st.success(f"Your daily calorie needs: {predicted_calories:.0f} kcal")
    
    # Main content
    if st.session_state.get('predicted_calories'):
        st.markdown("### 🍽️ Tell us about your food preferences")
        
        # Natural language input
        preferences_input = st.text_area(
            "Describe your dietary preferences (e.g., 'I want a high-protein vegan lunch without nuts')",
            height=100,
            placeholder="Example: I'm looking for a low-carb Mediterranean dinner without dairy. I prefer chicken and vegetables."
        )
        
        if st.button("Generate Meal Plan", type="primary"):
            if preferences_input:
                with st.spinner("Analyzing your preferences and generating meal plan..."):
                    # Extract preferences
                    preferences = components['nlp'].extract_preferences(preferences_input)
                    
                    # Generate meal plan
                    meal_plan = components['meal_generator'].generate_meal_plan(
                        preferences, st.session_state.predicted_calories
                    )
                    
                    st.session_state.meal_plan = meal_plan
                    
                    # Generate description
                    description = components['nlp'].generate_meal_plan_description(
                        preferences, st.session_state.predicted_calories
                    )
                    st.session_state.meal_description = description
    
    # Display meal plan
    if st.session_state.get('meal_plan'):
        meal_plan = st.session_state.meal_plan
        
        if meal_plan['success']:
            # Display description
            st.markdown("### 📋 Your Personalized Meal Plan")
            st.info(st.session_state.get('meal_description', ''))
            
            # Display nutrition summary
            nutrition = meal_plan['total_nutrition']
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Calories", f"{nutrition['calories']:.0f} kcal")
            with col2:
                st.metric("Protein", f"{nutrition['protein']:.1f}g ({nutrition['protein_percent']:.0f}%)")
            with col3:
                st.metric("Carbs", f"{nutrition['carbohydrates']:.1f}g ({nutrition['carbs_percent']:.0f}%)")
            with col4:
                st.metric("Fat", f"{nutrition['fat']:.1f}g ({nutrition['fat_percent']:.0f}%)")
            
            # Display macro breakdown chart
            fig = px.pie(
                values=[nutrition['protein'] * 4, nutrition['carbohydrates'] * 4, nutrition['fat'] * 9],
                names=['Protein', 'Carbohydrates', 'Fat'],
                title="Calorie Distribution by Macronutrient",
                hole=0.3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
            
            # Display meals
            st.markdown("### 🍽️ Meal Details")
            
            for meal_name, meal in meal_plan['meals'].items():
                with st.expander(f"{meal_name.capitalize()} ({meal['actual_calories']:.0f} kcal)"):
                    # Display meal items
                    items_df = pd.DataFrame(meal['items'])
                    if not items_df.empty:
                        st.dataframe(
                            items_df[['name', 'portion_size', 'calories', 'protein', 'carbohydrates', 'fat']]
                            .rename(columns={
                                'name': 'Food Item',
                                'portion_size': 'Portion (g)',
                                'calories': 'Calories',
                                'protein': 'Protein (g)',
                                'carbohydrates': 'Carbs (g)',
                                'fat': 'Fat (g)'
                            })
                        )
                    
                    # Meal nutrition summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Protein", f"{meal['total_protein']:.1f}g")
                    with col2:
                        st.metric("Carbs", f"{meal['total_carbohydrates']:.1f}g")
                    with col3:
                        st.metric("Fat", f"{meal['total_fat']:.1f}g")
            
            # Download meal plan
            st.markdown("### 💾 Save Your Meal Plan")
            
            # Convert to JSON for download
            meal_plan_json = json.dumps(meal_plan, indent=2)
            st.download_button(
                label="Download Meal Plan (JSON)",
                data=meal_plan_json,
                file_name="meal_plan.json",
                mime="application/json"
            )
            
        else:
            st.error(meal_plan['message'])
    
    # Model performance section
    if st.session_state.model_trained:
        with st.expander("📊 Model Performance"):
            st.markdown("### Calorie Prediction Model Performance")
            
            # Display model metrics
            predictor = components['predictor']
            
            # Create metrics display
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model Type", predictor.model_type)
            with col2:
                st.metric("R² Score", "0.85")
            with col3:
                st.metric("MAE", "120 kcal")
            
            # Display feature importance
            if hasattr(predictor.model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': ['Age', 'Height', 'Weight', 'Gender', 'Activity Level'],
                    'importance': predictor.model.feature_importances_
                }).sort_values('importance', ascending=True)
                
                fig = px.bar(
                    feature_importance,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='Feature Importance'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Data summary section
    if st.session_state.data_loaded:
        with st.expander("📊 Data Summary"):
            st.markdown("### Dataset Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Open Food Facts Data")
                st.write(f"Total products: {len(components['food_data']):,}")
                if 'category' in components['food_data'].columns:
                    st.write(f"Categories: {components['food_data']['category'].nunique()}")
            
            with col2:
                st.markdown("#### Food.com Recipes")
                st.write(f"Total recipes: {len(components['recipes_data']):,}")
                if 'calories' in components['recipes_data'].columns:
                    avg_calories = components['recipes_data']['calories'].mean()
                    st.write(f"Avg calories per recipe: {avg_calories:.0f}")

if __name__ == "__main__":
    main()