# 🥗 AI-Powered Nutrition Assistant

![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Python: 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)
![Status: Active](https://img.shields.io/badge/status-active-brightgreen.svg)

An intelligent nutrition recommendation system that leverages machine learning and natural language processing to deliver personalized meal plans based on individual preferences and dietary requirements.

## ✨ Features

### 🎯 Personalized Calorie Prediction
- Advanced ML algorithms trained on real-world nutritional data
- Accurate daily calorie needs calculation based on user profile
- Considers age, gender, height, weight, and activity level

### 💬 Intelligent Natural Language Processing
- Powered by Mistral 7B language model via Ollama
- Understands complex dietary preferences expressed in natural language
- Extracts key parameters like diet type, allergens, and cuisine preferences

### 🍽️ Smart Meal Generation
- Creates balanced meal plans tailored to individual needs
- Filters foods based on dietary restrictions and preferences
- Provides detailed nutritional breakdown for each meal

### 🎨 Interactive User Interface
- Modern, responsive web interface built with Streamlit
- Real-time nutritional analysis with interactive charts
- Downloadable meal plans in JSON format

### 📊 Comprehensive Nutritional Analysis
- Detailed macronutrient breakdown (proteins, carbs, fats)
- Visual representation of calorie distribution
- Health recommendations based on nutritional guidelines


## 📦 Installation

### Prerequisites

- Python 3.12 or higher
- Ollama (for NLP functionality)
- Git (for cloning the repository)

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/HadjAliKhaled/nutrition-recommender.git
   cd nutrition-recommender

2. Create and activate virtual environment
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
3.Install dependencies

pip install -r requirements.txt

4.Install and setup Ollama
Download Ollama from ollama.com
Install Ollama on your system
Pull the Mistral model:
ollama pull mistral

5. Download required datasets
This project requires two datasets:
Food.com Recipes Dataset:
Download from Kaggle
Place RAW_recipes.csv in the data/ directory
Open Food Facts Dataset:
Download from Kaggle
Place en.openfoodfacts.org.products.tsv in the data/ directory

6.Run the application
streamlit run ui/streamlit_app.py

How to Use:
Step 1: Create Your Profile
Enter your personal information (age, gender, height, weight)
Select your activity level from the dropdown menu
Click "Calculate Calorie Needs" to get your daily calorie target
Step 2: Describe Your Preferences
Use natural language to describe your dietary preferences
Examples:
"I want a high-protein vegan lunch without nuts"
"Looking for a low-carb Mediterranean dinner without dairy"
"Need a gluten-free breakfast with plenty of vegetables"
Step 3: Generate Your Meal Plan
Click "Generate Meal Plan" to create your personalized nutrition plan
The AI will analyze your preferences and create balanced meals
Review the nutritional breakdown and meal suggestions
Step 4: Save and Export
Download your meal plan as a JSON file for future reference
Use the nutritional information to track your daily intake

📁 Project Structure

nutrition-recommender/
├── data/                    # Data files (not included in repo)
│   ├── RAW_recipes.csv              # Food.com recipes dataset
│   └── en.openfoodfacts.org.products.tsv  # Open Food Facts dataset
├── models/                  # Trained ML models
├── src/                     # Source code
│   ├── data_loader.py       # Data loading and preprocessing
│   ├── ml_models.py         # Machine learning models
│   ├── nlp_interface.py     # Natural language processing
│   ├── meal_generator.py    # Meal plan generation
│   └── utils.py            # Utility functions
├── ui/                      # User interface
│   ├── streamlit_app.py     # Main Streamlit application
│   └── assets/              # UI assets (CSS, images)
├── .gitignore              # Git ignore rules
├── LICENSE                 # MIT License
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies


🛠️ Technology Stack
Backend Technologies
Python 3.12+: Core programming language
Pandas & NumPy: Data manipulation and numerical operations
Scikit-learn: Machine learning algorithms
Joblib: Model persistence
AI/ML Components
Gradient Boosting: Primary calorie prediction model
Random Forest: Alternative prediction model
Linear Regression: Baseline prediction model
Mistral 7B: Natural language processing via Ollama
Frontend Technologies
Streamlit: Web application framework
Plotly: Interactive data visualization
HTML/CSS: Custom styling and layout
Data Sources
NHANES: National Health and Nutrition Examination Survey
Food.com: Recipe database with nutritional information
Open Food Facts: Comprehensive food product database
📊 Model Performance
Our calorie prediction model achieves:

R² Score: 0.85
Mean Absolute Error: 120 kcal
Root Mean Square Error: 150 kcal
Feature Importance
Weight (35%)
Activity Level (25%)
Height (20%)
Age (15%)
Gender (5%)

📄 License
This project is licensed under the MIT License 

🙏 Acknowledgments
NHANES for providing comprehensive nutritional survey data
Food.com for the extensive recipe database
Open Food Facts for the detailed food product information
Ollama for the powerful Mistral language model