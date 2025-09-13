import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    def __init__(self, data_path="data"):
        self.data_path = data_path
        self.food_data = None
        self.recipes_data = None
        self.user_data = None
        
    def load_openfoodfacts_data(self):
        """Load and preprocess Open Food Facts TSV data"""
        file_path = os.path.join(self.data_path, "en.openfoodfacts.org.products.tsv")
        
        print("Loading Open Food Facts data...")
        
        # Define columns we need (to save memory)
        cols_to_use = [
            'product_name', 'energy_100g', 'proteins_100g', 
            'carbohydrates_100g', 'fat_100g', 'fiber_100g',
            'sugars_100g', 'sodium_100g', 'main_category',
            'pnns_groups_1', 'pnns_groups_2'
        ]
        
        # Read TSV file with only necessary columns
        # Use chunking if memory is an issue
        try:
            # First, let's try to load the full file
            self.food_data = pd.read_csv(file_path, sep='\t', usecols=cols_to_use, low_memory=False)
        except MemoryError:
            # If memory error, use chunking
            print("Memory error detected, using chunking...")
            chunks = pd.read_csv(file_path, sep='\t', usecols=cols_to_use, 
                               low_memory=False, chunksize=100000)
            self.food_data = pd.concat(chunks, ignore_index=True)
        
        print(f"Loaded {len(self.food_data)} food products")
        
        # Clean and preprocess
        # Drop rows with missing product name or energy
        self.food_data = self.food_data.dropna(subset=['product_name', 'energy_100g'])
        
        # Remove products with 0 calories (invalid entries)
        self.food_data = self.food_data[self.food_data['energy_100g'] > 0]
        
        # Fill missing nutritional values with 0
        nutrient_cols = ['proteins_100g', 'carbohydrates_100g', 'fat_100g', 
                        'fiber_100g', 'sugars_100g', 'sodium_100g']
        for col in nutrient_cols:
            if col in self.food_data.columns:
                self.food_data[col] = self.food_data[col].fillna(0)
        
        # Remove outliers (unrealistic nutritional values)
        self.food_data = self.food_data[
            (self.food_data['energy_100g'] <= 1000) &  # Max 1000kcal per 100g
            (self.food_data['proteins_100g'] <= 100) &
            (self.food_data['carbohydrates_100g'] <= 100) &
            (self.food_data['fat_100g'] <= 100)
        ]
        
        # Create a simplified category column
        if 'main_category' in self.food_data.columns:
            self.food_data['category'] = self.food_data['main_category'].fillna('Unknown')
        elif 'pnns_groups_1' in self.food_data.columns:
            self.food_data['category'] = self.food_data['pnns_groups_1'].fillna('Unknown')
        else:
            self.food_data['category'] = 'Unknown'
        
        print(f"After cleaning: {len(self.food_data)} food products")
        return self.food_data
    
    def load_foodcom_recipes(self):
        """Load and preprocess Food.com recipes data"""
        file_path = os.path.join(self.data_path, "RAW_recipes.csv")
        
        print("Loading Food.com recipes...")
        
        # Load the recipes data
        self.recipes_data = pd.read_csv(file_path)
        
        print(f"Loaded {len(self.recipes_data)} recipes")
        
        # Parse nutrition column
        # The nutrition column is a string: "[calories, # protein, # fat, # sodium, # carbs, # sugar, # fiber]"
        if 'nutrition' in self.recipes_data.columns:
            # Remove brackets and split
            nutrition_split = self.recipes_data['nutrition'].str.strip('[]').str.split(',', expand=True)
            
            # Convert to numeric
            for i in range(7):
                nutrition_split[i] = pd.to_numeric(nutrition_split[i], errors='coerce')
            
            # Assign to columns
            self.recipes_data['calories'] = nutrition_split[0]
            self.recipes_data['protein'] = nutrition_split[1]
            self.recipes_data['fat'] = nutrition_split[2]
            self.recipes_data['sodium'] = nutrition_split[3]
            self.recipes_data['carbs'] = nutrition_split[4]
            self.recipes_data['sugar'] = nutrition_split[5]
            self.recipes_data['fiber'] = nutrition_split[6]
            
            # Drop original nutrition column
            self.recipes_data = self.recipes_data.drop('nutrition', axis=1)
        
        # Clean ingredients and steps
        if 'ingredients' in self.recipes_data.columns:
            # Convert string representation of list to actual list
            self.recipes_data['ingredients'] = self.recipes_data['ingredients'].apply(
                lambda x: eval(x) if isinstance(x, str) else x
            )
        
        if 'steps' in self.recipes_data.columns:
            # Convert string representation of list to actual list
            self.recipes_data['steps'] = self.recipes_data['steps'].apply(
                lambda x: eval(x) if isinstance(x, str) else x
            )
        
        # Remove recipes with missing calories
        self.recipes_data = self.recipes_data.dropna(subset=['calories'])
        
        # Remove outliers
        self.recipes_data = self.recipes_data[
            (self.recipes_data['calories'] > 0) &
            (self.recipes_data['calories'] <= 5000)  # Max 5000 calories per recipe
        ]
        
        print(f"After cleaning: {len(self.recipes_data)} recipes")
        return self.recipes_data
    
    def create_sample_users(self, n_users=1000):
        """Create synthetic user profiles for training"""
        np.random.seed(42)
        
        # Generate user data
        ages = np.random.randint(18, 65, n_users)
        genders = np.random.choice(['Male', 'Female'], n_users)
        heights = np.random.normal(170, 10, n_users)
        weights = np.random.normal(70, 15, n_users)
        activity_levels = np.random.choice(['Sedentary', 'Light', 'Moderate', 'Active', 'Very Active'], n_users)
        
        # Calculate BMR using Mifflin-St Jeor Equation
        bmr = np.where(
            genders == 'Male',
            10 * weights + 6.25 * heights - 5 * ages + 5,
            10 * weights + 6.25 * heights - 5 * ages - 161
        )
        
        # Apply activity multipliers
        activity_multipliers = {
            'Sedentary': 1.2,
            'Light': 1.375,
            'Moderate': 1.55,
            'Active': 1.725,
            'Very Active': 1.9
        }
        
        daily_calories = bmr * np.array([activity_multipliers[al] for al in activity_levels])
        
        # Create DataFrame
        self.user_data = pd.DataFrame({
            'user_id': range(n_users),
            'age': ages,
            'gender': genders,
            'height': heights,
            'weight': weights,
            'activity_level': activity_levels,
            'bmr': bmr,
            'daily_calories': daily_calories
        })
        
        # Save to CSV
        self.user_data.to_csv(os.path.join(self.data_path, "sample_users.csv"), index=False)
        
        print(f"Created {n_users} sample user profiles")
        return self.user_data
    
    def prepare_training_data(self):
        """Prepare data for ML model training"""
        if self.user_data is None:
            self.create_sample_users()
        
        # Features and target
        X = self.user_data[['age', 'gender', 'height', 'weight', 'activity_level']]
        y = self.user_data['daily_calories']
        
        # Encode categorical variables
        le_gender = LabelEncoder()
        le_activity = LabelEncoder()
        
        X['gender_encoded'] = le_gender.fit_transform(X['gender'])
        X['activity_encoded'] = le_activity.fit_transform(X['activity_level'])
        
        # Drop original categorical columns
        X = X.drop(['gender', 'activity_level'], axis=1)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler,
            'le_gender': le_gender,
            'le_activity': le_activity
        }
    
    def get_food_categories(self):
        """Get unique food categories from Open Food Facts"""
        if self.food_data is not None and 'category' in self.food_data.columns:
            return sorted(self.food_data['category'].unique())
        return []
    
    def get_recipe_categories(self):
        """Get unique recipe categories from Food.com"""
        if self.recipes_data is not None and 'tags' in self.recipes_data.columns:
            # Extract unique tags
            all_tags = []
            for tags in self.recipes_data['tags'].dropna():
                if isinstance(tags, str):
                    all_tags.extend(eval(tags))
            return sorted(list(set(all_tags)))
        return []