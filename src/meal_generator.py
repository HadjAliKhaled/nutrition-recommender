import pandas as pd
import numpy as np
from typing import List, Dict, Any
import random

class MealGenerator:
    def __init__(self, food_data: pd.DataFrame, recipes_data: pd.DataFrame = None):
        self.food_data = food_data
        self.recipes_data = recipes_data
        self.nutrient_targets = {
            'protein': (0.1, 0.35),  # 10-35% of calories
            'fat': (0.2, 0.35),     # 20-35% of calories
            'carbohydrates': (0.45, 0.65)  # 45-65% of calories
        }
    
    def generate_meal_plan(self, preferences: Dict[str, Any], daily_calories: float) -> Dict[str, Any]:
        """Generate a personalized meal plan"""
        # Filter foods based on preferences
        filtered_foods = self._filter_foods(preferences)
        
        if filtered_foods.empty:
            return {
                'success': False,
                'message': 'No foods match your preferences. Please try different preferences.'
            }
        
        # Calculate meal calorie distribution
        meal_distribution = self._get_meal_distribution(preferences.get('meal_type'))
        
        # Generate meals
        meals = {}
        for meal_type, calorie_percent in meal_distribution.items():
            meal_calories = daily_calories * calorie_percent
            meals[meal_type] = self._generate_single_meal(
                filtered_foods, meal_calories, meal_type, preferences
            )
        
        # Calculate total nutrition
        total_nutrition = self._calculate_total_nutrition(meals)
        
        return {
            'success': True,
            'daily_calories': daily_calories,
            'meals': meals,
            'total_nutrition': total_nutrition,
            'preferences': preferences
        }
    
    def _filter_foods(self, preferences: Dict[str, Any]) -> pd.DataFrame:
        """Filter foods based on dietary preferences"""
        filtered = self.food_data.copy()
        
        # Apply diet type filters
        if preferences.get('diet_type'):
            diet_type = preferences['diet_type'].lower()
            
            if diet_type == 'vegan':
                # Remove animal products
                filtered = filtered[
                    ~filtered['product_name'].str.contains(
                        'meat|poultry|fish|dairy|egg|cheese|milk|butter|cream|honey|gelatin',
                        case=False, na=False
                    )
                ]
            elif diet_type == 'vegetarian':
                # Remove meat and fish but keep dairy and eggs
                filtered = filtered[
                    ~filtered['product_name'].str.contains(
                        'meat|poultry|fish|seafood|bacon|ham|sausage',
                        case=False, na=False
                    )
                ]
            elif diet_type == 'keto':
                # Keep low-carb foods
                if 'carbohydrates_100g' in filtered.columns:
                    filtered = filtered[filtered['carbohydrates_100g'] < 10]
            elif diet_type == 'low_fat':
                if 'fat_100g' in filtered.columns:
                    filtered = filtered[filtered['fat_100g'] < 5]
        
        # Apply allergen filters
        allergen_patterns = {
            'nuts': 'nuts|peanuts|almonds|walnuts|cashews|pecans',
            'dairy': 'dairy|milk|cheese|yogurt|butter|cream|whey',
            'gluten': 'wheat|barley|rye|bread|pasta|cereal|flour',
            'eggs': 'egg|eggs|mayonnaise',
            'soy': 'soy|soya|tofu|edamame',
            'fish': 'fish|seafood|shellfish|shrimp|crab|lobster',
            'sesame': 'sesame|tahini'
        }
        
        for allergen in preferences.get('allergens', []):
            if allergen in allergen_patterns:
                pattern = allergen_patterns[allergen]
                filtered = filtered[
                    ~filtered['product_name'].str.contains(
                        pattern, case=False, na=False
                    )
                ]
        
        # Apply cuisine filter
        if preferences.get('cuisine'):
            cuisine = preferences['cuisine'].lower()
            cuisine_keywords = {
                'italian': 'pasta|pizza|risotto|parmesan|mozzarella',
                'mexican': 'taco|burrito|quesadilla|salsa|avocado',
                'asian': 'rice|noodle|soy|ginger|sesame',
                'american': 'burger|bbq|steak|potato',
                'french': 'croissant|baguette|quiche|cheese',
                'mediterranean': 'olive|feta|hummus|pita|lamb'
            }
            
            if cuisine in cuisine_keywords:
                pattern = cuisine_keywords[cuisine]
                filtered = filtered[
                    filtered['product_name'].str.contains(
                        pattern, case=False, na=False
                    )
                ]
        
        return filtered
    
    def _get_meal_distribution(self, meal_type: str = None) -> Dict[str, float]:
        """Get calorie distribution for meals"""
        if meal_type == 'breakfast':
            return {'breakfast': 1.0}
        elif meal_type == 'lunch':
            return {'lunch': 1.0}
        elif meal_type == 'dinner':
            return {'dinner': 1.0}
        elif meal_type == 'snack':
            return {'snack': 1.0}
        else:
            return {
                'breakfast': 0.25,
                'lunch': 0.35,
                'dinner': 0.30,
                'snack': 0.10
            }
    
    def _generate_single_meal(self, foods: pd.DataFrame, target_calories: float, 
                            meal_type: str, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a single meal"""
        # Select random foods that fit the calorie target
        meal_items = []
        total_calories = 0
        total_protein = 0
        total_carbs = 0
        total_fat = 0
        
        # Try to create a balanced meal
        attempts = 0
        while total_calories < target_calories * 0.8 and attempts < 50:
            # Select a random food
            food = foods.sample(1).iloc[0]
            
            # Determine portion size (in grams)
            portion_size = min(100, target_calories / (food['energy_100g'] + 1) * 100)
            
            # Calculate nutrition for this portion
            item_calories = (food['energy_100g'] * portion_size) / 100
            item_protein = (food.get('proteins_100g', 0) * portion_size) / 100
            item_carbs = (food.get('carbohydrates_100g', 0) * portion_size) / 100
            item_fat = (food.get('fat_100g', 0) * portion_size) / 100
            
            # Add to meal if it doesn't exceed target too much
            if total_calories + item_calories <= target_calories * 1.2:
                meal_items.append({
                    'name': food['product_name'],
                    'portion_size': round(portion_size, 1),
                    'calories': round(item_calories, 1),
                    'protein': round(item_protein, 1),
                    'carbohydrates': round(item_carbs, 1),
                    'fat': round(item_fat, 1)
                })
                
                total_calories += item_calories
                total_protein += item_protein
                total_carbs += item_carbs
                total_fat += item_fat
            
            attempts += 1
        
        return {
            'meal_type': meal_type,
            'target_calories': round(target_calories, 1),
            'actual_calories': round(total_calories, 1),
            'items': meal_items,
            'total_protein': round(total_protein, 1),
            'total_carbohydrates': round(total_carbs, 1),
            'total_fat': round(total_fat, 1)
        }
    
    def _calculate_total_nutrition(self, meals: Dict[str, Any]) -> Dict[str, float]:
        """Calculate total nutrition for the day"""
        total_calories = sum(meal['actual_calories'] for meal in meals.values())
        total_protein = sum(meal['total_protein'] for meal in meals.values())
        total_carbs = sum(meal['total_carbohydrates'] for meal in meals.values())
        total_fat = sum(meal['total_fat'] for meal in meals.values())
        
        return {
            'calories': round(total_calories, 1),
            'protein': round(total_protein, 1),
            'carbohydrates': round(total_carbs, 1),
            'fat': round(total_fat, 1),
            'protein_percent': round((total_protein * 4 / total_calories) * 100, 1) if total_calories > 0 else 0,
            'carbs_percent': round((total_carbs * 4 / total_calories) * 100, 1) if total_calories > 0 else 0,
            'fat_percent': round((total_fat * 9 / total_calories) * 100, 1) if total_calories > 0 else 0
        }
    
    def get_nutrition_breakdown(self, nutrition: Dict[str, float]) -> Dict[str, Any]:
        """Get detailed nutrition breakdown with recommendations"""
        breakdown = {
            'calories': {
                'value': nutrition['calories'],
                'unit': 'kcal',
                'status': 'good'
            },
            'protein': {
                'value': nutrition['protein'],
                'unit': 'g',
                'percent': nutrition['protein_percent'],
                'status': 'good' if 10 <= nutrition['protein_percent'] <= 35 else 'warning'
            },
            'carbohydrates': {
                'value': nutrition['carbohydrates'],
                'unit': 'g',
                'percent': nutrition['carbs_percent'],
                'status': 'good' if 45 <= nutrition['carbs_percent'] <= 65 else 'warning'
            },
            'fat': {
                'value': nutrition['fat'],
                'unit': 'g',
                'percent': nutrition['fat_percent'],
                'status': 'good' if 20 <= nutrition['fat_percent'] <= 35 else 'warning'
            }
        }
        
        return breakdown