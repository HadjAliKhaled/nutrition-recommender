import ollama
import json
import re
from typing import Dict, List, Any

class NutritionNLP:
    def __init__(self, model_name="mistral"):
        self.model_name = model_name
        self.dietary_keywords = {
            'diet_type': {
                'vegan': ['vegan', 'plant-based', 'no animal products'],
                'vegetarian': ['vegetarian', 'no meat', 'no fish'],
                'keto': ['keto', 'ketogenic', 'low carb', 'no carbs'],
                'paleo': ['paleo', 'caveman', 'ancestral'],
                'mediterranean': ['mediterranean', 'med', 'greek', 'italian'],
                'low_fat': ['low fat', 'fat free', 'no fat'],
                'low_sodium': ['low sodium', 'no salt', 'salt free'],
                'low_sugar': ['low sugar', 'sugar free', 'no sugar'],
                'high_protein': ['high protein', 'protein rich', 'protein packed'],
                'high_fiber': ['high fiber', 'fiber rich', 'high fibre']
            },
            'allergens': {
                'nuts': ['nuts', 'peanuts', 'almonds', 'walnuts', 'cashews'],
                'dairy': ['dairy', 'milk', 'cheese', 'yogurt', 'butter'],
                'gluten': ['gluten', 'wheat', 'barley', 'rye', 'bread'],
                'eggs': ['eggs', 'egg'],
                'soy': ['soy', 'soya', 'tofu'],
                'fish': ['fish', 'seafood', 'shellfish'],
                'sesame': ['sesame', 'tahini']
            },
            'meal_type': {
                'breakfast': ['breakfast', 'morning', 'brunch'],
                'lunch': ['lunch', 'noon', 'midday'],
                'dinner': ['dinner', 'supper', 'evening'],
                'snack': ['snack', 'snacks', 'bite'],
                'dessert': ['dessert', 'sweet', 'treat']
            },
            'cuisine': {
                'italian': ['italian', 'pasta', 'pizza', 'risotto'],
                'mexican': ['mexican', 'taco', 'burrito', 'quesadilla'],
                'asian': ['asian', 'chinese', 'japanese', 'thai', 'indian'],
                'american': ['american', 'burger', 'bbq', 'steak'],
                'french': ['french', 'croissant', 'quiche', 'souffle'],
                'mediterranean': ['mediterranean', 'greek', 'hummus', 'falafel']
            }
        }
    
    def extract_preferences(self, user_input: str) -> Dict[str, Any]:
        """Extract dietary preferences from natural language input"""
        # Create a structured prompt for the LLM
        prompt = f"""
        You are a nutrition expert AI assistant. Extract dietary preferences from the user's input.
        
        User input: "{user_input}"
        
        Return a JSON object with the following structure:
        {{
            "diet_type": "string or null",
            "allergens": ["list of strings"],
            "avoided_ingredients": ["list of strings"],
            "preferred_ingredients": ["list of strings"],
            "meal_type": "string or null",
            "cuisine": "string or null",
            "health_goals": ["list of strings"],
            "special_requests": ["list of strings"]
        }}
        
        Rules:
        1. If a category is not mentioned, use null for strings or empty array for lists.
        2. Be specific with ingredients (e.g., "chicken" instead of "meat").
        3. Include both explicit and implicit preferences.
        4. For health_goals, include things like "weight loss", "muscle gain", "energy boost", etc.
        5. For special_requests, include things like "quick to prepare", "budget-friendly", etc.
        
        Return only valid JSON without any additional text.
        """
        
        try:
            # Get response from LLM
            response = ollama.generate(model=self.model_name, prompt=prompt)
            response_text = response['response']
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                preferences = json.loads(json_match.group())
                return self._post_process_preferences(preferences)
            else:
                return self._fallback_extraction(user_input)
                
        except Exception as e:
            print(f"LLM extraction failed: {e}")
            return self._fallback_extraction(user_input)
    
    def _post_process_preferences(self, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and standardize extracted preferences"""
        # Standardize diet_type
        if preferences.get('diet_type'):
            diet_type = preferences['diet_type'].lower()
            for key, synonyms in self.dietary_keywords['diet_type'].items():
                if any(syn in diet_type for syn in synonyms):
                    preferences['diet_type'] = key
                    break
        
        # Standardize allergens
        allergens = []
        for allergen in preferences.get('allergens', []):
            allergen_lower = allergen.lower()
            for key, synonyms in self.dietary_keywords['allergens'].items():
                if any(syn in allergen_lower for syn in synonyms):
                    allergens.append(key)
                    break
        preferences['allergens'] = list(set(allergens))
        
        # Standardize meal_type
        if preferences.get('meal_type'):
            meal_type = preferences['meal_type'].lower()
            for key, synonyms in self.dietary_keywords['meal_type'].items():
                if any(syn in meal_type for syn in synonyms):
                    preferences['meal_type'] = key
                    break
        
        # Standardize cuisine
        if preferences.get('cuisine'):
            cuisine = preferences['cuisine'].lower()
            for key, synonyms in self.dietary_keywords['cuisine'].items():
                if any(syn in cuisine for syn in synonyms):
                    preferences['cuisine'] = key
                    break
        
        return preferences
    
    def _fallback_extraction(self, user_input: str) -> Dict[str, Any]:
        """Fallback method using keyword matching"""
        input_lower = user_input.lower()
        preferences = {
            "diet_type": None,
            "allergens": [],
            "avoided_ingredients": [],
            "preferred_ingredients": [],
            "meal_type": None,
            "cuisine": None,
            "health_goals": [],
            "special_requests": []
        }
        
        # Extract diet type
        for diet_type, keywords in self.dietary_keywords['diet_type'].items():
            if any(keyword in input_lower for keyword in keywords):
                preferences['diet_type'] = diet_type
                break
        
        # Extract allergens
        for allergen, keywords in self.dietary_keywords['allergens'].items():
            if any(keyword in input_lower for keyword in keywords):
                preferences['allergens'].append(allergen)
        
        # Extract meal type
        for meal_type, keywords in self.dietary_keywords['meal_type'].items():
            if any(keyword in input_lower for keyword in keywords):
                preferences['meal_type'] = meal_type
                break
        
        # Extract cuisine
        for cuisine, keywords in self.dietary_keywords['cuisine'].items():
            if any(keyword in input_lower for keyword in keywords):
                preferences['cuisine'] = cuisine
                break
        
        # Extract health goals
        health_goals = {
            'weight loss': ['lose weight', 'weight loss', 'slim down', 'cutting'],
            'muscle gain': ['gain muscle', 'muscle gain', 'bulking', 'build muscle'],
            'energy boost': ['energy', 'boost energy', 'more energy'],
            'heart health': ['heart', 'cardio', 'cholesterol'],
            'digestive health': ['digestion', 'gut health', 'fiber']
        }
        
        for goal, keywords in health_goals.items():
            if any(keyword in input_lower for keyword in keywords):
                preferences['health_goals'].append(goal)
        
        return preferences
    
    def generate_meal_plan_description(self, preferences: Dict[str, Any], daily_calories: float) -> str:
        """Generate a natural language description of the meal plan"""
        prompt = f"""
        Create a brief, friendly description of a personalized meal plan based on these preferences:
        
        Daily Calories: {daily_calories:.0f}
        Diet Type: {preferences.get('diet_type', 'No specific diet')}
        Allergens to Avoid: {', '.join(preferences.get('allergens', []))}
        Meal Type: {preferences.get('meal_type', 'All meals')}
        Cuisine Preference: {preferences.get('cuisine', 'No preference')}
        Health Goals: {', '.join(preferences.get('health_goals', []))}
        
        Make it encouraging and highlight key benefits. Keep it under 100 words.
        """
        
        try:
            response = ollama.generate(model=self.model_name, prompt=prompt)
            return response['response'].strip()
        except:
            return f"Your personalized {daily_calories:.0f} calorie meal plan tailored to your preferences."