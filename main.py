from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import pickle
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware

# Initialize global variables
df = None
rating_matrix = None
recommender = None
hotel = None

# Configure Gemini API
genai.configure(api_key="AIzaSyATeuavfgyi58IOrWbYnikFchU4BCoZuhw")
generation_config = {
    "temperature": 0.5,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# FastAPI application setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Your application code continues here...


@app.on_event("startup")
async def load_data():
    """Load all necessary data at startup."""
    global df, rating_matrix, recommender, hotel
    try:
        with open('hotel/hotel.pkl', 'rb') as f:
            hotel = pickle.load(f)
        print('Hotel data loaded...')

        with open('food/df.pkl', 'rb') as f:
            df = pickle.load(f)
        print('Food data loaded...')
        
        with open('food/rating_matrix.pkl', 'rb') as f:
            rating_matrix = pickle.load(f)
        print('Rating matrix loaded...')
        
        with open('food/recommender.pkl', 'rb') as f:
            recommender = pickle.load(f)
        print('Recommender model loaded...')
    except Exception as e:
        print(f"Error loading pickle files: {e}")
        raise HTTPException(status_code=500, detail="Failed to load data")

# Helper functions for recommendation logic

def Get_Food_Recommendations(title: str):
    """Get food recommendations based on food title."""
    user = df[df['Name'] == title]
    if user.empty:
        raise HTTPException(status_code=404, detail="Food not found")
    user_index = np.where(rating_matrix.index == int(user['Food_ID']))[0][0]
    user_ratings = rating_matrix.iloc[user_index]
    reshaped = user_ratings.values.reshape(1, -1)
    distances, indices = recommender.kneighbors(reshaped, n_neighbors=16)
    nearest_neighbors_indices = rating_matrix.iloc[indices[0]].index[1:]
    nearest_neighbors = pd.DataFrame({'Food_ID': nearest_neighbors_indices})
    result = pd.merge(nearest_neighbors, df, on='Food_ID', how='left')
    return result[['Name']].head().to_dict(orient='records')

def requirementbased(city, number, features):
    """Get hotel recommendations based on requirements."""
    hotel['city'] = hotel['city'].str.lower()
    hotel['roomamenities'] = hotel['roomamenities'].str.lower()
    features = features.lower()
    features_tokens = word_tokenize(features)
    sw = stopwords.words('english')
    lemm = WordNetLemmatizer()
    f1_set = {w for w in features_tokens if w not in sw}
    f_set = set(lemm.lemmatize(se) for se in f1_set)
    reqbased = hotel[hotel['city'] == city.lower()]
    reqbased = reqbased[reqbased['guests_no'] == number]
    reqbased = reqbased.set_index(np.arange(reqbased.shape[0]))
    cos = []
    for i in range(reqbased.shape[0]):
        temp_tokens = word_tokenize(reqbased['roomamenities'][i])
        temp1_set = {w for w in temp_tokens if w not in sw}
        temp_set = set(lemm.lemmatize(se) for se in temp1_set)
        rvector = temp_set.intersection(f_set)
        cos.append(len(rvector))
    reqbased['similarity'] = cos
    reqbased = reqbased.sort_values(by='similarity', ascending=False)
    reqbased.drop_duplicates(subset='hotelcode', keep='first', inplace=True)
    return reqbased[['hotelname', 'roomtype', 'guests_no', 'starrating', 'address', 'roomamenities', 'ratedescription']].head(5)

def clean_json_response(response):
    """Clean JSON response from Gemini API."""
    cleaned_response = response.replace("json", "").replace("", "").strip()
    return cleaned_response

# FastAPI routes

@app.get("/")
def read_root():
    """Welcome message on root endpoint."""
    return {"message": "Welcome to the Hotel & Food Recommendation API"}

class RecommendationRequest(BaseModel):
    """Request body for food recommendations."""
    title: str

class HotelRecommendationRequest(BaseModel):
    """Request body for hotel recommendations."""
    city: str
    number_of_guests: int
    features: str

@app.post("/food_recommendations/")
async def get_food_recommendations(request: RecommendationRequest):
    """Get food recommendations based on food title."""
    title = request.title
    recommendations = Get_Food_Recommendations(title)
    return {"recommendations": recommendations}

@app.post("/hotel_recommendations/")
async def get_hotel_recommendations(request: HotelRecommendationRequest):
    """Get hotel recommendations based on city, number of guests, and features."""
    city = request.city
    number_of_guests = request.number_of_guests
    features = request.features
    recommendations = requirementbased(city, number_of_guests, features)
    return {"recommendations": recommendations.to_dict(orient="records")}

class TravelRequest(BaseModel):
    """Request body for travel recommendations."""
    budget: str
    starting_location: str
    group_size: int
    preference_type: str

@app.post("/recommend_travel")
async def recommend_travel(request: TravelRequest):
    """Generate travel recommendations using Gemini API."""
    try:
        prompt = f"""
        Provide a travel recommendation in JSON format based on the following details:
        - Budget: {request.budget}
        - Starting Location: {request.starting_location}
        - Group Size: {request.group_size}
        - Preference Type: {request.preference_type}

        Return the response in the following JSON structure:
        {{
            "recommended_destinations": {{
                "name": "string",
                "description": "string",
                "highlights": ["string"]
            }},
            "estimated_costs": {{
                "flights": int,
                "accommodation": int,
                "daily_expenses": int,
                "total_cost": int,
                "currency": "INR"
            }},
            "accommodation": [
                {{
                    "type": "string",
                    "price_range": "₹5000-₹8000",
                    "suggested_options": ["string"]
                }}
            ],
            "travel_logistics": {{
                "recommended_transport": "string",
                "travel_duration": "string",
                "visa_requirements": "string",
                "local_transportation": "string"
            }},
            "best_time_to_visit": {{
                "recommended_months": ["string"],
                "weather": "string",
                "peak_season": "string",
                "off_peak_season": "string"
            }}
        }}
        """
        
        # Start chat session
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(prompt)  # No await since it's not async
        rr = clean_json_response(response.text)
        
        return {"recommendation": rr}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
