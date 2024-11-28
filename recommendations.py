from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import pickle
from fastapi.middleware.cors import CORSMiddleware

df = None
rating_matrix = None
recommender = None
hotel = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "http://127.0.0.1:3001"],  # Include both localhost and 127.0.0.1
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def load_data():
    global df, rating_matrix, recommender, hotel
    
    try:
        with open('hotel/hotel.pkl', 'rb') as f:
            hotel = pickle.load(f)
        print('Hotel data loaded...')

        with open('food/df.pkl', 'rb') as f:
            df = pickle.load(f)
        print('df loaded...')
        
        with open('food/rating_matrix.pkl', 'rb') as f:
            rating_matrix = pickle.load(f)
        print('rating_matrix loaded...')
        
        with open('food/recommender.pkl', 'rb') as f:
            recommender = pickle.load(f)
        print('recommender loaded...')
    except Exception as e:
        print(f"Error loading pickle files: {e}")
        raise HTTPException(status_code=500, detail="Failed to load data")

class RecommendationRequest(BaseModel):
    title: str

class HotelRecommendationRequest(BaseModel):
    city: str
    number_of_guests: int
    features: str

def Get_Food_Recommendations(title: str):
    # Filter user based on title
    user = df[df['Name'] == title]
    
    if user.empty:
        raise HTTPException(status_code=404, detail="Food not found")
    
    user_index = np.where(rating_matrix.index == int(user['Food_ID']))[0][0]
    user_ratings = rating_matrix.iloc[user_index]

    reshaped = user_ratings.values.reshape(1, -1)
    distances, indices = recommender.kneighbors(reshaped, n_neighbors=16)
    
    # Get nearest neighbor food indices (ignoring the first one as it's the user themselves)
    nearest_neighbors_indices = rating_matrix.iloc[indices[0]].index[1:]
    nearest_neighbors = pd.DataFrame({'Food_ID': nearest_neighbors_indices})
    
    # Merge with the original dataframe to get the food details
    result = pd.merge(nearest_neighbors, df, on='Food_ID', how='left')
    
    return result[['Name']].head().to_dict(orient='records')

def requirementbased(city, number, features):
    # Ensure required libraries are loaded and available
    hotel['city'] = hotel['city'].str.lower()
    hotel['roomamenities'] = hotel['roomamenities'].str.lower()
    features = features.lower()
    
    # Tokenize the input features
    features_tokens = word_tokenize(features)
    sw = stopwords.words('english')
    lemm = WordNetLemmatizer()

    # Remove stopwords and apply lemmatization
    f1_set = {w for w in features_tokens if w not in sw}
    f_set = set(lemm.lemmatize(se) for se in f1_set)
    
    # Filter hotels by city and guest number
    reqbased = hotel[hotel['city'] == city.lower()]
    reqbased = reqbased[reqbased['guests_no'] == number]
    reqbased = reqbased.set_index(np.arange(reqbased.shape[0]))
    
    # Compute similarity based on room amenities
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

@app.post("/food_recommendations/")
async def get_recommendations(request: RecommendationRequest):
    title = request.title
    recommendations = Get_Food_Recommendations(title)
    return {"recommendations": recommendations}

@app.post("/hotel_recommendations/")
async def get_hotel_recommendations(request: HotelRecommendationRequest):
    city = request.city
    number_of_guests = request.number_of_guests
    features = request.features
    
    recommendations = requirementbased(city, number_of_guests, features)
    
    return {"recommendations": recommendations.to_dict(orient="records")}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host="127.0.0.1" , port=8001)
