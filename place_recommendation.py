import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Gemini API Configuration
genai.configure(api_key="AIzaSyATeuavfgyi58IOrWbYnikFchU4BCoZuhw")

# Generation config for Gemini
generation_config = {
    "temperature": 0.5,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Create the model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# FastAPI app
app = FastAPI(port=8080)
# Request model
class TravelRequest(BaseModel):
    budget: str
    starting_location: str
    group_size: int
    preference_type: str

# Travel recommendation endpoint
@app.post("/recommend_travel")
async def recommend_travel(request: TravelRequest):
    try:
        # Construct prompt for Gemini
        prompt = f"""
        Provide a detailed travel recommendation based on the following details:
        - Budget: {request.budget}
        - Starting Location: {request.starting_location}
        - Group Size: {request.group_size}
        - Preference Type: {request.preference_type}

        Give a comprehensive travel plan including:
        1. Recommended destinations
        2. Estimated costs
        3. Accommodation suggestions
        4. Travel logistics
        5. Best time to visit

        give result in 500 words
        """

        # Start chat session and send message
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(prompt)

        # Return the recommendation
        return {
            "recommendation": response.text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn main:app --reload