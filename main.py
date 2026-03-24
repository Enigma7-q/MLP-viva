import streamlit as st
import pandas as pd
import re
import json
import os
from openai import OpenAI

# --- Configuration ---
# Get your free key from https://openrouter.ai/keys
OPENROUTER_API_KEY = OPENROUTER_API_KEY
CACHE_FILE = "proctor_summaries.json"

# Initialize the OpenRouter client using the OpenAI SDK
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

def standardize_id(raw_id):
    """Cleans messy IDs (e.g., mlp_level1_viva16 -> L1_16)"""
    if not isinstance(raw_id, str):
        return ""
    
    raw_id = raw_id.lower()
    level_match = re.search(r'(?:l|level)[_\s]*(\d+)', raw_id)
    proctor_match = re.search(r'(\d+)$', raw_id)
    
    if level_match and proctor_match:
        return f"L{level_match.group(1)}_{proctor_match.group(1)}"
    return raw_id 

@st.cache_data(ttl=600) 
def load_data(sheet_url):
    """Loads the public Google Sheet directly into a Pandas DataFrame"""
    csv_url = "./data.csv"  # For local testing, replace with the actual CSV URL if needed
    df = pd.read_csv(csv_url)
    
    col_name = "Viva Proctor ID (the id which was sent to you in email for viva)"
    df['Standard_ID'] = df[col_name].apply(standardize_id)
    return df

# --- Caching Functions ---
def load_summary_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_summary_cache(cache_data):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache_data, f, indent=4)

def generate_summary(reviews):
    """Sends the bundled reviews to OpenRouter for a structured summary"""
    
    system_prompt = """
    You are a helpful assistant helping college students prepare for their viva exams. 
    Read the following student reviews about a specific proctor. 
    Generate a clear, encouraging summary with these three distinct sections: 
    1. **Proctor Personality/Vibe** 2. **Most Common Technical Questions** 3. **Top Tips for Success**
    
    If the reviews say no technical questions were asked, state that clearly.
    our target is to give to the point concise yet complete answers to help students prepare for their viva. Avoid unnecessary fluff and keep the summary focused on actionable insights.
    """
    
    # We are using a free model on OpenRouter. 
    # You can explore others at openrouter.ai/models?max_price=0
    response = client.chat.completions.create(
        model="nvidia/nemotron-3-super-120b-a12b:free",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Here are the raw reviews:\n{reviews}"}
        ],
        # Optional: Adding your site URL/Name for OpenRouter rankings
        extra_headers={
            "HTTP-Referer": "https://vivaprep.streamlit.app", 
            "X-Title": "Viva Proctor Prep",
        }
    )
    
    return response.choices[0].message.content

# --- App UI ---
st.set_page_config(page_title="Viva Proctor Prep", page_icon="🎓")
st.title("🎓 Viva Proctor Prep")
st.write("Search your assigned proctor ID to see what they usually ask and how to prepare.")

# --- Setup Requirements ---
SHEET_LINK = "https://docs.google.com/spreadsheets/d/1d5eKfnT9jX3ktKMBYjs3p4rC8jZfeBaxMr7U8E5GIJU/edit" 

try:
    df = load_data(SHEET_LINK)
except Exception as e:
    st.error("Waiting for a valid Google Sheet link to be added to the code...")
    st.stop()

summary_cache = load_summary_cache()

# --- Search Section ---
user_search = st.text_input("Enter your Proctor ID (e.g., level1_viva16, L1_16):")

if st.button("Get Proctor Summary"):
    if user_search:
        search_id = standardize_id(user_search)
        filtered_df = df[df['Standard_ID'] == search_id]
        current_review_count = len(filtered_df)
        
        if filtered_df.empty:
            st.warning(f"No reviews found for proctor **{search_id}**. You're flying blind! Be sure to add your experience to the sheet after your viva.")
        else:
            st.success(f"Found {current_review_count} review(s) for **{search_id}**!")
            
            # --- The Smart Cache Logic ---
            if search_id in summary_cache and summary_cache[search_id].get("review_count") == current_review_count:
                st.info("⚡ Loaded from saved summaries (No API call used!)")
                st.markdown(summary_cache[search_id]["summary"])
            
            else:
                reviews_text = ""
                for index, row in filtered_df.iterrows():
                    questions = row.get("Questions asked by the Viva examiner (Write in points)", "None logged.")
                    tips = row.get("Any suggestions/tips to fellow students for this examiner?", "None logged.")
                    reviews_text += f"\nReview {index + 1}:\nQuestions: {questions}\nTips: {tips}\n---"
                
                with st.spinner("AI is reading the reviews and writing a fresh summary via OpenRouter..."):
                    try:
                        if OPENROUTER_API_KEY == "PASTE_YOUR_OPENROUTER_API_KEY_HERE":
                            st.error("⚠️ You need to paste your OpenRouter API key in the code to see the AI summary.")
                        else:
                            summary = generate_summary(reviews_text)
                            st.markdown(summary)
                            
                            summary_cache[search_id] = {
                                "review_count": current_review_count,
                                "summary": summary
                            }
                            save_summary_cache(summary_cache)
                            
                    except Exception as e:
                        st.error(f"An error occurred while generating the summary. Make sure you are using a valid free model on OpenRouter! Error details: {e}")
