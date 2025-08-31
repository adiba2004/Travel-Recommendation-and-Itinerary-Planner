import streamlit as st
import pyrebase
from datetime import datetime
from streamlit_option_menu import option_menu
import time
import traceback
from io import BytesIO
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import pickle
from dotenv import load_dotenv
import os
from google import generativeai as genai
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.units import inch
from reportlab.lib import colors
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import traceback  
import re

###########################################################################################
# Add this CSS at the beginning of your script
st.markdown("""
<style>
    /* Main section headers */
    .main-header {
        color: #2c3e50;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #3498db;
    }
    
    /* Day headers */
    .day-header {
        color: #3498db;
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 1.8rem;
        margin-bottom: 0.8rem;
    }
    
    /* Time blocks */
    .time-block {
        color: #e67e22;
        font-size: 1.1rem;
        font-weight: 500;
        margin-top: 1.2rem;
        margin-bottom: 0.5rem;
        margin-left: 1rem;
    }
    
    /* Activity items */
    .activity-item {
        margin-left: 2.5rem;
        margin-bottom: 0.7rem;
        line-height: 1.6;
        position: relative;
    }
    
    /* Bullet points */
    .activity-item:before {
        content: "•";
        color: #3498db;
        font-weight: bold;
        position: absolute;
        left: -1rem;
    }
    
    /* Price highlighting */
    .price {
        color: #27ae60;
        font-weight: 600;
        background-color: #f8f9fa;
        padding: 0.1rem 0.3rem;
        border-radius: 3px;
    }
    
    /* Special sections */
    .special-section {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px dashed #bdc3c7;
    }
    
    /* Hidden gems header */
    .gems-header {
        color: #9b59b6;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

###########################################################################################
# Configure the API key for Itinerary planner
os.environ["GOOGLE_API_KEY"] = "YOUR-API-KEY"
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

###########################################################################################
# Initialize all session state variables
DEFAULTS = {
    'location': "",
    'budget': "Mid-range",  # Changed from "Medium" to match selectbox options
    'travel_dates': "",
    'travel_party': "Solo",
    'activities_interests': [],
    'accommodation_type': "Hotels",
    'proximity_to_attractions': [],
    'mode_of_transport': "Car",
    'rental_services': [],
    'public_transport_preferences': [],
    'dietary_restrictions': [],
    'interest_in_local_cuisine': [],
    'dining_experience': "Casual",
    'ambiance_preferences': "Any",
    'cuisine_variety': [],
    'itinerary_data': None,
    'pdf_buffer': None
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

###########################################################################################
# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user' not in st.session_state:
    st.session_state.user = None
if 'firebase' not in st.session_state:
    st.session_state.firebase = None
if 'db' not in st.session_state:
    st.session_state.db = None

# Initialize session state for recommendation system
if 'recommendation_type' not in st.session_state:
    st.session_state.recommendation_type = None  # 'new_user' or 'returning_user'

###########################################################################################
# Firebase Configuration
firebaseConfig = {
    'apiKey': "YOUR-API-KEY",
    'authDomain': "travel-recommendation-sy-f29e1.firebaseapp.com",
    'projectId': "travel-recommendation-sy-f29e1",
    'databaseURL': "https://travel-recommendation-sy-f29e1-default-rtdb.europe-west1.firebasedatabase.app/",
    'storageBucket': "travel-recommendation-sy-f29e1.appspot.com",
    'messagingSenderId': "986432290513",
    'appId': "1:986432290513:web:2969df8f4719e9b7c64ff0",
    'measurementId': "G-KVHN8JW254"
}

###########################################################################################
# Initialize Firebase
if not st.session_state.firebase:
    st.session_state.firebase = pyrebase.initialize_app(firebaseConfig)
    st.session_state.auth = st.session_state.firebase.auth()
    st.session_state.db = st.session_state.firebase.database()
    st.session_state.storage = st.session_state.firebase.storage()

# Initialize session state for Itinerary
# if 'itinerary_data' not in st.session_state:
    # st.session_state.itinerary_data = None
# if 'pdf_buffer' not in st.session_state:
    # st.session_state.pdf_buffer = None

###########################################################################################
# Load data
@st.cache_data
def load_data():
    travel = pickle.load(open('artifacts/place_list.pkl', 'rb'))
    similarity = pickle.load(open('artifacts/similarity.pkl', 'rb'))
    links_data = pd.read_excel("Data/Links.xlsx")
    return travel, similarity, links_data

travel, content_sim, links_data = load_data()

###########################################################################################
# Recommendation Engine Class
class RecommendationEngine:
    def __init__(self, db):
        self.db = db
        self.user_history = None
        self.model = None
        self.city_pivot = None
        self.mae = None  # Track MAE score

    def evaluate_model(self, ratings_df):
        """Evaluate model using train-test split and MAE"""
        try:
            # Split data (80% train, 20% test)
            train_data, test_data = train_test_split(ratings_df, test_size=0.2, random_state=42)
            
            # Create pivot tables with consistent columns
            all_users = sorted(set(ratings_df['User'].unique()))
            train_pivot = train_data.pivot_table(
                index='City',
                columns='User',
                values='Rating',
                fill_value=0
            ).reindex(columns=all_users, fill_value=0)
            
            test_pivot = test_data.pivot_table(
                index='City',
                columns='User',
                values='Rating',
                fill_value=0
            ).reindex(columns=all_users, fill_value=0)
            
            # Ensure same number of features
            missing_cols = set(train_pivot.columns) - set(test_pivot.columns)
            for col in missing_cols:
                test_pivot[col] = 0
            test_pivot = test_pivot[train_pivot.columns]
            
            # Train model
            self.model = NearestNeighbors(metric='cosine', algorithm='brute')
            self.model.fit(csr_matrix(train_pivot))
            
            # Predict and calculate MAE
            distances, indices = self.model.kneighbors(csr_matrix(test_pivot), n_neighbors=3)
            
            # Get average ratings from nearest neighbors
            predicted = []
            for i in range(len(test_pivot)):
                neighbor_ratings = train_pivot.iloc[indices[i]].mean(axis=1)
                predicted.append(neighbor_ratings.mean())
                
            true_ratings = test_data.groupby('City')['Rating'].mean().values
            self.mae = mean_absolute_error(true_ratings[:len(predicted)], predicted)
            
            return self.mae
            
        except Exception as e:
            print(f"Evaluation failed: {str(e)}")
            traceback.print_exc()  # Add this for detailed error logging
            return None
        
    def load_user_history(self, user_id):
        """Load existing travel history from Firebase"""
        self.user_history = self.db.child("travel_history").child(user_id).get().val()
        return self.user_history
        
    def train_collaborative_model(self):
        """Train model using existing user data"""
        all_ratings = self._get_all_ratings()
        
        if len(all_ratings) < 5:
            print("Insufficient data for evaluation (need ≥5 ratings)")
            return

        if len(all_ratings) >= 5:  # Minimum threshold for evaluation
            # Perform evaluation
            mae = self.evaluate_model(all_ratings)
            if mae is not None:
                print(f"Model trained with MAE: {mae:.4f}")
            
            self.city_pivot = all_ratings.pivot_table(
                index='City',
                columns='User',
                values='Rating',
                fill_value=0
            )
            self.model = NearestNeighbors(metric='cosine', algorithm='brute').fit(
                csr_matrix(self.city_pivot)
            )
            
    def _get_all_ratings(self):
        """Combine all users' ratings from Firebase"""
        all_data = []
        histories = self.db.child("travel_history").get().val() or {}
        
        for user_id, experiences in histories.items():
            for exp_id, exp in experiences.items():
                all_data.append({
                    'User': user_id,
                    'City': exp['city'],
                    'Rating': exp['rating']
                })
        return pd.DataFrame(all_data)

    def get_recommendations(self, user_id, current_city):
        """Smart recommendation router"""
        if not self.load_user_history(user_id):
            # Fallback to content-based if no history exists
            return self._content_based_recommend([current_city])
            
        if self.model:
            try:
                city_idx = self.city_pivot.index.get_loc(current_city)
                _, indices = self.model.kneighbors(
                    self.city_pivot.iloc[city_idx].values.reshape(1, -1),
                    n_neighbors=4
                )
                return [self.city_pivot.index[i] for i in indices.flatten()[1:]]
            except:
                return self._content_based_recommend([current_city])
        return self._content_based_recommend([current_city])
    
    def _content_based_recommend(self, selected_cities):
        """Content-based recommendation logic"""
        if not selected_cities:
            return []
        
        sim_scores = np.zeros(len(travel))
        for city in selected_cities:
            idx = travel[travel['city'] == city].index[0]
            sim_scores += content_sim[idx]
        sim_scores /= len(selected_cities)
        
        # Get top recommendations excluding selected cities
        rec_indices = np.argsort(sim_scores)[::-1]
        recommendations = []
        for i in rec_indices:
            city_name = travel.iloc[i].city
            if city_name not in selected_cities:  # Exclude already selected cities
                recommendations.append(city_name)
                if len(recommendations) >= 5:
                    break          
        return recommendations

###########################################################################################
# Admin or not functions
def is_admin():
    """Check if current user is an admin"""
    if not st.session_state.logged_in or not st.session_state.user:
        return False
    user_data = st.session_state.db.child("users").child(st.session_state.user['localId']).get().val()
    return user_data and user_data.get("isAdmin", False)

###########################################################################################
# To display the recommendated cities
def get_city_url(city_name):
    match = links_data[links_data['City'] == city_name]
    return match.iloc[0]['URL'] if not match.empty else "#"
    
def get_city_info(city_name):
    match = links_data[links_data['City'] == city_name]
    if not match.empty:
        return {
            'Country': match.iloc[0].get('Country', 'N/A'),
            'Population': match.iloc[0].get('Population', 'N/A'),
            'Area (sq mi)': match.iloc[0].get('Area (sq mi)', 'N/A')
        }
    return None

def display_recommendations(cities):
    """Display recommendations in 5-column grid with tooltips"""
    st.write("Your Recommendations are: ")

    if not cities:
        st.write("No recommendations available.")
        return

    # Display the recommendations in rows of 5
    num_columns = 5
    cols = st.columns(num_columns)

    for i, city in enumerate(cities):
        col_index = i % num_columns
        with cols[col_index]:
            # Get city info
            url = get_city_url(city)
            city_info = get_city_info(city)
            # Prepare tooltip
            if city_info:
                tooltip_text = (
                    f"Country: {city_info.get('Country', 'N/A')}<br>"
                    f"Population: {city_info.get('Population', 'N/A')}<br>"
                    f"Area: {city_info.get('Area (sq mi)', 'N/A')} sq mi"
                )
            else:
                tooltip_text = "No info available"

            # Create HTML for clickable city with tooltip
            link_html = f'''
            <div class="tooltip">
                <a href="{url}" style="font-weight: bold; text-decoration: none; color: black;">{city}</a>
                <span class="tooltiptext">{tooltip_text}</span>
            </div>
            '''
            st.markdown(
                f'<div style="text-align: center; margin: 10px;">{link_html}</div>', 
                unsafe_allow_html=True
            )
        # Create new row after 5 cities
        if col_index == num_columns - 1 and i < len(cities) - 1:
            cols = st.columns(num_columns)

    # Add CSS for tooltips
    st.markdown("""
    <style>
    .tooltip {
        position: relative;
        display: inline-block;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
    """, unsafe_allow_html=True)

###########################################################################################
# Itinerary Planner
def get_currency_symbol(location):
    """Return appropriate currency symbol based on location"""
    india_keywords = ['india', 'delhi', 'mumbai', 'bangalore', 'chennai', 
                     'hyderabad', 'kolkata', 'pune', 'ahmedabad']
    
    if any(keyword in location.lower() for keyword in india_keywords):
        return 'Rs.'  # Use "Rs." instead of "₹" to avoid PDF issues
    elif 'usa' in location.lower() or 'united states' in location.lower():
        return '$'
    elif 'euro' in location.lower() or any(keyword in location.lower() 
          for keyword in ['france', 'germany', 'spain', 'italy', 'paris', 'berlin']):
        return '€'
    elif 'uk' in location.lower() or 'united kingdom' in location.lower():
        return '£'
    elif 'japan' in location.lower() or 'tokyo' in location.lower():
        return '¥'
    elif 'australia' in location.lower() or 'sydney' in location.lower():
        return 'A$'
    else:
        return '$'  # Default to dollar

def clean_markdown(text, currency_symbol='Rs.'):
    """Clean markdown with dynamic currency"""
    if not isinstance(text, str):
        return text
    
    # Replace all variations of Indian Rupee with "Rs."
    text = re.sub(r'[₹]', 'Rs.', text)  # Replace ₹ with Rs.
    text = re.sub(r'(?i)\b(rs|inr)\b\.?\s*(\d+)', r'Rs.\2', text)  # Replace "Rs 1000" with "Rs.1000"
    
    # Replace other currency symbols (unchanged)
    text = re.sub(r'(?i)\b(usd)\b\.?\s*(\d+)', r'$\2', text)
    text = re.sub(r'(?i)\b(eur)\b\.?\s*(\d+)', r'€\2', text)
    text = re.sub(r'(?i)\b(gbp)\b\.?\s*(\d+)', r'£\2', text)
    
    # Clean other markdown
    text = text.replace('## ', '').replace('# ', '')
    text = text.replace('**', '').replace('* ', '• ')
    text = re.sub(r'\b(\d+)\s*m\b', f'{currency_symbol}\\1', text)  # Fix "1500m" type errors
    
    return text

def get_response(prompt, input):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-001')
        response = model.generate_content([prompt, input], stream=False)
        return response.text
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return None

def create_pdf(itinerary_data):
    buffer = BytesIO()
    currency = itinerary_data.get('currency', '₹')
    
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=20*mm,
        rightMargin=20*mm,
        topMargin=15*mm,
        bottomMargin=15*mm
    )
    
    styles = getSampleStyleSheet()
    
    # Modify existing styles
    styles['BodyText'].fontSize = 10
    styles['BodyText'].leading = 13
    styles['BodyText'].spaceAfter = 6
    styles['BodyText'].wordWrap = 'LTR'
    
    # Add bullet style if it doesn't exist
    if not hasattr(styles, 'Bullet'):
        styles.add(ParagraphStyle(
            name='Bullet',
            fontSize=10,
            leading=13,
            leftIndent=10,
            bulletIndent=5,
            spaceAfter=4,
            wordWrap='LTR'
        ))
    
    story = []
    
    # Title
    story.append(Paragraph(
        f"<b>{itinerary_data['location'].upper()} TRAVEL ITINERARY</b>",
        ParagraphStyle(
            name='Title',
            fontSize=16,
            alignment=1,
            spaceAfter=12,
            fontName='Helvetica-Bold',
            textColor=colors.HexColor('#2E86C1')
        )
    ))
    
    # Add date
    story.append(Paragraph(
        f"Generated on: {datetime.now().strftime('%B %d, %Y')}",
        ParagraphStyle(
            name='Date',
            fontSize=8,
            alignment=1,
            spaceAfter=20,
            textColor=colors.grey
        )
    ))
    
    # Trip Details
    story.append(Paragraph(
        "<b><u>Trip Summary</u></b>",
        ParagraphStyle(
            name='Heading1',
            fontSize=12,
            spaceBefore=12,
            spaceAfter=8,
            fontName='Helvetica-Bold'
        )
    ))
    
    details = [
        f"<b>Location:</b> {itinerary_data['location']}",
        f"<b>Budget:</b> {itinerary_data['budget']}",
        f"<b>Duration:</b> {itinerary_data['travel_dates']}",
        f"<b>Travel Party:</b> {itinerary_data['travel_party']}",
        f"<b>Activities:</b> {', '.join(itinerary_data['activities_interests'])}"
    ]
    
    for detail in details:
        story.append(Paragraph(detail, styles['BodyText']))
        story.append(Spacer(1, 4))
    
    story.append(Spacer(1, 15))
    
    # Process each section
    sections = {
        'trip_plan': 'Daily Itinerary',
        'accommodation': 'Recommended Stays',
        'transport': 'Transport Options',
        'food': 'Dining Guide',
        'budget_estimate': 'Budget Breakdown'
    }
    
    for section, title in sections.items():
        if section in itinerary_data and itinerary_data[section]:
            # Add section header
            story.append(Paragraph(
                f"<b><u>{title}</u></b>",
                ParagraphStyle(
                    name='SectionHeader',
                    fontSize=12,
                    spaceBefore=15,
                    spaceAfter=8,
                    fontName='Helvetica-Bold'
                )
            ))
            
            content = clean_markdown(itinerary_data[section], currency)
            paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
            
            for para in paragraphs:
                if para.startswith('•'):
                    story.append(Paragraph(para, styles['Bullet']))
                else:
                    story.append(Paragraph(para, styles['BodyText']))
                story.append(Spacer(1, 4))
            
            story.append(Spacer(1, 12))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def generate_itinerary():
    with st.spinner('Creating your perfect itinerary...'):
        location = st.session_state.location
        currency = get_currency_symbol(location)
        
        itinerary_data = {
            'location': location,
            'currency': currency,
            'budget': st.session_state.budget,
            'travel_dates': st.session_state.travel_dates,
            'travel_party': st.session_state.travel_party,
            'activities_interests': st.session_state.activities_interests,
            'accommodation_type': st.session_state.accommodation_type,
            'proximity_to_attractions': st.session_state.proximity_to_attractions,
            'mode_of_transport': st.session_state.mode_of_transport,
            'rental_services': st.session_state.rental_services,
            'public_transport_preferences': st.session_state.public_transport_preferences,
            'dietary_restrictions': st.session_state.dietary_restrictions,
            'interest_in_local_cuisine': st.session_state.interest_in_local_cuisine,
            'dining_experience': st.session_state.dining_experience,
            'ambiance_preferences': st.session_state.ambiance_preferences,
            'cuisine_variety': st.session_state.cuisine_variety
        }
        
        # Generate all sections with currency-aware prompts
        prompts = {
            'trip_plan': (f"""
                Create a detailed travel itinerary with:
                - Day-by-day activities with times
                - Must-see attractions with ENTRY FEES IN {currency}
                - Hidden gems
                - Safety tips
                Format with clear bullet points
                ALWAYS use {currency} for currency
                Don't give any info in tables
                """, f"""
                Location: {itinerary_data['location']}
                Budget: {itinerary_data['budget']}
                Duration: {itinerary_data['travel_dates']}
                Group: {itinerary_data['travel_party']}
                Interests: {', '.join(itinerary_data['activities_interests'])}
                """),
                
            'accommodation': (f"""
                Recommend accommodations with:
                - Top 5 options with ratings
                - Price ranges in {currency}
                - Key amenities
                - Proximity to attractions
                Present in simple format
                Use ONLY {currency} for prices
                Don't give any info in tables
                """, f"""
                Location: {itinerary_data['location']}
                Budget: {itinerary_data['budget']}
                Type: {itinerary_data['accommodation_type']}
                Proximity: {', '.join(itinerary_data['proximity_to_attractions'])}
                Group: {itinerary_data['travel_party']}
                """),
                
            'transport': (f"""
                Provide transport options including:
                - Best transportation methods
                - Rental options with prices in {currency}
                - Public transit tips
                - Estimated costs in {currency}
                Format with clear sections
                Use ONLY {currency} for prices
                Don't give any info in tables
                """, f"""
                Location: {itinerary_data['location']}
                Mode: {itinerary_data['mode_of_transport']}
                Rentals: {', '.join(itinerary_data['rental_services'])}
                Public Transport: {', '.join(itinerary_data['public_transport_preferences'])}
                """),
                
            'food': (f"""
                Suggest dining options with:
                - Recommended restaurants with price ranges in {currency}
                - Local specialties
                - Dietary accommodations
                Group by meal type
                Use ONLY {currency} for prices
                Don't give any info in tables
                """, f"""
                Location: {itinerary_data['location']}
                Diets: {', '.join(itinerary_data['dietary_restrictions'])}
                Cuisine Interests: {', '.join(itinerary_data['interest_in_local_cuisine'])}
                Experience: {itinerary_data['dining_experience']}
                Ambiance: {itinerary_data['ambiance_preferences']}
                Cuisines: {', '.join(itinerary_data['cuisine_variety'])}
                """),
                
            'budget_estimate': (f"""
                Create detailed budget estimate:
                - Breakdown by category in {currency}
                - Daily costs in {currency}
                - Total estimate in {currency}
                - Money-saving tips
                Use ONLY {currency} for all amounts
                Don't give any info in tables
                """, f"""
                Location: {itinerary_data['location']}
                Duration: {itinerary_data['travel_dates']}
                Budget Level: {itinerary_data['budget']}
                Accommodation: {itinerary_data['accommodation_type']}
                Transport: {itinerary_data['mode_of_transport']}
                Food: {itinerary_data['dining_experience']}
                Activities: {', '.join(itinerary_data['activities_interests'])}
                """)
        }
        
        for section, (prompt_template, input_text) in prompts.items():
            response = get_response(prompt_template, input_text)
            itinerary_data[section] = clean_markdown(response or "No information available", currency)
        
        st.session_state.itinerary_data = itinerary_data
        st.session_state.pdf_buffer = create_pdf(itinerary_data)

def format_itinerary(content):
    """Format itinerary content with proper spacing and styling"""
    paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
    formatted = []
    
    for para in paragraphs:
        # Handle day headers
        if "Day" in para and ":" in para:
            formatted.append(f'<div class="day-header">{para}</div>')
        
        # Handle time blocks
        elif any(x in para for x in ["Morning", "Breakfast","Lunch", "Afternoon", "Evening", "Dinner", "Night"]) and ":" in para:
            formatted.append(f'<div class="time-block">{para}</div>')
        
        # Handle activity items
        elif para.startswith('•'):
            # Highlight prices
            if "Entry:" in para or "approx." in para:
                para = re.sub(r'(Entry:|approx\.)\s*(₹?\d[\d,]*)', 
                             r'\1 <span class="price">\2</span>', para)
            formatted.append(f'<div class="activity-item">{para[1:]}</div>')
        
        # Handle special sections
        elif "Hidden Gems:" in para:
            formatted.append('<div class="special-section">')
            formatted.append(f'<div class="gems-header">{para}</div>')
        elif "Safety Tips:" in para:
            formatted.append(f'<div class="gems-header">{para}</div>')
        
        # Handle regular paragraphs
        else:
            formatted.append(f'<p>{para}</p>')
    
    return ''.join(formatted)

###########################################################################################
# Main Application
if not st.session_state.logged_in:
    # LOGIN/SIGNUP PAGE
    selected = option_menu(
        menu_title=None,
        options=["Login", "Sign Up"],
        icons=["box-arrow-in-left", "box-arrow-in-right"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )

    st.title("🌍 Travel Recommendation System")

    # Login Page
    if selected == "Login":
        with st.form("login_form"):
            st.subheader("Login")
            email = st.text_input('Email address', placeholder='Enter your email')
            password = st.text_input('Password', type='password', placeholder='Enter your password')
            submit_button = st.form_submit_button('Login')
        
            if submit_button:
                if email and password:
                    try:
                        user = st.session_state.auth.sign_in_with_email_and_password(email, password)
                        user_data = st.session_state.db.child("users").child(user['localId']).get().val()
                        
                        if not user_data:
                            st.error("User data not found. Please sign up.")
                            st.stop()
                            
                        st.session_state.update({
                            'logged_in': True,
                            'user': {
                                'email': email,
                                'username': user_data['username'],
                                'localId': user['localId'],
                                'isAdmin': user_data.get('isAdmin', False)
                            }
                        })
                        st.success('Login successful!')
                        time.sleep(1)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f'Login failed: {str(e)}')
                        st.code(f"Error details:\n{traceback.format_exc()}", language='bash')
                else:
                    st.warning('Please fill in both fields.')

    # Sign Up Page
    elif selected == "Sign Up":
        with st.form("signup_form"):
            st.subheader('Sign Up')
            email = st.text_input('Email', placeholder='Enter Your Email')
            username = st.text_input('Username', placeholder='Enter Your Username')
            password = st.text_input('Password', placeholder='Enter Your Password', type='password')
            c_password = st.text_input('Confirm Password', placeholder='Confirm Your Password', type='password')
            submit_button = st.form_submit_button('Create my account')

            if submit_button:
                # Validation checks
                if not all([email, username, password, c_password]):
                    st.error("All fields are required!")
                elif '@' not in email or '.' not in email.split('@')[-1]:
                    st.error("Please enter a valid email address")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters")
                elif password != c_password:
                    st.error("Passwords don't match!")
                else:
                    try:
                        user = st.session_state.auth.create_user_with_email_and_password(email, password)
                        # Store in new structure
                        st.session_state.db.child("users").child(user['localId']).set({
                            "username": username,
                            "email": email,
                            "isAdmin": False
                        })
                        # Update session
                        st.session_state.update({
                            'logged_in': True,
                            'user': {
                                'email': email,
                                'username': username,
                                'localId': user['localId'],
                                'isAdmin': False
                            }
                        })
                        st.success('Account created successfully!')
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        error_msg = str(e)
                        if "EMAIL_EXISTS" in error_msg:
                            st.error("Email already registered")
                        elif "WEAK_PASSWORD" in error_msg:
                            st.error("Password should be at least 6 characters")
                        else:
                            st.error(f'Account creation failed: {error_msg}')

else:
    # MAIN APP PAGE
    # st.set_page_config(layout="wide")
    engine = RecommendationEngine(st.session_state.db)

    # Navigation Bar
    with st.sidebar:
        # Base menu options
        menu_options = ["Home", "Travel Recommendation", "Itinerary Planner", "Travel History"]
        icons = ["house-fill", "airplane-fill", "card-checklist", "geo-alt-fill"]
        
        # Add Admin option if needed
        if is_admin():
            menu_options.insert(-1, "Admin")
            icons.insert(-1, "shield-lock")
        
        # Always add Logout at the end
        menu_options.append("Logout")
        icons.append("box-arrow-right")
        
        selected = option_menu(
            menu_title=None,
            options=menu_options,
            icons=icons,
            menu_icon="cast",
            default_index=0,
        )

    ###########################################################################################
    # Home Page
    if selected == "Home":
        st.title("🌍 Travel Recommendation System")
        st.markdown("""
            <div style="text-align: justify;">
                Welcome to our Personalized Travel Recommendation System...
            </div>
            """, unsafe_allow_html=True)
        st.markdown("")
        st.markdown(
            '''
            <div style="text-align: justify;">
                Whether you're seeking cultural adventures, relaxation, or thrilling activities, our platform ensures every recommendation aligns perfectly with your unique travel style. 
                With seamless integration of real-time data and user feedback, we bring you the most relevant and insightful suggestions to simplify your travel planning process.
            </div>
            ''',
            unsafe_allow_html=True
        ) 
        st.markdown("")

        # Key Features in Columns
        st.header("Why Choose Us?")
        st.markdown("")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image('Data/Pictures/Tailored Recommendations.jpg', caption="Tailored Recommendations", use_container_width=True)
            st.markdown('''<div style="text-align: center; font-weight: bold;">Personalized Itineraries</div>''', unsafe_allow_html=True)

        with col2:
            st.image('Data/Pictures/Real-Time Updates.jpg', caption="Real-Time Updates", use_container_width=True)
            st.markdown('''<div style="text-align: center; font-weight: bold;">Up-to-date Travel Insights</div>''', unsafe_allow_html=True)

        with col3:
            st.image('Data/Pictures/Effortless Planning.jpg', caption="Effortless Planning", use_container_width=True)
            st.markdown('''<div style="text-align: center; font-weight: bold;">Simplified Travel Planning</div>''', unsafe_allow_html=True)

        # Highlighted Call-to-Action
        st.markdown("""
            ---
            **Plan smarter, Travel better, and Explore the world with confidence.**
            """)

    ###########################################################################################
    # Travel Recommendation
    elif selected == "Travel Recommendation":
        # Initialize the recommendation engine
        engine = RecommendationEngine(st.session_state.db)
        
        # Determine user type if not already set
        if st.session_state.recommendation_type is None:
            user_history = engine.load_user_history(st.session_state.user['localId'])
            st.session_state.recommendation_type = 'returning_user' if user_history else 'new_user'
        
        st.header("🌍 Recommended Cities For You")
        
        if st.session_state.recommendation_type == 'new_user':
            st.subheader("Tell us what you like to get started")
            selected_cities = st.multiselect(
                "Select cities you think you'll enjoy:", 
                sorted(travel['city'].unique())
            )
            
            if st.button("Get Recommendations"):
                if selected_cities:
                    recs = engine._content_based_recommend(selected_cities)
                    display_recommendations(recs)
                else:
                    st.warning("Please select at least one city")
        
        else:  # returning user
            # Load user history and train model
            user_history = engine.load_user_history(st.session_state.user['localId'])
            engine.train_collaborative_model()
            
            ### ADD THE MAE METRIC DISPLAY RIGHT HERE (right after training)
            #if engine.mae is not None:
                #st.sidebar.metric("Recommendation Accuracy (MAE)", 
                                #f"{engine.mae:.3f}",
                                #help="Lower values indicate better predictions (0 = perfect)") '''
            
            if user_history:
                with st.spinner("Analyzing your travel history..."):
                    # Define city name normalization mapping
                    city_name_mapping = {
                        # Asia
                        'Singapore': 'Singapore City',
                        'Bombay': 'Mumbai',
                        'Peking': 'Beijing',
                        'Calcutta': 'Kolkata',
                        'Madras': 'Chennai',
                        'Bangalore': 'Bengaluru',
                        'Saigon': 'Ho Chi Minh City',
                        
                        # North America
                        'NYC': 'New York City',
                        'New York': 'New York City',
                        'San Fran': 'San Francisco',
                        'Frisco': 'San Francisco',
                        'LA': 'Los Angeles',
                        'Vegas': 'Las Vegas',
                        'Philly': 'Philadelphia',
                        'DC': 'Washington DC',
                        'Washington': 'Washington DC',
                        'Miami Beach': 'Miami',
                        
                        # Europe
                        'Munich': 'München',
                        'Prague': 'Praha',
                        'Florence': 'Firenze',
                        'Venice': 'Venezia',
                        'Naples': 'Napoli',
                        'Copenhagen': 'København',
                        'The Hague': 'Den Haag',
                        
                        # Middle East
                        'Tel Aviv': 'Tel Aviv-Yafo',
                        'Ben Gurion': 'Tel Aviv-Yafo',
                        
                        # Other common variations
                        'St.': 'Saint',
                        'S.': 'San',
                        'Ft.': 'Fort',
                    }
                    
                    # Process and normalize user's travel history
                    visited_cities = set()
                    valid_rated_cities = []
                    
                    for exp in user_history.values():
                        original_city = exp['city']
                        normalized_city = city_name_mapping.get(original_city, original_city)
                        
                        # Check if city exists in our database
                        if not travel[travel['city'] == normalized_city].empty:
                            visited_cities.add(normalized_city)
                            valid_rated_cities.append((normalized_city, exp['rating']))
                    
                    if not valid_rated_cities:
                        st.warning("None of your visited cities are in our recommendation database")
                    else:
                        # Get top 3 rated valid cities
                        top_cities = sorted(valid_rated_cities, key=lambda x: x[1], reverse=True)[:3]
                        all_recommendations = []
                        
                        # 1. Collaborative filtering recommendations
                        for city, _ in top_cities:
                            try:
                                city_recs = engine.get_recommendations(st.session_state.user['localId'], city)
                                if city_recs:
                                    all_recommendations.extend([
                                        city_name_mapping.get(rec, rec) 
                                        for rec in city_recs 
                                        if city_name_mapping.get(rec, rec) not in visited_cities
                                    ])
                            except Exception as e:
                                st.error(f"Error getting recommendations for {city}: {str(e)}")
                                continue
                        
                        # 2. Content-based recommendations if needed
                        if len(all_recommendations) < 8:
                            try:
                                content_recs = engine._content_based_recommend([city for city, _ in top_cities])
                                if content_recs:
                                    all_recommendations.extend([
                                        city_name_mapping.get(rec, rec) 
                                        for rec in content_recs 
                                        if city_name_mapping.get(rec, rec) not in visited_cities
                                    ])
                            except Exception as e:
                                st.error(f"Content-based recommendation error: {str(e)}")
                        
                        # 3. Add popular cities as fallback
                        if len(all_recommendations) < 8:
                            popular_cities = [
                                'Paris', 'London', 'Tokyo', 'Dubai',
                                'Barcelona', 'Rome', 'Sydney', 'Berlin',
                                'Amsterdam', 'Bangkok', 'Vienna', 'Seoul'
                            ]
                            all_recommendations.extend([
                                city for city in popular_cities
                                if (city_name_mapping.get(city, city) not in visited_cities and
                                    city in travel['city'].values)
                            ][:8-len(all_recommendations)])
                        
                        # Final deduplication
                        final_recommendations = []
                        seen_cities = set()
                        
                        for city in all_recommendations:
                            normalized = city_name_mapping.get(city, city)
                            if normalized not in seen_cities:
                                final_recommendations.append(city)
                                seen_cities.add(normalized)
                        
                        # Display results
                        if final_recommendations:
                            st.success("Based on your travel history, we recommend:")
                            display_recommendations(final_recommendations[:15])
                        else:
                            st.info("Couldn't generate recommendations. Please try adding more cities.")
            else:
                st.info("Add some travel experiences first")
                st.session_state.recommendation_type = 'new_user'

    ###########################################################################################
    # Itinerary Planner
    elif selected == "Itinerary Planner":
        st.title("🌍 Smart Itinerary Planner")
        st.subheader("Plan your perfect trip anywhere in the world!")

        with st.form("itinerary_form"):
            st.header("Trip Details")
            st.text_input("Enter the location (city/country):", key="location")
            st.selectbox("Enter your budget:", ["Low", "Mid-range", "Luxury"], key="budget")
            st.text_input("Enter travel duration (e.g., 5 days, 1 week):", key="travel_dates")
            st.selectbox("Travel party:", ["Solo", "Couple", "Family", "Friends", "Business"], key="travel_party")
            st.multiselect("Activities:", ["Adventure", "Relaxation", "Cultural", "Shopping", "Nightlife", "Nature", "Historical"], key="activities_interests")
            
            st.header("Accommodation")
            st.selectbox("Accommodation type:", ["Hotels", "Hostels", "Vacation Rentals", "Camping", "Resorts"], key="accommodation_type")
            st.multiselect("Proximity preferences:", ["City center", "Near attractions", "Quiet area", "Beachfront", "Downtown"], key="proximity_to_attractions")
            
            st.header("Transport")
            st.selectbox("Transport mode:", ["Car", "Train", "Flight", "Bike", "Public Transport"], key="mode_of_transport")
            st.multiselect("Rental services needed:", ["Car Rentals", "Bike Rentals", "Scooter Rentals", "None"], key="rental_services")
            st.multiselect("Public transport preferences:", ["Buses", "Metro", "Trams", "Taxis", "Ride-sharing"], key="public_transport_preferences")
            
            st.header("Dining Preferences")
            st.multiselect("Dietary needs:", ["Vegetarian", "Vegan", "Gluten-Free", "Halal", "Kosher", "None"], key="dietary_restrictions")
            st.multiselect("Cuisine interests:", ["Local", "International", "Fine Dining", "Street Food", "Vegetarian"], key="interest_in_local_cuisine")
            st.selectbox("Dining experience:", ["Casual", "Fine Dining", "Mix of both"], key="dining_experience")
            st.selectbox("Ambiance:", ["Romantic", "Family-Friendly", "Trendy", "Traditional", "Any"], key="ambiance_preferences")
            st.multiselect("Cuisines:", ["Italian", "Thai", "Indian", "Fusion", "Local Specialties"], key="cuisine_variety")
            
            # Fixed submit button - using st.form_submit_button()
            submitted = st.form_submit_button("Generate My Itinerary", on_click=generate_itinerary)

        # Display results
        if st.session_state.itinerary_data:
            st.success("Itinerary created successfully!")
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Itinerary", "Accommodation", "Transport", "Food", "Budget"])
            
            with tab1:
                st.markdown('<div class="section-header">Daily Itinerary</div>', unsafe_allow_html=True)
                itinerary_content = st.session_state.itinerary_data['trip_plan']
                st.markdown(format_itinerary(itinerary_content), unsafe_allow_html=True)

            with tab2:
                st.markdown('<div class="section-header">Recommended Stays</div>', unsafe_allow_html=True)
                st.markdown(format_itinerary(st.session_state.itinerary_data['accommodation']), unsafe_allow_html=True)

            with tab3:
                st.markdown('<div class="section-header">Transport Options</div>', unsafe_allow_html=True)
                st.markdown(format_itinerary(st.session_state.itinerary_data['transport']), unsafe_allow_html=True)

            with tab4:
                st.markdown('<div class="section-header">Dining Guide</div>', unsafe_allow_html=True)
                st.markdown(format_itinerary(st.session_state.itinerary_data['food']), unsafe_allow_html=True)

            with tab5:
                st.markdown('<div class="section-header">Budget Breakdown</div>', unsafe_allow_html=True)
                st.markdown(format_itinerary(st.session_state.itinerary_data['budget_estimate']), unsafe_allow_html=True)
            
            if st.session_state.pdf_buffer:
                st.download_button(
                    label="Download PDF Itinerary",
                    data=st.session_state.pdf_buffer,
                    file_name=f"{st.session_state.itinerary_data['location']}_itinerary.pdf",
                    mime="application/pdf",
                    key="pdf_download"
                )

    ###########################################################################################
    # Travel History
    elif selected == "Travel History":
        st.header("✈️ My Travel Experiences")
        
        # Simple Add Experience Form
        with st.form("add_experience"):
            city = st.text_input("City Visited", key="city")
            country = st.text_input("Country", key="country")
            rating = st.slider("Rating (1-5)", 1, 5, 3)
            submitted = st.form_submit_button("Save Experience")
            
            if submitted:
                st.session_state.db.child("travel_history").child(st.session_state.user['localId']).push({
                    "city": city.strip(),
                    "country": country.strip(),
                    "rating": rating,
                    "timestamp": datetime.now().isoformat(),
                    "username": st.session_state.user['username']
                })
                st.success("Experience saved!")
                st.session_state.recommendation_type = 'returning_user'
                st.rerun()

        # Display User's History
        user_history = st.session_state.db.child("travel_history").child(st.session_state.user['localId']).get().val()
        if user_history:
            st.subheader("Your Travel Log")
            for exp_id, exp in user_history.items():
                cols = st.columns([4, 1])
                with cols[0]:
                    st.write(f"**{exp['city']}, {exp['country']}** (Rated {exp['rating']}/5)")
                    st.caption(f"Visited on {exp['timestamp'][:10]}")
                with cols[1]:
                    if st.button("Delete", key=f"del_{exp_id}", help="Delete this entry"):
                        st.session_state.db.child("travel_history").child(st.session_state.user['localId']).child(exp_id).remove()
                        st.rerun()
                st.divider()
        else:
            st.info("No travel experiences yet")
            st.session_state.recommendation_type = 'new_user'

    ###########################################################################################
    # Admin Page
    elif selected == "Admin" and is_admin():
        st.header("Admin Dashboard")
        
        # 1. Display All Travel Logs
        st.subheader("All User Travel Logs")
        all_history = st.session_state.db.child("travel_history").get().val() or {}
        
        if all_history:
            # Prepare data for display
            display_data = []
            for user_id, experiences in all_history.items():
                for exp_id, exp in experiences.items():
                    display_data.append({
                        "User": exp.get('username', 'Unknown'),
                        "City": exp.get('city', ''),
                        "Country": exp.get('country', ''),
                        "Rating": f"{exp.get('rating', 0)}/5",
                        "Date": exp.get('timestamp', '')[:10]
                    })
            
            # Show in a compact, scrollable table
            st.dataframe(
                pd.DataFrame(display_data),
                height=300,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No travel history data found")
        
        # 2. Export Button
        st.divider()
        st.subheader("Data Export")
        if st.button("📥 Export All Travel Data"):
            try:
                all_data = []
                histories = st.session_state.db.child("travel_history").get().val() or {}
                
                for user_id, experiences in histories.items():
                    for exp_id, exp in experiences.items():
                        all_data.append({
                            "User": exp.get('username', ''),
                            "City": exp.get('city', ''),
                            "Country": exp.get('country', ''),
                            "Rating": exp.get('rating', 0),
                            "When": exp.get('timestamp', '')[:10]
                        })
                
                df = pd.DataFrame(all_data)
                st.download_button(
                    label="Download as CSV",
                    data=df.to_csv(index=False).encode('utf-8'),
                    file_name="travel_export.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Export failed: {e}")

    ###########################################################################################
    # Logout
    elif selected == "Logout":
        st.session_state.clear()

        st.rerun()
