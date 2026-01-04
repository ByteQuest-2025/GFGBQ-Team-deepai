import streamlit as st
import datetime
from PIL import Image
from mistralai import Mistral
import requests
import os
from dotenv import load_dotenv
import json
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import csv
from difflib import SequenceMatcher
import time

# Load environment variables
load_dotenv()

# Configure Mistral AI API
api_key = os.getenv("MISTRAL_API_KEY")
mistral_client = None
if api_key:
    try:
        mistral_client = Mistral(api_key=api_key)
    except Exception as e:
        st.warning(f"⚠️ Error initializing Mistral AI: {str(e)}")
        mistral_client = None
else:
    st.warning("⚠️ MISTRAL_API_KEY not found in environment. Using fallback mode.")

# Configure Groq API for Audio Transcription
groq_api_key = os.getenv("GROQ_API_KEY")
groq_api_available = groq_api_key is not None
if not groq_api_key:
    st.info("💡 Tip: Get a FREE Groq API key at https://console.groq.com/keys for fast audio transcription!")

# Configure Hugging Face API
hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
hf_api_available = hf_api_key is not None
if not hf_api_key:
    st.info("💡 Tip: Get a FREE Hugging Face API key at https://huggingface.co/settings/tokens for image captioning.")

# Data persistence
COMPLAINTS_FILE = "complaints_data.csv"

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="CivicFix AI - Grievance Redressal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Data Persistence Functions
# ----------------------------
def save_complaint_to_csv(complaint_record):
    """Save complaint to CSV file"""
    file_exists = os.path.isfile(COMPLAINTS_FILE)
    
    with open(COMPLAINTS_FILE, 'a', newline='', encoding='utf-8') as f:
        fieldnames = ['id', 'text', 'category', 'department', 'priority_score', 
                     'priority_level', 'summary', 'submitted_at', 'status', 'sentiment',
                     'impact_score', 'estimated_resolution_days', 'original_language']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        writer.writerow(complaint_record)

def load_complaints_from_csv():
    """Load complaints from CSV"""
    if not os.path.isfile(COMPLAINTS_FILE):
        return []
    
    complaints = []
    with open(COMPLAINTS_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['priority_score'] = int(row['priority_score'])
            row['impact_score'] = int(row.get('impact_score', 1))
            row['estimated_resolution_days'] = int(row.get('estimated_resolution_days', 7))
            complaints.append(row)
    return complaints

# ----------------------------
# Initialize Session State
# ----------------------------
if 'complaints' not in st.session_state:
    st.session_state.complaints = load_complaints_from_csv()

if 'demo_loaded' not in st.session_state:
    st.session_state.demo_loaded = False

# ----------------------------
# AI Helper Functions
# ----------------------------
def transcribe_audio(audio_file):
    """Transcribe audio using Groq's Whisper API (FAST & FREE)"""
    
    if not groq_api_key:
        st.info("💡 To enable automatic voice transcription, add a FREE Groq API key to your .env file.")
        st.info("Get it at: https://console.groq.com/keys (no credit card required!)")
        return "Voice transcription requires Groq API key. Please describe your complaint in the text box below.", False
    
    try:
        # Read audio file
        audio_file.seek(0)
        audio_bytes = audio_file.read()
        
        # Check file size (Groq limit is 25MB)
        file_size_mb = len(audio_bytes) / (1024 * 1024)
        if file_size_mb > 25:
            return "Audio file too large (max 25MB). Please record a shorter message.", False
        
        # Get file extension
        file_name = audio_file.name
        file_ext = file_name.split('.')[-1].lower()
        
        # Supported formats by Groq
        supported_formats = ['flac', 'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'ogg', 'wav', 'webm']
        if file_ext not in supported_formats:
            return f"Audio format '{file_ext}' not supported. Please use: {', '.join(supported_formats)}", False
        
        with st.spinner("🎙️ Transcribing with Groq Whisper (Lightning Fast)..."):
            # Groq API endpoint
            url = "https://api.groq.com/openai/v1/audio/transcriptions"
            
            headers = {
                "Authorization": f"Bearer {groq_api_key}"
            }
            
            # Prepare the file for multipart upload
            files = {
                "file": (file_name, audio_bytes, f"audio/{file_ext}")
            }
            
            data = {
                "model": "whisper-large-v3",  # Groq's fastest Whisper model
                "language": "en",  # Can be changed to auto-detect or specific language
                "response_format": "json",
                "temperature": 0.0  # Lower temperature for more accurate transcription
            }
            
            # Make the request
            response = requests.post(
                url,
                headers=headers,
                files=files,
                data=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get('text', '').strip()
                
                if text and len(text) > 3:
                    st.success("✅ Audio transcribed successfully with Groq Whisper!")
                    return text, True
                else:
                    return "Transcription returned empty text. Please try speaking more clearly.", False
                    
            elif response.status_code == 400:
                error_msg = response.json().get('error', {}).get('message', 'Bad request')
                st.error(f"❌ Error: {error_msg}")
                return f"Audio processing error: {error_msg}", False
                
            elif response.status_code == 401:
                st.error("❌ Invalid Groq API key. Please check your .env file.")
                return "Invalid API key. Please check your Groq API key.", False
                
            elif response.status_code == 413:
                return "Audio file too large. Please record a shorter message.", False
                
            elif response.status_code == 429:
                st.error("❌ Rate limit exceeded. Please wait a moment and try again.")
                return "Rate limit exceeded. Please try again in a moment.", False
                
            else:
                st.error(f"❌ Transcription failed with status code: {response.status_code}")
                return f"Transcription failed. Please try again or describe your complaint manually.", False
        
    except requests.exceptions.Timeout:
        st.error("❌ Request timed out. Please try a shorter audio file.")
        return "Request timed out. Please try a shorter audio clip.", False
        
    except requests.exceptions.ConnectionError:
        st.error("❌ Network connection error. Please check your internet connection.")
        return "Network error. Please check your connection and try again.", False
        
    except Exception as e:
        st.error(f"❌ Transcription error: {str(e)}")
        return f"Error processing audio file: {str(e)}", False

def detect_and_translate(text):
    """Detect language and translate to English if needed"""
    if not mistral_client or not text.strip():
        return text, "English (en)"
    
    try:
        prompt = f"""Detect the language of this text and translate to English if it's not English.
Text: "{text}"

Return ONLY valid JSON:
{{
    "detected_language": "language_name (code)",
    "is_english": true/false,
    "english_translation": "translated text or original if English"
}}"""
        
        response = mistral_client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}]
        )
        
        result_text = response.choices[0].message.content.strip()
        result_text = result_text.replace("```json", "").replace("```", "").strip()
        result = json.loads(result_text)
        
        return result['english_translation'], result['detected_language']
    except:
        return text, "English (en)"

def analyze_sentiment(text):
    """Analyze sentiment of complaint"""
    if not mistral_client or not text.strip():
        return "Neutral"
    
    try:
        prompt = f"""Analyze the sentiment/tone of this complaint. Choose ONE word from: Angry, Frustrated, Concerned, Calm, Neutral.
Text: "{text}"
Return only one word, nothing else."""
        
        response = mistral_client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}]
        )
        
        sentiment = response.choices[0].message.content.strip()
        valid_sentiments = ["Angry", "Frustrated", "Concerned", "Calm", "Neutral"]
        if sentiment in valid_sentiments:
            return sentiment
        return "Neutral"
    except:
        return "Neutral"

def calculate_impact_score(text, category):
    """Calculate how many people might be affected"""
    text_lower = text.lower()
    
    high_impact_keywords = ["entire area", "whole street", "many people", "community", 
                           "neighborhood", "all residents", "public", "everyone", "entire city"]
    medium_impact_keywords = ["my building", "few people", "nearby", "local area", "our colony"]
    
    impact_score = 1  # Default: affects individual
    
    if any(keyword in text_lower for keyword in high_impact_keywords):
        impact_score = 3
    elif any(keyword in text_lower for keyword in medium_impact_keywords):
        impact_score = 2
    
    high_impact_categories = ["Public Safety", "Utilities", "Civic Infrastructure"]
    if category in high_impact_categories:
        impact_score = min(3, impact_score + 1)
    
    return impact_score

def get_estimated_resolution_time(category, priority_score):
    """Estimate resolution time based on category and priority"""
    base_times = {
        "Civic Infrastructure": 7,
        "Sanitation": 3,
        "Public Safety": 1,
        "Healthcare": 2,
        "Education": 5,
        "Utilities": 4,
        "Administrative Delays": 10,
        "Other": 7
    }
    
    base_days = base_times.get(category, 7)
    
    if priority_score >= 7:
        return max(1, int(base_days * 0.5))
    elif priority_score <= 3:
        return int(base_days * 1.5)
    return base_days

def find_similar_complaints(new_text, existing_complaints, threshold=0.6):
    """Find similar complaints using text similarity"""
    if not new_text.strip() or len(existing_complaints) == 0:
        return []
    
    similar = []
    for complaint in existing_complaints[-30:]:  # Check last 30
        similarity = SequenceMatcher(None, new_text.lower()[:200], 
                                    complaint['text'].lower()[:200]).ratio()
        if similarity > threshold:
            similar.append({
                'id': complaint['id'],
                'similarity': f"{similarity*100:.0f}%",
                'text': complaint['text'][:150],
                'category': complaint['category'],
                'status': complaint['status']
            })
    
    return sorted(similar, key=lambda x: float(x['similarity'].rstrip('%')), reverse=True)

def analyze_image_with_ai(image):
    """Analyze uploaded image using Mistral Pixtral Vision Model"""
    
    # First, try Mistral Pixtral if we have Mistral API key
    if mistral_client:
        try:
            # Convert to PIL Image and encode to base64
            if hasattr(image, 'read'):
                image.seek(0)
                pil_image = Image.open(image)
            elif isinstance(image, Image.Image):
                pil_image = image
            else:
                return None
            
            # Resize if too large (max 1024px on longest side for efficiency)
            max_size = 1024
            if max(pil_image.size) > max_size:
                ratio = max_size / max(pil_image.size)
                new_size = tuple(int(dim * ratio) for dim in pil_image.size)
                pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG", quality=85)
            buffered.seek(0)
            image_base64 = base64.b64encode(buffered.read()).decode('utf-8')
            
            with st.spinner("🔍 Analyzing image with Mistral Pixtral Vision AI..."):
                # Use Mistral's vision model
                response = mistral_client.chat.complete(
                    model="pixtral-12b-2409",  # Mistral's vision model
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": """Analyze this image as a citizen grievance complaint photo. Identify what civic problem or issue is shown.

Look for:
- Infrastructure damage (potholes, broken roads, cracks, damaged bridges, broken pavements)
- Sanitation issues (garbage piles, waste, littering, dirty areas, overflowing bins)
- Safety hazards (broken streetlights, damaged structures, unsafe conditions, exposed wires)
- Utility problems (water leakage, drainage issues, electricity problems, broken pipes)
- Public facilities issues (damaged parks, broken benches, vandalism)
- Healthcare/Education facility problems
- Any other civic issues

Provide a detailed 3-4 sentence description of:
1. What specific problem/issue you see in the image
2. The severity/urgency of the issue
3. Which government department should handle this

Be specific and factual. If you cannot identify a clear civic issue, describe what you see."""
                                },
                                {
                                    "type": "image_url",
                                    "image_url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            ]
                        }
                    ]
                )
                
                description = response.choices[0].message.content.strip()
                
                if description and len(description) > 20:
                    st.success("✅ Image analyzed successfully with Mistral Pixtral Vision AI!")
                    return f"Image Analysis: {description}"
                else:
                    st.warning("⚠️ Image analysis returned limited information.")
                    return None
                    
        except Exception as e:
            st.warning(f"⚠️ Mistral Pixtral analysis error: {str(e)}")
            st.info("💡 Falling back to HuggingFace image captioning...")
    
    # Fallback to HuggingFace BLIP if Mistral fails or unavailable
    if hf_api_key:
        try:
            if hasattr(image, 'read'):
                image.seek(0)
                pil_image = Image.open(image)
            elif isinstance(image, Image.Image):
                pil_image = image
            else:
                return None
            
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG", quality=85)
            buffered.seek(0)
            image_bytes = buffered.read()
            
            headers = {"Authorization": f"Bearer {hf_api_key}"}
            
            models = [
                "Salesforce/blip-image-captioning-large",
                "Salesforce/blip-image-captioning-base",
            ]
            
            for model_name in models:
                try:
                    API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
                    
                    with st.spinner(f"🔍 Analyzing with {model_name.split('/')[-1]}..."):
                        response = requests.post(API_URL, headers=headers, data=image_bytes, timeout=45)
                    
                    if response.status_code == 200:
                        result = response.json()
                        caption = None
                        
                        if isinstance(result, list) and len(result) > 0:
                            if isinstance(result[0], dict):
                                caption = result[0].get('generated_text', '') or result[0].get('caption', '')
                            else:
                                caption = str(result[0])
                        elif isinstance(result, dict):
                            caption = result.get('generated_text', '') or result.get('caption', '')
                        
                        if caption and len(caption.strip()) > 10:
                            st.success(f"✅ Image analyzed with {model_name.split('/')[-1]}")
                            return f"Image shows: {caption.strip()}. This appears to be a citizen grievance requiring government attention."
                            
                    elif response.status_code == 503:
                        st.info(f"⏳ Model loading... Trying next option...")
                        time.sleep(3)
                        continue
                        
                except Exception as e:
                    continue
        
        except Exception as e:
            pass
    
    # Enhanced fallback with basic analysis
    st.warning("⚠️ Automatic image analysis unavailable. Using enhanced fallback...")
    
    try:
        if hasattr(image, 'read'):
            image.seek(0)
            pil_image = Image.open(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            return "Image uploaded as visual evidence for grievance complaint."
        
        width, height = pil_image.size
        
        # Get basic image properties
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        file_size_kb = len(buffered.getvalue()) / 1024
        
        if pil_image.mode != 'RGB':
            pil_image_rgb = pil_image.convert('RGB')
        else:
            pil_image_rgb = pil_image
        
        pil_image_small = pil_image_rgb.resize((50, 50))
        pixels = list(pil_image_small.getdata())
        avg_brightness = sum(sum(pixel) for pixel in pixels) / (len(pixels) * 3)
        
        brightness = "bright outdoor scene" if avg_brightness > 180 else "low-light/indoor scene" if avg_brightness < 80 else "moderately lit scene"
        quality = "high-resolution" if width * height > 1000000 else "standard quality"
        
        return f"A {quality} {brightness} image ({width}x{height}px, {file_size_kb:.1f}KB) has been uploaded as visual evidence for this grievance complaint. The image shows potential civic infrastructure or public service issues that require manual review by government officials to determine the specific nature and urgency of the complaint."
            
    except Exception as e:
        return "Image uploaded as visual evidence. Manual review required to assess the grievance."

def analyze_complaint_with_ai(text, image=None):
    """Use AI to analyze complaint and return structured data"""
    
    image_description = ""
    combined_text = text
    
    if image:
        st.info("🔍 Analyzing uploaded image...")
        image_analysis = analyze_image_with_ai(image)
        
        if image_analysis and "manual review recommended" not in image_analysis.lower():
            image_description = image_analysis
            st.success("✅ Image analysis complete!")
            
            # Show what was detected in the image
            with st.expander("👁️ View Image Analysis Results"):
                st.write(image_analysis)
            
            # Combine text and image analysis
            if text and text.strip():
                combined_text = f"Citizen complaint text: {text}\n\n{image_analysis}"
            else:
                combined_text = image_analysis
        else:
            # Fallback if image analysis fails
            if text and text.strip():
                combined_text = text
            else:
                combined_text = "Visual evidence uploaded for grievance complaint. Image content could not be automatically analyzed."
    
    if not mistral_client:
        return analyze_complaint_fallback(combined_text)
    
    try:
        prompt = f"""You are analyzing a citizen grievance complaint for a government redressal system. 

Complaint Information:
{combined_text}

Analyze this complaint and provide a JSON response with these fields:

1. category: Choose the most appropriate from ["Civic Infrastructure", "Sanitation", "Public Safety", "Healthcare", "Education", "Utilities", "Administrative Delays", "Other"]

2. department: Assign to the appropriate government department (e.g., "Public Works Department", "Municipal Sanitation Dept", "Police Department", "Health Department", "Education Department", "Electricity Board", "Water Supply Dept", "District Administration")

3. priority_score: Rate from 1-10 where 10 is most urgent. Consider:
   - Safety risks to citizens
   - Health hazards
   - Number of people affected
   - Time sensitivity
   - Severity of the issue

4. priority_level: Based on priority_score:
   - "High 🔴" if score >= 7
   - "Medium 🟠" if score >= 4
   - "Low 🟢" if score < 4

5. summary: Write a clear 2-3 sentence summary of the complaint

6. urgency_reasons: List 2-3 specific reasons explaining the priority score

Respond with ONLY valid JSON in this exact format:
{{
    "category": "...",
    "department": "...",
    "priority_score": <number>,
    "priority_level": "...",
    "summary": "...",
    "urgency_reasons": ["reason 1", "reason 2", "reason 3"]
}}"""

        response = mistral_client.chat.complete(
            model="mistral-medium-latest",
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Clean markdown code blocks
        response_text = response_text.replace("```json", "").replace("```", "").strip()
        
        # Parse JSON
        result = json.loads(response_text)
        
        # Validate all required fields exist
        required_keys = ['category', 'department', 'priority_score', 'priority_level', 'summary', 'urgency_reasons']
        if not all(key in result for key in required_keys):
            st.warning("⚠️ AI response incomplete. Using enhanced fallback analysis...")
            return analyze_complaint_fallback(combined_text)
        
        # Validate priority_score is a number
        if not isinstance(result['priority_score'], (int, float)):
            result['priority_score'] = 5
        
        return result
        
    except json.JSONDecodeError as e:
        st.warning(f"⚠️ Could not parse AI response. Using fallback analysis...")
        return analyze_complaint_fallback(combined_text)
    except Exception as e:
        st.warning(f"⚠️ AI analysis error: {str(e)}. Using fallback analysis...")
        return analyze_complaint_fallback(combined_text)

def analyze_complaint_fallback(text):
    """Fallback rule-based analysis"""
    text_lower = text.lower()
    
    if any(word in text_lower for word in ["road", "pothole", "bridge", "street", "infrastructure"]):
        category, department = "Civic Infrastructure", "Public Works Department"
    elif any(word in text_lower for word in ["garbage", "waste", "sanitation", "dirty", "trash"]):
        category, department = "Sanitation", "Municipal Sanitation Dept"
    elif any(word in text_lower for word in ["crime", "unsafe", "security", "theft", "violence"]):
        category, department = "Public Safety", "Police Department"
    elif any(word in text_lower for word in ["hospital", "medicine", "health", "doctor", "medical"]):
        category, department = "Healthcare", "Health Department"
    elif any(word in text_lower for word in ["school", "education", "teacher", "student"]):
        category, department = "Education", "Education Department"
    elif any(word in text_lower for word in ["electricity", "power", "light", "current"]):
        category, department = "Utilities", "Electricity Board"
    elif any(word in text_lower for word in ["water", "supply", "pipeline", "drainage"]):
        category, department = "Utilities", "Water Supply Dept"
    else:
        category, department = "Administrative Delays", "District Administration"
    
    urgent_words = ["urgent", "emergency", "danger", "critical", "immediate", "hazard"]
    if any(word in text_lower for word in urgent_words):
        priority_score, priority_level = 9, "High 🔴"
    elif len(text) > 150:
        priority_score, priority_level = 6, "Medium 🟠"
    else:
        priority_score, priority_level = 3, "Low 🟢"
    
    summary = text[:200] + "..." if len(text) > 200 else text
    
    return {
        "category": category,
        "department": department,
        "priority_score": priority_score,
        "priority_level": priority_level,
        "summary": summary,
        "urgency_reasons": ["Based on keyword analysis", "Content length considered"]
    }

def load_demo_data():
    """Load demo complaints for presentation"""
    demo_complaints = [
        {
            "text": "Massive pothole on MG Road near City Mall causing daily accidents. Two-wheeler riders at extreme risk. Urgent repair needed!",
            "category": "Civic Infrastructure",
            "priority_score": 9,
            "department": "Public Works Department",
            "sentiment": "Angry"
        },
        {
            "text": "Garbage not collected for 5 days in Vaishali Nagar, Sector 12. Foul smell, health hazard for children and elderly.",
            "category": "Sanitation",
            "priority_score": 8,
            "department": "Municipal Sanitation Dept",
            "sentiment": "Frustrated"
        },
        {
            "text": "Street lights not working in Mansarovar residential area for 3 weeks. Women feeling unsafe during evening hours.",
            "category": "Public Safety",
            "priority_score": 7,
            "department": "Electricity Board",
            "sentiment": "Concerned"
        },
        {
            "text": "Water supply disrupted in entire Malviya Nagar for 48 hours. Affecting 500+ families. No prior notice given.",
            "category": "Utilities",
            "priority_score": 8,
            "department": "Water Supply Dept",
            "sentiment": "Frustrated"
        },
        {
            "text": "Government school in Raja Park lacks basic facilities. No drinking water, damaged toilets affecting 300 students.",
            "category": "Education",
            "priority_score": 7,
            "department": "Education Department",
            "sentiment": "Concerned"
        },
        {
            "text": "Stray dogs attacking residents in Civil Lines area. Child was bitten yesterday. Need immediate action.",
            "category": "Public Safety",
            "priority_score": 9,
            "department": "Municipal Corporation",
            "sentiment": "Angry"
        },
        {
            "text": "Birth certificate application pending for 3 months at Jaipur Municipal Corporation. No response from officials.",
            "category": "Administrative Delays",
            "priority_score": 5,
            "department": "District Administration",
            "sentiment": "Frustrated"
        },
        {
            "text": "Primary health center in Sanganer has no doctor for last 2 weeks. Patients being turned away daily.",
            "category": "Healthcare",
            "priority_score": 8,
            "department": "Health Department",
            "sentiment": "Concerned"
        }
    ]
    
    for i, demo in enumerate(demo_complaints):
        complaint_id = f"DEMO-{datetime.datetime.now().strftime('%Y%m%d')}-{i+1:03d}"
        submitted_time = (datetime.datetime.now() - datetime.timedelta(days=i, hours=i*2)).strftime("%Y-%m-%d %H:%M:%S")
        
        complaint_record = {
            "id": complaint_id,
            "text": demo['text'],
            "category": demo['category'],
            "department": demo['department'],
            "priority_score": demo['priority_score'],
            "priority_level": "High 🔴" if demo['priority_score'] >= 7 else "Medium 🟠" if demo['priority_score'] >= 4 else "Low 🟢",
            "summary": demo['text'][:150] + "...",
            "submitted_at": submitted_time,
            "status": "Open",
            "sentiment": demo['sentiment'],
            "impact_score": calculate_impact_score(demo['text'], demo['category']),
            "estimated_resolution_days": get_estimated_resolution_time(demo['category'], demo['priority_score']),
            "original_language": "English (en)"
        }
        
        st.session_state.complaints.append(complaint_record)
    
    st.session_state.demo_loaded = True

# ----------------------------
# HERO SECTION
# ----------------------------
st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='font-size: 2.5rem; font-weight: 700; color: #FFFFFF; margin-bottom: 0.5rem;'>
            🏛️ CivicFix AI - Intelligent Grievance Redressal
        </h1>
        <p style='font-size: 1.3rem; font-weight: 500; color: #FFFFFF; margin-bottom: 0.5rem;'>
            Smart, Transparent & Fast Citizen Complaint Resolution
        </p>
        <p style='font-size: 1rem; color: #FFFFFF; max-width: 800px; margin: 0 auto;'>
            AI-powered platform that automatically categorizes, prioritizes, and routes citizen grievances to the right authorities
        </p>
    </div>
""", unsafe_allow_html=True)

st.divider()

# ----------------------------
# Feature Cards
# ----------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    with st.container(border=True):
        st.markdown("<div style='text-align: center;'><h3>🧠 AI Analysis</h3></div>", unsafe_allow_html=True)
        st.write("NLP-powered understanding of complaints")

with col2:
    with st.container(border=True):
        st.markdown("<div style='text-align: center;'><h3>⚡ Smart Priority</h3></div>", unsafe_allow_html=True)
        st.write("Urgent cases identified & escalated")

with col3:
    with st.container(border=True):
        st.markdown("<div style='text-align: center;'><h3>🎯 Auto Routing</h3></div>", unsafe_allow_html=True)
        st.write("Instant department assignment")

with col4:
    with st.container(border=True):
        st.markdown("<div style='text-align: center;'><h3>🌍 Multi-lingual</h3></div>", unsafe_allow_html=True)
        st.write("Supports mixed languages")

st.divider()

# ----------------------------
# Sidebar Navigation
# ----------------------------
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    page = st.radio(
        "Choose a page",
        ["🏠 Submit Grievance", "📊 Admin Dashboard"],
        label_visibility="collapsed"
    )
    
    st.divider()
    st.markdown("### 📈 Quick Stats")
    total_complaints = len(st.session_state.complaints)
    high_priority = sum(1 for c in st.session_state.complaints if c.get('priority_score', 0) >= 7)
    st.metric("Total Complaints", total_complaints)
    st.metric("High Priority", high_priority)
    
    st.divider()
    
    # Demo data loader
    if not st.session_state.demo_loaded:
        if st.button("🎬 Load Demo Data", use_container_width=True):
            with st.spinner("Loading demo complaints..."):
                load_demo_data()
            st.success("✅ Demo data loaded!")
            st.rerun()
    else:
        if st.button("🗑️ Clear Demo Data", use_container_width=True):
            st.session_state.complaints = [c for c in st.session_state.complaints if not c['id'].startswith('DEMO-')]
            st.session_state.demo_loaded = False
            st.success("✅ Demo data cleared!")
            st.rerun()
    
    st.divider()
    
    # API Status
    st.markdown("### 🔌 API Status")
    if mistral_client:
        st.success("✅ Mistral AI")
    else:
        st.error("❌ Mistral AI")
    
    if groq_api_available:
        st.success("✅ Groq Whisper")
    else:
        st.error("❌ Groq Whisper")
    
    if hf_api_available:
        st.success("✅ HuggingFace")
    else:
        st.info("⚪ HuggingFace")

# ----------------------------
# PAGE: Submit Grievance
# ----------------------------
if page == "🏠 Submit Grievance":
    st.markdown("## 📝 Submit a Grievance")
    
    with st.container(border=True):
        st.markdown("#### 📋 Choose how you want to submit your complaint")

        # Create tabs with better styling
        tab1, tab2, tab3 = st.tabs(
            ["✍️ Text Complaint", "📷 Photo Evidence", "🎤 Voice Complaint"]
        )

        grievance_text = ""
        image_file = None
        audio_file = None
        original_text = ""
        detected_language = "English (en)"

        # ---- TEXT ----
        with tab1:
            st.markdown("##### ✍️ Describe your complaint in text")
            grievance_text = st.text_area(
                "Write your complaint here:",
                height=180,
                placeholder="Example: Garbage has not been collected for 3 days in my area causing severe health issues...",
                key="text_input_main",
                help="Describe the issue in detail. You can write in English, Hindi, or any regional language."
            )
            original_text = grievance_text
            
            if grievance_text:
                st.success(f"✅ Text input received ({len(grievance_text)} characters)")

        # ---- PHOTO ----
        with tab2:
            st.markdown("##### 📷 Upload photo evidence of the issue")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                image_file = st.file_uploader(
                    "Choose an image file:",
                    type=["jpg", "jpeg", "png"],
                    key="image_uploader_main",
                    help="Upload a clear photo showing the civic issue"
                )
            
            with col2:
                if image_file:
                    st.success("✅ Image uploaded successfully!")
                    file_size = image_file.size / 1024
                    st.caption(f"📎 {image_file.name} ({file_size:.1f} KB)")
            
            if image_file:
                # Display image with better styling
                img = Image.open(image_file)
                st.image(img, caption="📸 Your Photo Evidence", use_column_width=True)
                
                # AI capability info
                with st.expander("🤖 AI Image Analysis Capabilities", expanded=False):
                    if mistral_client:
                        st.success("✅ **Mistral Pixtral Vision** - Advanced AI image analysis enabled!")
                        st.write("• Automatically identifies potholes, garbage, damaged infrastructure")
                        st.write("• Detects safety hazards and civic issues")
                        st.write("• Generates detailed descriptions")
                    elif hf_api_available:
                        st.info("✅ **HuggingFace BLIP** - Backup image analysis available")
                    else:
                        st.warning("⚠️ Add MISTRAL_API_KEY for advanced vision analysis")
                
                # Optional text description
                st.markdown("---")
                st.markdown("##### ✍️ Additional details (optional)")
                image_description = st.text_area(
                    "Add more context to help AI understand better:",
                    height=120,
                    placeholder="Example: This pothole has been here for 2 weeks and caused 3 accidents. Located near the main market.",
                    key="image_text_desc",
                    help="Optional: Add location, duration, or other important details"
                )
                if image_description:
                    grievance_text = image_description
                    original_text = image_description
                    st.success(f"✅ Additional text added ({len(image_description)} characters)")

        # ---- AUDIO ----
        with tab3:
            st.markdown("##### 🎤 Submit your complaint by voice")
            
            # Tips section with better design
            with st.expander("💡 Tips for best audio transcription results", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**📱 Supported Formats:**")
                    st.write("• MP3, WAV, M4A")
                    st.write("• OGG, WEBM, FLAC")
                    st.write("• MP4, MPEG, MPGA")
                    st.write("")
                    st.write("**🎯 Best Practices:**")
                    st.write("• Speak clearly and slowly")
                    st.write("• Minimize background noise")
                    st.write("• Keep under 25MB (~10 mins)")
                with col2:
                    st.write("**🌍 Languages Supported:**")
                    st.write("• English, Hindi")
                    st.write("• Regional Indian languages")
                    st.write("")
                    st.write("**⚡ Processing Speed:**")
                    st.write("• Groq Whisper: 5-10 seconds")
                    st.write("• Ultra-fast transcription")
            
            # API status
            if groq_api_key:
                st.success("✅ Voice transcription enabled with **Groq Whisper AI** (Lightning Fast!)")
            else:
                st.warning("⚠️ Audio transcription unavailable - Add GROQ_API_KEY to enable")
                st.caption("Get FREE API key at: https://console.groq.com/keys")
            
            # File uploader
            col1, col2 = st.columns([1, 1])
            
            with col1:
                audio_file = st.file_uploader(
                    "Choose an audio file:",
                    type=["mp3", "wav", "m4a", "ogg", "webm", "flac", "mp4", "mpeg", "mpga"],
                    key="audio_uploader_main",
                    help="Record your complaint and upload. AI will transcribe automatically."
                )
            
            with col2:
                if audio_file:
                    st.success("✅ Audio uploaded successfully!")
                    file_size_mb = audio_file.size / (1024 * 1024)
                    st.caption(f"📎 {audio_file.name} ({file_size_mb:.2f} MB)")
            
            if audio_file:
                # Audio player
                st.audio(audio_file, format=f'audio/{audio_file.type.split("/")[-1]}')
                
                st.info("💡 Click **'Submit Grievance'** button below - audio will be automatically transcribed!")
                
                # Optional text description
                st.markdown("---")
                st.markdown("##### ✍️ Additional details (optional)")
                manual_desc = st.text_area(
                    "Add written details to complement your voice complaint:",
                    height=120,
                    placeholder="Example: Location details, specific landmarks, how long this has been an issue...",
                    key="audio_text_desc",
                    help="Optional: Add extra context that might not be in the audio"
                )
                if manual_desc:
                    original_text = manual_desc
                    st.success(f"✅ Additional text added ({len(manual_desc)} characters)")

    # ----------------------------
    # Submit Button with improved styling
    # ----------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    
    # SMART PRIORITY DETECTION - Latest upload wins
    # Priority: Audio > Image > Text (most recently uploaded takes precedence)
    has_text_only = grievance_text.strip() != "" and not image_file and not audio_file
    has_image_only = image_file is not None and not audio_file
    has_audio_only = audio_file is not None
    
    # Determine what to show and process
    if has_audio_only:
        # Audio is present - only show audio status
        if original_text:
            st.info(f"🎤 **Audio + Additional Text ready**")
        else:
            st.info(f"🎤 **Audio ready for transcription**")
        current_mode = "audio"
    elif has_image_only:
        # Image is present (no audio) - only show image status
        if grievance_text:
            st.info(f"📷 **Image + Additional Text ready**")
        else:
            st.info(f"📷 **Image ready for AI analysis**")
        current_mode = "image"
    elif has_text_only:
        # Only text is present
        st.info(f"📝 **Text complaint ready** ({len(grievance_text)} characters)")
        current_mode = "text"
    else:
        current_mode = None
    
    left, center, right = st.columns([3, 2, 3])
    with center:
        submit = st.button(
            "🚀 Submit Grievance", 
            use_container_width=True, 
            type="primary",
            help="Click to submit your complaint for AI analysis"
        )

    # ----------------------------
    # AI Analysis Section
    # ----------------------------
    if submit:
        # Validate that at least one input is provided
        if not has_text_only and not has_image and not has_audio:
            st.error("⚠️ Please provide at least one input: text, photo, or audio.")
        else:
            # Store what we're analyzing to prevent re-analysis
            analyzing_text = grievance_text if has_text_only else ""
            analyzing_image = image_file if has_image else None
            analyzing_audio = audio_file if has_audio else None
            analyzing_original = original_text
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process based on current input type
            if analyzing_audio:
                # AUDIO: Transcribe first
                status_text.text("🎙️ Step 1/4: Transcribing audio...")
                progress_bar.progress(25)
                
                with st.spinner("🎙️ Transcribing audio with Groq Whisper..."):
                    transcription, success = transcribe_audio(analyzing_audio)
                
                if success:
                    st.success("✅ Audio transcribed successfully!")
                    
                    with st.expander("👁️ View Transcribed Text"):
                        st.write(transcription)
                    
                    # Combine with additional text if provided
                    if analyzing_original:
                        analyzing_text = f"{transcription}\n\nAdditional details: {analyzing_original}"
                    else:
                        analyzing_text = transcription
                    
                    # Language detection
                    if mistral_client and transcription.strip():
                        status_text.text("🌍 Step 2/4: Detecting language...")
                        progress_bar.progress(40)
                        
                        with st.spinner("🌍 Detecting language..."):
                            translated_text, detected_language = detect_and_translate(transcription)
                        
                        if "English" not in detected_language:
                            st.info(f"🌐 Detected Language: **{detected_language}** → Translated to English")
                            with st.expander("📄 View Original Transcription"):
                                st.write(transcription)
                            if analyzing_original:
                                analyzing_text = f"{translated_text}\n\nAdditional details: {analyzing_original}"
                            else:
                                analyzing_text = translated_text
                    
                    # Clear image to prevent analysis
                    analyzing_image = None
                else:
                    st.error(transcription)
                    st.error("❌ Audio transcription failed.")
                    progress_bar.empty()
                    status_text.empty()
                    st.stop()
            
            elif analyzing_image:
                # IMAGE: Keep image and optional text
                status_text.text("🖼️ Step 1/4: Preparing image analysis...")
                progress_bar.progress(25)
                # analyzing_image already set
                # analyzing_text already has optional text from image tab
            
            elif analyzing_text:
                # TEXT ONLY: Language detection
                status_text.text("🌍 Step 1/4: Detecting language...")
                progress_bar.progress(25)
                
                with st.spinner("🌍 Detecting language..."):
                    translated_text, detected_language = detect_and_translate(analyzing_text)
                
                if "English" not in detected_language:
                    st.info(f"🌐 Detected Language: **{detected_language}** → Translated to English")
                    with st.expander("📄 View Original Text"):
                        st.write(analyzing_original)
                    analyzing_text = translated_text
            
            # AI Analysis
            status_text.text("🔍 Step 2/4: Analyzing complaint content...")
            progress_bar.progress(50)
            time.sleep(0.3)
            
            status_text.text("🧠 Step 3/4: AI classifying and prioritizing...")
            progress_bar.progress(75)
            time.sleep(0.3)
            
            status_text.text("🎯 Step 4/4: Routing to department...")
            progress_bar.progress(90)
            
            with st.spinner("🤖 AI is analyzing your complaint..."):
                analysis = analyze_complaint_with_ai(
                    analyzing_text if analyzing_text.strip() else "",
                    analyzing_image
                )
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            
            # Validate analysis
            required_keys = ['category', 'department', 'priority_score', 'priority_level', 'summary', 'urgency_reasons']
            if not all(key in analysis for key in required_keys):
                fallback_text = analyzing_text if analyzing_text.strip() else "Image uploaded"
                analysis = analyze_complaint_fallback(fallback_text)
            
            # Additional analysis
            sentiment = analyze_sentiment(analyzing_text if analyzing_text else "")
            impact_score = calculate_impact_score(analyzing_text if analyzing_text else "", analysis['category'])
            resolution_days = get_estimated_resolution_time(analysis['category'], analysis['priority_score'])
            
            st.success("✅ Grievance Submitted Successfully!")
            
            # Check for similar complaints
            if analyzing_text.strip():
                similar = find_similar_complaints(analyzing_text, st.session_state.complaints)
                if similar:
                    with st.expander(f"⚠️ Found {len(similar)} similar complaint(s) - Click to view", expanded=False):
                        st.warning("Similar complaints detected. This might be a recurring issue in your area.")
                        for s in similar:
                            st.markdown(f"**{s['id']}** - {s['category']} ({s['similarity']} match)")
                            st.caption(f"Status: {s['status']} | {s['text'][:100]}...")
                            st.divider()

            st.markdown("## 🔍 AI Analysis & Routing Results")

            with st.container(border=True):
                # Main metrics with better styling
                c1, c2, c3, c4, c5 = st.columns(5)

                with c1:
                    st.metric(
                        label="📂 Category",
                        value=analysis['category'],
                        help="AI-identified complaint category"
                    )
                
                with c2:
                    priority_color = "🔴" if analysis['priority_score'] >= 7 else "🟠" if analysis['priority_score'] >= 4 else "🟢"
                    st.metric(
                        label="⚠️ Priority",
                        value=analysis['priority_level'],
                        help=f"Urgency level: {priority_color}"
                    )
                
                with c3:
                    st.metric(
                        label="🏢 Department",
                        value=analysis['department'],
                        help="Assigned government department"
                    )
                
                with c4:
                    st.metric(
                        label="📊 Priority Score",
                        value=f"{analysis['priority_score']}/10",
                        help="AI-calculated urgency score"
                    )
                
                with c5:
                    sentiment_emoji = {"Angry": "😡", "Frustrated": "😤", "Concerned": "😟", "Calm": "😐", "Neutral": "😐"}
                    st.metric(
                        label="😊 Sentiment",
                        value=f"{sentiment} {sentiment_emoji.get(sentiment, '😐')}",
                        help="Detected citizen emotion"
                    )

                st.markdown("---")
                
                st.markdown("### 📄 Complaint Summary")
                st.info(analysis['summary'])

                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.markdown("### ⚡ Urgency Analysis")
                    for i, reason in enumerate(analysis['urgency_reasons'], 1):
                        st.write(f"**{i}.** {reason}")
                
                with col_right:
                    st.markdown("### 📊 Impact & Timeline")
                    impact_labels = {1: "Individual", 2: "Local Area", 3: "Community-wide"}
                    impact_icons = {1: "👤", 2: "🏘️", 3: "🌆"}
                    
                    st.write(f"{impact_icons[impact_score]} **Impact Level:** {impact_labels[impact_score]} (Score: {impact_score}/3)")
                    st.write(f"⏱️ **Estimated Resolution:** {resolution_days} days")
                    st.write(f"🌍 **Language:** {detected_language}")
                
                st.markdown("---")

                st.markdown("### 📋 What Happens Next?")
                
                steps_col1, steps_col2 = st.columns(2)
                
                with steps_col1:
                    st.write(f"**1️⃣ Routing**")
                    st.caption(f"→ Complaint sent to **{analysis['department']}**")
                    
                    st.write(f"**2️⃣ Priority Assignment**")
                    st.caption(f"→ **{analysis['priority_level']}** (Score: {analysis['priority_score']}/10)")
                    
                    st.write(f"**3️⃣ Timeline**")
                    st.caption(f"→ Expected resolution in **{resolution_days} days**")
                
                with steps_col2:
                    st.write(f"**4️⃣ Notifications**")
                    st.caption("→ Updates via SMS/Email (if registered)")
                    
                    st.write(f"**5️⃣ Tracking**")
                    st.caption("→ Use Complaint ID below to track status")
                    
                    st.write(f"**6️⃣ Resolution**")
                    st.caption("→ Department will address the issue")

                st.markdown("---")

                st.markdown("### 🆔 Complaint Information")
                complaint_id = f"CIV-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
                submitted_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                info_col1, info_col2 = st.columns(2)
                
                with info_col1:
                    st.markdown("**Complaint ID:**")
                    st.code(complaint_id, language=None)
                    st.caption("📋 Save this ID to track your complaint")
                
                with info_col2:
                    status_info = {
                        "Status": "Open",
                        "Submitted": submitted_at,
                        "AI Confidence": "High" if mistral_client else "Fallback",
                        "Processing Time": "< 15 seconds"
                    }
                    
                    for key, value in status_info.items():
                        st.write(f"**{key}:** {value}")
                
                # Mock notification
                st.toast("📧 Confirmation sent to registered contact", icon="✅")

            # Store complaint
            complaint_record = {
                "id": complaint_id,
                "text": analyzing_original if analyzing_original else analyzing_text,
                "category": analysis['category'],
                "department": analysis['department'],
                "priority_score": analysis['priority_score'],
                "priority_level": analysis['priority_level'],
                "summary": analysis['summary'],
                "submitted_at": submitted_at,
                "status": "Open",
                "sentiment": sentiment,
                "impact_score": impact_score,
                "estimated_resolution_days": resolution_days,
                "original_language": detected_language
            }
            
            st.session_state.complaints.append(complaint_record)
            save_complaint_to_csv(complaint_record)
            
            # Force page refresh to clear file uploaders after successful submission
            st.rerun()

# ----------------------------
# PAGE: Admin Dashboard
# ----------------------------
elif page == "📊 Admin Dashboard":
    st.markdown("## 📊 Admin Dashboard - Grievance Analytics")
    
    if len(st.session_state.complaints) == 0:
        st.info("📭 No complaints submitted yet. Submit a grievance or load demo data to see analytics.")
    else:
        df = pd.DataFrame(st.session_state.complaints)
        
        # Key Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Complaints", len(df))
        col2.metric("High Priority", len(df[df['priority_score'] >= 7]))
        col3.metric("Medium Priority", len(df[(df['priority_score'] >= 4) & (df['priority_score'] < 7)]))
        col4.metric("Open Cases", len(df[df['status'] == 'Open']))
        col5.metric("Avg Priority", f"{df['priority_score'].mean():.1f}/10")
        
        st.divider()
        
        # Filters
        st.markdown("### 🔍 Filters")
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            category_filter = st.multiselect(
                "Filter by Category", 
                df['category'].unique(), 
                default=list(df['category'].unique())
            )
        
        with filter_col2:
            priority_filter = st.multiselect(
                "Filter by Priority", 
                df['priority_level'].unique(),
                default=list(df['priority_level'].unique())
            )
        
        with filter_col3:
            dept_filter = st.multiselect(
                "Filter by Department",
                df['department'].unique(),
                default=list(df['department'].unique())
            )
        
        # Apply filters
        filtered_df = df[
            (df['category'].isin(category_filter)) &
            (df['priority_level'].isin(priority_filter)) &
            (df['department'].isin(dept_filter))
        ]
        
        st.caption(f"Showing {len(filtered_df)} of {len(df)} complaints")
        
        st.divider()
        
        # Charts Row 1
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("### 📂 Complaints by Category")
            if len(filtered_df) > 0:
                category_counts = filtered_df['category'].value_counts()
                fig1, ax1 = plt.subplots(figsize=(8, 6))
                colors = plt.cm.Set3(range(len(category_counts)))
                ax1.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
                ax1.axis('equal')
                st.pyplot(fig1)
            else:
                st.info("No data to display")
        
        with col_chart2:
            st.markdown("### ⚠️ Priority Distribution")
            if len(filtered_df) > 0:
                priority_counts = filtered_df['priority_level'].value_counts()
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                colors_priority = {'High 🔴': '#ff4444', 'Medium 🟠': '#ff8800', 'Low 🟢': '#44ff44'}
                bar_colors = [colors_priority.get(p, '#cccccc') for p in priority_counts.index]
                ax2.bar(priority_counts.index, priority_counts.values, color=bar_colors)
                ax2.set_ylabel('Number of Complaints')
                ax2.set_xlabel('Priority Level')
                plt.xticks(rotation=15)
                st.pyplot(fig2)
            else:
                st.info("No data to display")
        
        st.divider()
        
        # Charts Row 2
        col_chart3, col_chart4 = st.columns(2)
        
        with col_chart3:
            st.markdown("### 🏢 Department Workload")
            if len(filtered_df) > 0:
                dept_counts = filtered_df['department'].value_counts()
                st.bar_chart(dept_counts)
            else:
                st.info("No data to display")
        
        with col_chart4:
            st.markdown("### 😊 Sentiment Analysis")
            if len(filtered_df) > 0 and 'sentiment' in filtered_df.columns:
                sentiment_counts = filtered_df['sentiment'].value_counts()
                fig4, ax4 = plt.subplots(figsize=(8, 6))
                sentiment_colors = {'Angry': '#ff0000', 'Frustrated': '#ff6600', 'Concerned': '#ffcc00', 'Calm': '#00cc00', 'Neutral': '#888888'}
                bar_colors_sentiment = [sentiment_colors.get(s, '#cccccc') for s in sentiment_counts.index]
                ax4.barh(sentiment_counts.index, sentiment_counts.values, color=bar_colors_sentiment)
                ax4.set_xlabel('Number of Complaints')
                plt.tight_layout()
                st.pyplot(fig4)
            else:
                st.info("No sentiment data available")
        
        st.divider()
        
        # Timeline
        if len(filtered_df) > 0:
            st.markdown("### 📈 Complaints Timeline")
            try:
                timeline_df = filtered_df.copy()
                timeline_df['date'] = pd.to_datetime(timeline_df['submitted_at']).dt.date
                daily_counts = timeline_df.groupby('date').size()
                st.line_chart(daily_counts)
            except:
                st.info("Timeline data unavailable")
        
        st.divider()
        
        # Recent Complaints Table
        st.markdown("### 📋 Recent Complaints")
        display_cols = ['id', 'category', 'department', 'priority_level', 'priority_score', 'sentiment', 'submitted_at', 'status']
        available_cols = [col for col in display_cols if col in filtered_df.columns]
        
        display_df = filtered_df[available_cols].copy()
        display_df = display_df.sort_values('submitted_at', ascending=False)
        
        st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)
        
        # Export options
        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            if st.button("📥 Export Filtered Data as CSV", use_container_width=True):
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download CSV File",
                    data=csv,
                    file_name=f"grievances_filtered_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col_export2:
            if st.button("📥 Export All Data as CSV", use_container_width=True):
                csv_all = df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download All Data",
                    data=csv_all,
                    file_name=f"grievances_all_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        st.divider()
        
        # High Priority Alerts
        high_priority_complaints = filtered_df[filtered_df['priority_score'] >= 8]
        if len(high_priority_complaints) > 0:
            st.markdown("### 🚨 Critical Priority Alerts")
            for _, complaint in high_priority_complaints.iterrows():
                with st.expander(f"🔴 {complaint['id']} - {complaint['category']} (Score: {complaint['priority_score']}/10)"):
                    st.write(f"**Department:** {complaint['department']}")
                    st.write(f"**Submitted:** {complaint['submitted_at']}")
                    st.write(f"**Summary:** {complaint['summary']}")
                    st.write(f"**Status:** {complaint['status']}")

# ----------------------------
# Footer
# ----------------------------
st.divider()
st.markdown("""
    <div style='text-align: center; padding: 1rem 0;'>
        <p style='font-size: 0.95rem; font-weight: 600; color: #374151; margin-bottom: 0.3rem;'>
            🏆 CivicFix AI - Hackathon Prototype
        </p>
        <p style='font-size: 0.85rem; color: #6b7280;'>
            Built for Byte Quest 2025 | Team Deep AI
        </p>
        <p style='font-size: 0.8rem; color: #9ca3af;'>
            Powered by Mistral AI • Groq Whisper • HuggingFace • Streamlit
        </p>
    </div>
""", unsafe_allow_html=True)