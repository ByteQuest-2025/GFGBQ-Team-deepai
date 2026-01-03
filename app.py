import streamlit as st
import datetime
from PIL import Image
from mistralai import Mistral
import os
from dotenv import load_dotenv
import json
import pandas as pd
import matplotlib.pyplot as plt

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

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="AI Grievance Redressal",
    layout="wide"
)

# ----------------------------
# HERO SECTION
# ----------------------------
st.markdown("# 🏛️ AI-Powered Grievance Redressal System")
st.markdown(
    "### 📢 Intelligent, Transparent & Fast Citizen Complaint Resolution"
)
st.caption(
    "Citizens can submit grievances using **text, images, or voice**. "
    "AI automatically classifies, prioritizes, and routes them to the right authority."
)
st.divider()

# ----------------------------
# FEATURE CARDS (HACKATHON WOW)
# ----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.markdown("### 🧠 Smart AI Analysis")
        st.write(
            "Automatically understands complaints using NLP and AI models."
        )

with col2:
    with st.container(border=True):
        st.markdown("### ⚡ Priority Detection")
        st.write(
            "Urgent and critical grievances are identified and escalated faster."
        )

with col3:
    with st.container(border=True):
        st.markdown("### 🏢 Auto Routing")
        st.write(
            "Complaints are routed to the correct department instantly."
        )

st.divider()

# ----------------------------
# INITIALIZE SESSION STATE
# ----------------------------
if 'complaints' not in st.session_state:
    st.session_state.complaints = []

# ----------------------------
# AI-POWERED HELPER FUNCTIONS
# ----------------------------
def analyze_complaint_with_ai(text, image=None):
    """Use Mistral AI to analyze complaint and return structured data"""
    
    if not mistral_client:
        # Fallback to simple rule-based analysis
        return analyze_complaint_fallback(text)
    
    try:
        prompt = f"""Analyze the following citizen grievance complaint and provide a JSON response with:
1. category: One of ["Civic Infrastructure", "Sanitation", "Public Safety", "Healthcare", "Education", "Utilities", "Administrative Delays", "Other"]
2. department: The appropriate government department to handle this (e.g., "Public Works Department", "Municipal Sanitation Dept", "Police Department", "Health Department", "Education Department", "Electricity Board", "Water Supply Dept", "District Administration")
3. priority_score: A number from 1-10 where 10 is most urgent (consider factors like safety, health risks, impact on public, time sensitivity)
4. priority_level: "High 🔴" if score >= 7, "Medium 🟠" if score >= 4, "Low 🟢" otherwise
5. summary: A brief 2-3 sentence summary of the complaint
6. urgency_reasons: List of 2-3 key reasons why this priority was assigned

Complaint text: "{text}"

Return ONLY valid JSON in this exact format:
{{
    "category": "...",
    "department": "...",
    "priority_score": <number>,
    "priority_level": "...",
    "summary": "...",
    "urgency_reasons": ["...", "...", "..."]
}}"""

        response = mistral_client.chat.complete(
            model="mistral-medium-latest",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Clean response text (remove markdown code blocks if present)
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        result = json.loads(response_text)
        return result
        
    except json.JSONDecodeError as e:
        response_text = response.choices[0].message.content.strip() if 'response' in locals() else "N/A"
        st.error(f"JSON parsing error: {str(e)}. Response was: {response_text[:200] if response_text != 'N/A' else 'N/A'}")
        return analyze_complaint_fallback(text)
    except Exception as e:
        st.error(f"AI Analysis Error: {str(e)}")
        return analyze_complaint_fallback(text)


def analyze_complaint_fallback(text):
    """Fallback rule-based analysis when AI is unavailable"""
    text_lower = text.lower()
    
    # Category detection
    if any(word in text_lower for word in ["road", "pothole", "bridge", "street", "infrastructure"]):
        category = "Civic Infrastructure"
        department = "Public Works Department"
    elif any(word in text_lower for word in ["garbage", "waste", "sanitation", "dirty", "clean"]):
        category = "Sanitation"
        department = "Municipal Sanitation Dept"
    elif any(word in text_lower for word in ["crime", "unsafe", "security", "theft", "violence"]):
        category = "Public Safety"
        department = "Police Department"
    elif any(word in text_lower for word in ["hospital", "medicine", "health", "doctor", "medical"]):
        category = "Healthcare"
        department = "Health Department"
    elif any(word in text_lower for word in ["school", "education", "teacher", "student"]):
        category = "Education"
        department = "Education Department"
    elif any(word in text_lower for word in ["electricity", "power", "light", "current"]):
        category = "Utilities"
        department = "Electricity Board"
    elif any(word in text_lower for word in ["water", "supply", "pipeline"]):
        category = "Utilities"
        department = "Water Supply Dept"
    else:
        category = "Administrative Delays"
        department = "District Administration"
    
    # Priority detection
    urgent_words = ["urgent", "emergency", "danger", "critical", "immediate", "asap"]
    if any(word in text_lower for word in urgent_words):
        priority_score = 9
        priority_level = "High 🔴"
    elif len(text) > 150:
        priority_score = 6
        priority_level = "Medium 🟠"
    else:
        priority_score = 3
        priority_level = "Low 🟢"
    
    return {
        "category": category,
        "department": department,
        "priority_score": priority_score,
        "priority_level": priority_level,
        "summary": text[:200] + "..." if len(text) > 200 else text,
        "urgency_reasons": ["Based on keyword analysis", "Length of complaint considered"]
    }


def analyze_image_with_ai(image):
    """Analyze uploaded image - Note: Mistral AI doesn't support vision API yet"""
    # Mistral AI doesn't have vision capabilities like Gemini
    # For now, we'll return a message asking user to describe the image
    # In production, you could integrate a separate vision API or use OCR
    return "Image uploaded. Please describe the issue in the image for AI analysis. Visual analysis will be added in future updates."

# ----------------------------
# SIDEBAR NAVIGATION
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

# ----------------------------
# GRIEVANCE INPUT SECTION
# ----------------------------
if page == "🏠 Submit Grievance":
    st.markdown("## 📝 Submit a Grievance")

    with st.container(border=True):
        st.markdown(
            "#### Choose how you want to submit your complaint"
        )

        tab1, tab2, tab3 = st.tabs(
            ["✍️ Text Complaint", "📷 Photo Evidence", "🎤 Voice Complaint"]
        )

        grievance_text = ""
        image_file = None

        # ---- TEXT ----
        with tab1:
            grievance_text = st.text_area(
                "Describe the issue",
                height=160,
                placeholder="Example: Garbage has not been collected for 3 days in my area..."
            )

        # ---- PHOTO ----
        with tab2:
            image_file = st.file_uploader(
                "Upload a photo (JPG / PNG)",
                type=["jpg", "jpeg", "png"]
            )
            if image_file:
                img = Image.open(image_file)
                st.image(
                    img,
                    caption="Uploaded Photo Evidence",
                    use_column_width=True
                )
                # Note: Mistral AI doesn't support vision, so we'll ask for description
                st.info("📸 Please describe the issue shown in the image below for AI analysis.")
                image_description = st.text_area(
                    "Describe what you see in the image",
                    height=100,
                    placeholder="Example: Large pothole on Main Street, about 2 feet wide..."
                )
                if image_description:
                    grievance_text = f"Image complaint: {image_description}"

        # ---- AUDIO ----
        with tab3:
            audio_file = st.file_uploader(
                "Upload a voice complaint (MP3 / WAV)",
                type=["mp3", "wav"]
            )
            if audio_file:
                st.audio(audio_file)
                grievance_text = "Complaint submitted using voice input"

    # ----------------------------
    # CENTERED SUBMIT BUTTON (NO CSS)
    # ----------------------------
    st.markdown("<br>", unsafe_allow_html=True)

    left, center, right = st.columns([3, 2, 3])
    with center:
        submit = st.button("🚀 Submit Grievance", use_container_width=True)

    # ----------------------------
    # AI OUTPUT SECTION
    # ----------------------------
    if submit:
        if grievance_text.strip() == "":
            st.error("⚠️ Please submit a grievance using text, photo, or audio.")
        else:
            with st.spinner("🤖 AI is analyzing your complaint..."):
                analysis = analyze_complaint_with_ai(grievance_text, image_file)

            st.success("✅ Grievance Submitted Successfully")

            st.markdown("## 🔍 AI Analysis & Routing")

            with st.container(border=True):
                c1, c2, c3, c4 = st.columns(4)

                c1.metric("📂 Category", analysis['category'])
                c2.metric("⚠️ Priority Level", analysis['priority_level'])
                c3.metric("🏢 Routed Department", analysis['department'])
                c4.metric("📊 Priority Score", f"{analysis['priority_score']}/10")

                st.markdown("### 📄 Complaint Summary")
                st.write(analysis['summary'])

                st.markdown("### 🔍 Full Complaint Text")
                st.write(grievance_text)

                st.markdown("### ⚡ Urgency Analysis")
                for reason in analysis['urgency_reasons']:
                    st.write(f"• {reason}")

                st.markdown("### 🕒 System Status")
                status_info = {
                    "Status": "Open",
                    "Submitted At": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Complaint ID": f"GR-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
                    "AI Confidence": "High" if mistral_client else "Fallback Mode"
                }
                st.json(status_info)

            # Store complaint in session state
            complaint_record = {
                "id": status_info["Complaint ID"],
                "text": grievance_text,
                "category": analysis['category'],
                "department": analysis['department'],
                "priority_score": analysis['priority_score'],
                "priority_level": analysis['priority_level'],
                "summary": analysis['summary'],
                "submitted_at": status_info["Submitted At"],
                "status": "Open"
            }
            st.session_state.complaints.append(complaint_record)

# ----------------------------
# ADMIN DASHBOARD
# ----------------------------
elif page == "📊 Admin Dashboard":
    st.markdown("## 📊 Admin Dashboard - Grievance Analytics")
    
    if len(st.session_state.complaints) == 0:
        st.info("📭 No complaints submitted yet. Submit a grievance to see analytics.")
    else:
        df = pd.DataFrame(st.session_state.complaints)
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Complaints", len(df))
        col2.metric("High Priority", len(df[df['priority_score'] >= 7]))
        col3.metric("Open Cases", len(df[df['status'] == 'Open']))
        col4.metric("Avg Priority Score", f"{df['priority_score'].mean():.1f}/10")
        
        st.divider()
        
        # Charts
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("### 📂 Complaints by Category")
            category_counts = df['category'].value_counts()
            fig1, ax1 = plt.subplots()
            ax1.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%', startangle=90)
            ax1.axis('equal')
            st.pyplot(fig1)
        
        with col_chart2:
            st.markdown("### ⚠️ Priority Distribution")
            priority_counts = df['priority_level'].value_counts()
            fig2, ax2 = plt.subplots()
            ax2.bar(priority_counts.index, priority_counts.values, color=['#ff4444', '#ff8800', '#44ff44'])
            ax2.set_ylabel('Number of Complaints')
            plt.xticks(rotation=45)
            st.pyplot(fig2)
        
        st.divider()
        
        # Department-wise breakdown
        st.markdown("### 🏢 Department-wise Breakdown")
        dept_counts = df['department'].value_counts()
        st.bar_chart(dept_counts)
        
        st.divider()
        
        # Recent Complaints Table
        st.markdown("### 📋 Recent Complaints")
        display_df = df[['id', 'category', 'department', 'priority_level', 'priority_score', 'submitted_at', 'status']].copy()
        display_df = display_df.sort_values('submitted_at', ascending=False)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Export option
        if st.button("📥 Export Data as CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"grievances_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

# ----------------------------
# FOOTER
# ----------------------------
st.divider()
st.caption(
    "🏆 Hackathon Prototype | AI for Good Governance | Built with Streamlit"
)
