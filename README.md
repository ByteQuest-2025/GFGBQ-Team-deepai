# Problem Statement
PS 12: AI for Grievance Redressal in Public Governance

# Project Name
CivicFix AI - Intelligent Citizen Grievance Portal

# Team Name
Deep AI

# Deployed Link
(To be added)

# Demo Video Link
https://drive.google.com/file/d/1yq18K0etuZcgB6P6aepg-Lx0zPywrs4L/view?usp=sharing

# PPT Link
https://1drv.ms/p/c/242A04F9105323F0/IQA1o9HTKexYTo2PhOf-X-xoAfO4esJGziax2CZSWzHtQz8?e=OtJXZq

---

## Project Overview
CivicFix AI is a smart governance platform that uses Generative AI to automatically categorize, prioritize, and route citizen complaints. It eliminates manual triage bottlenecks and ensures critical public issues are resolved faster.

## Features
- **Auto-Categorization:** AI detects if a complaint is about Roads, Sanitation, etc.
- **Urgency Scoring:** AI rates complaints (1-10) to highlight emergencies.
- **Admin Dashboard:** A visual view of city-wide issues.

## Tech Stack
- **Frontend/Backend:** Python (Streamlit)
- **AI Engine:** Mistral AI API (mistral-medium-latest), Groq AI
- **Data Analysis:** Pandas, Matplotlib
- **Environment:** python-dotenv

## Setup Instructions

1. **Clone the repository**
   ```git clone https://github.com/ByteQuest-2025/GFGBQ-Team-deepai.git
   ```

2. **Create a virtual environment** (if not already created)
   ``` python -m venv venv
   ```

3. **Activate virtual environment**
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up API Key**
   - Get your Mistral AI API key from: https://console.mistral.ai
   - Sign up/login and navigate to API keys section
   - Create a new API key
   - Create a `.env` file in the root directory
   - Add: `MISTRAL_API_KEY=your_api_key_here`

   -Get your Groq AI API key from - https://console.groq.com/keys
   -Create a new API key
   -In .env file, add 'GROQ_API_KEY=your_api_key_here'

6. **Run the application**
   ```
   streamlit run app.py
   ```

## Features Implemented

 **AI-Powered Classification**: Uses Mistral AI to intelligently categorize complaints into 8+ categories
 **Smart Prioritization**: AI assigns priority scores (1-10) based on urgency, safety, and impact
 **Image Support**: Upload photos with description for complaint documentation
 **Auto-Routing**: Automatically routes complaints to appropriate government departments
 **Admin Dashboard**: Real-time analytics with charts showing complaint distribution, priority levels, and department-wise breakdown
 **Multi-Modal Input**: Supports text, image, and voice complaint submissions
 **Multi language Input**: Supports all languages
 **Complaint Tracking**: Stores and tracks all complaints with unique IDs

## How It Works

1. **Submit Grievance**: Citizens can submit complaints via text, photo, or voice
2. **AI Analysis**: Mistral AI analyzes the complaint to:
   - Categorize the issue (Infrastructure, Sanitation, Safety, etc.)
   - Assign priority score (1-10) and level (High/Medium/Low)
   - Route to appropriate department
   - Generate summary and urgency reasons
3. **Admin Dashboard**: Government officials can view:
   - Total complaints and high-priority cases
   - Category-wise distribution (pie chart)
   - Priority level distribution (bar chart)
   - Department-wise breakdown
   - Recent complaints table with export functionality

## Categories Supported

- Civic Infrastructure
- Sanitation
- Public Safety
- Healthcare
- Education
- Utilities (Electricity, Water)
- Administrative Delays
- Other
