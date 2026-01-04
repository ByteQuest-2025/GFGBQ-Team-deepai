# AI-Powered Citizen Grievance Analysis System

## Project Name

CivicFix AI - Intelligent Citizen Grievance Portal

## Team Name

Deep AI

## Deployed Link

https://civicfix-deepai.streamlit.app/

## Demo Video Link

https://drive.google.com/file/d/1yq18K0etuZcgB6P6aepg-Lx0zPywrs4L/view?usp=sharing

## PPT Link

https://1drv.ms/p/c/242A04F9105323F0/IQA1o9HTKexYTo2PhOf-X-xoAfO4esJGziax2CZSWzHtQz8?e=OtJXZq

---

## 📌 Problem Statement

PS-12 : AI-Powered Citizen Grievance Analysis System

Public governance bodies receive a large number of citizen grievances every day in the form of unstructured text and voice complaints. Manual review and routing of these grievances is time-consuming, leading to delays, backlogs, and inefficient handling of critical issues. There is a need for an intelligent system that can automatically understand and organize citizen complaints.

## 🎯 Objective

To build an AI-driven platform that analyzes citizen grievances and automatically:

- Classifies complaints into relevant categories
- Generates concise summaries
- Identifies the priority level of each grievance

## 💡 Solution Overview

This project provides a web-based system where users can submit grievances through text or voice. Voice inputs are converted into text and processed using NLP techniques. The system returns structured insights such as complaint category, summary, and urgency, assisting faster and more transparent grievance handling.

## Project Overview

CivicFix AI is a smart governance platform that uses Generative AI to automatically categorize, prioritize, and route citizen complaints. It eliminates manual triage bottlenecks and ensures critical public issues are resolved faster.

## Features

- **Auto-Categorization:** AI detects if a complaint is about Roads, Sanitation, etc.
- **Urgency Scoring:** AI rates complaints (1-10) to highlight emergencies.
- **Admin Dashboard:** A visual view of city-wide issues.

## Tech Stack

- **Frontend/Backend:** Python (Streamlit)
- **AI Engine:** Mistral AI API (mistral-medium-latest) and Groq AI
- **Data Analysis:** Pandas, Matplotlib
- **Environment:** python-dotenv

## Setup Instructions

1.**Clone the repository**

```bash
git clone https://github.com/ByteQuest-2025/GFGBQ-Team-deepai
```

2.**Change the Directory**

```bash
cd GFGBQ-Team-deepai
```

3.**Create a virtual environment** (if not already created)

```bash
python -m venv venv
```

4.**Activate virtual environment**

- Windows: `venv\Scripts\activate`
- Linux/Mac: `source venv/bin/activate`

5.**Install dependencies**

```bash
pip install -r requirements.txt
```

6.**Set up API Key**

Get three FREE API keys:

a.Mistral AI (Required)

Visit: [Mistral](https://console.mistral.ai)
Create account → API keys → Create new key
Features: Text analysis, image analysis, language detection

b.Groq API (Required for audio)

Visit: [Groq](https://console.groq.com/keys)
Sign up (free, no credit card) → Create API Key
Features: Ultra-fast audio transcription (5-10 seconds)


6.**Create .env file in root**

```bash
MISTRAL_API_KEY=your_mistral_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

7.**Run the application**

```bash
streamlit run app.py
```

Opens at [http://localhost:8501](http://localhost:8501)

## Features Implemented

- ✅ **AI-Powered Classification**: Uses Mistral AI to intelligently categorize complaints into 8+ categories
- ✅ **Smart Prioritization**: AI assigns priority scores (1-10) based on urgency, safety, and impact
- ✅ **Image Support**: Upload photos with description for complaint documentation
- ✅ **Auto-Routing**: Automatically routes complaints to appropriate government departments
- ✅ **Admin Dashboard**: Real-time analytics with charts showing complaint distribution, priority levels, and department-wise breakdown
- ✅ **Multi-Modal Input**: Supports text, image, and voice complaint submissions
- ✅ **Complaint Tracking**: Stores and tracks all complaints with unique IDs

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

## 📃Categories Supported

- Civic Infrastructure
- Sanitation
- Public Safety
- Healthcare
- Education
- Utilities (Electricity, Water)
- Administrative Delays
- Other

---

## ⚠️ Disclaimer

This system is designed to assist grievance understanding and improve processing efficiency.
It does not replace human decision-making or take administrative actions independently.

---

THANK YOU
