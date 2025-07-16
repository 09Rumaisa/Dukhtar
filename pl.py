from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_community.tools import TavilySearchResults
from datetime import datetime, timedelta
import os
import json
import logging
from typing import Dict, Any, Optional, List
import requests
from dataclasses import dataclass
import sqlite3
from werkzeug.security import generate_password_hash
from dotenv import load_dotenv   
# Load environment variables
load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PregnancyData:
    """Data class for pregnancy information"""
    week: int
    trimester: int
    baby_size: str
    weight_gain: str
    symptoms: List[str]
    diet_tips: List[str]
    exercise_tips: List[str]
    warning_signs: List[str]

class PregnancyTracker:
    """Enhanced pregnancy tracker with comprehensive features"""
    
    def __init__(self):
        self.tavily_tool = TavilySearchResults(
            k=5, 
            tavily_api_key=os.getenv("TAVILY_API_KEY")
        )
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=self.openai_api_key
        )
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            openai_api_key=self.openai_api_key
        )
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for user tracking"""
        conn = sqlite3.connect('Dukhtar.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_pregnancy_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                last_period_date TEXT NOT NULL,
                due_date TEXT NOT NULL,
                current_week INTEGER NOT NULL,
                current_weight REAL,
                pre_pregnancy_weight REAL,
                height REAL,
                age INTEGER,
                language_preference TEXT DEFAULT 'english',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS weekly_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                week INTEGER NOT NULL,
                weight REAL,
                symptoms TEXT,
                notes TEXT,
                mood_score INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def calculate_pregnancy_week(self, last_period_date: str) -> int:
        """Calculate current pregnancy week based on last period date"""
        try:
            lmp = datetime.strptime(last_period_date, '%Y-%m-%d')
            today = datetime.now()
            days_pregnant = (today - lmp).days
            weeks_pregnant = days_pregnant // 7
            return max(1, min(weeks_pregnant, 42))  # Cap between 1-42 weeks
        except ValueError:
            logger.error(f"Invalid date format: {last_period_date}")
            return 1
    
    def get_trimester(self, week: int) -> int:
        """Get trimester based on week"""
        if week <= 12:
            return 1
        elif week <= 27:
            return 2
        else:
            return 3
    
    def create_vectorstore(self, urls: List[str]) -> Any:
        """Create vector store from multiple pregnancy-related URLs"""
        try:
            all_documents = []
            
            for url in urls:
                try:
                    loader = WebBaseLoader(url)
                    documents = loader.load()
                    all_documents.extend(documents)
                except Exception as e:
                    logger.warning(f"Failed to load {url}: {e}")
                    continue
            
            if not all_documents:
                raise ValueError("No documents loaded from any URL")
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            
            splits = text_splitter.split_documents(all_documents)
            
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory="./pregnancy_db"
            )
            
            return vectorstore
            
        except Exception as e:
            logger.error(f"Error creating vectorstore: {e}")
            raise
    
    def get_comprehensive_pregnancy_info(self, week: int, user_data: Dict) -> Dict[str, Any]:
        """Get comprehensive pregnancy information for specific week"""
        try:
            # Multiple reliable pregnancy information sources
            urls = [
                "https://www.whattoexpect.com/pregnancy/week-by-week/",
                "https://www.babycenter.com/pregnancy/week-by-week",
                "https://www.mayoclinic.org/healthy-lifestyle/pregnancy-week-by-week/basics/healthy-pregnancy/hlv-20049471"
            ]
            
            vectorstore = self.create_vectorstore(urls)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
            
            # Enhanced prompt template
            prompt_template = ChatPromptTemplate.from_template("""
            You are an expert pregnancy advisor for the Dukhtar app, specializing in Pakistani women's health.
            
            User Information:
            - Current Week: {week}
            - Trimester: {trimester}
            - Current Weight: {current_weight} kg
            - Pre-pregnancy Weight: {pre_pregnancy_weight} kg
            - Age: {age}
            - Language: {language}
            
            Context from reliable sources:
            {context}
            
            Please provide a comprehensive pregnancy guide for week {week} including:
            
            1. **Baby Development**: What's happening with the baby this week
            2. **Physical Changes**: What the mother might experience
            3. **Diet Plan**: Specific foods to eat and avoid (consider Pakistani cuisine)
            4. **Exercise Recommendations**: Safe exercises for this trimester
            5. **Weight Assessment**: Is the weight gain normal? (BMI considerations)
            6. **Symptoms to Expect**: Common symptoms this week
            7. **Warning Signs**: When to contact a doctor immediately
            8. **Cultural Considerations**: Tips relevant to Pakistani women
            9. **Mental Health**: Emotional support and tips
            
            Please respond in {language}. If Urdu is requested, provide accurate Urdu translations.
            Make the response informative, supportive, and culturally sensitive.
            
            Format as a well-structured article with clear sections.
            """)
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt_template}
            )
            
            # Prepare query
            trimester = self.get_trimester(week)
            query = f"Week {week} pregnancy information, trimester {trimester}, baby development, symptoms, diet, exercise"
            
            # Get AI response
            result = qa_chain.invoke({
                "query": query,
                "week": week,
                "trimester": trimester,
                "current_weight": user_data.get('current_weight', 'Not provided'),
                "pre_pregnancy_weight": user_data.get('pre_pregnancy_weight', 'Not provided'),
                "age": user_data.get('age', 'Not provided'),
                "language": user_data.get('language_preference', 'english'),
                "context": "{context}"
            })
            
            # Get additional real-time information
            additional_info = self.get_realtime_pregnancy_info(week, user_data.get('language_preference', 'english'))
            
            return {
                'week': week,
                'trimester': trimester,
                'main_content': result['result'],
                'additional_info': additional_info,
                'sources': [doc.metadata.get('source', 'Unknown') for doc in result.get('source_documents', [])],
                'weight_assessment': self.assess_weight_gain(user_data),
                'next_appointment_reminder': self.get_appointment_reminder(week),
                'emergency_contacts': self.get_emergency_contacts()
            }
            
        except Exception as e:
            logger.error(f"Error getting pregnancy info: {e}")
            return self.get_fallback_info(week, user_data)
    
    def get_realtime_pregnancy_info(self, week: int, language: str) -> Dict[str, Any]:
        """Get real-time pregnancy information using Tavily search"""
        try:
            search_queries = [
                f"pregnancy week {week} symptoms diet exercise 2024",
                f"week {week} pregnancy development baby size",
                f"pregnancy week {week} weight gain normal range"
            ]
            
            all_results = []
            for query in search_queries:
                try:
                    results = self.tavily_tool.run(query)
                    all_results.extend(results)
                except Exception as e:
                    logger.warning(f"Tavily search failed for query '{query}': {e}")
            
            return {
                'recent_research': all_results[:5],  # Top 5 most relevant results
                'search_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in realtime search: {e}")
            return {'recent_research': [], 'search_timestamp': datetime.now().isoformat()}
    
    def assess_weight_gain(self, user_data: Dict) -> Dict[str, Any]:
        """Assess if weight gain is within normal range"""
        try:
            current_weight = user_data.get('current_weight')
            pre_pregnancy_weight = user_data.get('pre_pregnancy_weight')
            height = user_data.get('height')  # in meters
            week = user_data.get('current_week', 1)
            
            if not all([current_weight, pre_pregnancy_weight, height]):
                return {'status': 'insufficient_data', 'message': 'Please provide complete weight and height information'}
            
            # Calculate BMI
            bmi = pre_pregnancy_weight / (height ** 2)
            weight_gained = current_weight - pre_pregnancy_weight
            
            # Weight gain recommendations based on BMI
            if bmi < 18.5:  # Underweight
                recommended_total = (12.5, 18.0)
            elif 18.5 <= bmi < 25:  # Normal weight
                recommended_total = (11.5, 16.0)
            elif 25 <= bmi < 30:  # Overweight
                recommended_total = (7.0, 11.5)
            else:  # Obese
                recommended_total = (5.0, 9.0)
            
            # Expected weight gain by week
            expected_gain = self.calculate_expected_weight_gain(week, recommended_total)
            
            status = 'normal'
            if weight_gained < expected_gain[0]:
                status = 'below_normal'
            elif weight_gained > expected_gain[1]:
                status = 'above_normal'
            
            return {
                'status': status,
                'current_gain': weight_gained,
                'expected_range': expected_gain,
                'total_recommended': recommended_total,
                'bmi': round(bmi, 1),
                'recommendations': self.get_weight_recommendations(status)
            }
            
        except Exception as e:
            logger.error(f"Error assessing weight gain: {e}")
            return {'status': 'error', 'message': 'Unable to assess weight gain'}
    
    def calculate_expected_weight_gain(self, week: int, total_range: tuple) -> tuple:
        """Calculate expected weight gain for current week"""
        if week <= 12:
            # First trimester: 1-4 lbs total
            factor = week / 12
            return (1 * factor, 4 * factor)
        else:
            # Second and third trimester: steady gain
            first_trimester_gain = 2.5  # Average
            remaining_weeks = week - 12
            weekly_gain = (total_range[0] - first_trimester_gain) / 28, (total_range[1] - first_trimester_gain) / 28
            return (
                first_trimester_gain + (weekly_gain[0] * remaining_weeks),
                first_trimester_gain + (weekly_gain[1] * remaining_weeks)
            )
    
    def get_weight_recommendations(self, status: str) -> List[str]:
        """Get weight gain recommendations based on status"""
        recommendations = {
            'normal': [
                "Your weight gain is within the healthy range",
                "Continue with balanced nutrition and regular exercise",
                "Monitor your weight weekly"
            ],
            'below_normal': [
                "Consider increasing caloric intake with nutritious foods",
                "Add healthy snacks between meals",
                "Consult your doctor if concerned",
                "Include protein-rich foods in every meal"
            ],
            'above_normal': [
                "Focus on portion control and balanced meals",
                "Limit processed and sugary foods",
                "Increase physical activity as recommended by your doctor",
                "Stay hydrated and eat plenty of vegetables"
            ]
        }
        return recommendations.get(status, [])
    
    def get_appointment_reminder(self, week: int) -> Dict[str, Any]:
        """Get appointment reminders based on pregnancy week"""
        appointments = {
            8: "First prenatal visit - confirm pregnancy, blood tests",
            12: "NT scan and blood tests for genetic screening",
            16: "Second trimester checkup, possible anatomy scan",
            20: "Detailed anatomy scan (20-week scan)",
            24: "Glucose screening test",
            28: "Third trimester begins - more frequent checkups",
            32: "Growth scan and presentation check",
            36: "Group B strep test, weekly checkups begin",
            40: "Due date - daily monitoring may be needed"
        }
        
        upcoming = []
        for app_week, description in appointments.items():
            if week <= app_week <= week + 2:
                upcoming.append({'week': app_week, 'description': description})
        
        return {
            'upcoming_appointments': upcoming,
            'next_routine_checkup': self.calculate_next_checkup(week)
        }
    
    def calculate_next_checkup(self, week: int) -> str:
        """Calculate when next routine checkup should be"""
        if week < 28:
            return "Every 4 weeks"
        elif week < 36:
            return "Every 2 weeks"
        else:
            return "Every week"
    
    def get_emergency_contacts(self) -> Dict[str, Any]:
        """Get emergency contact information"""
        return {
            'emergency_number': '115',  # Pakistan emergency number
            'pregnancy_helpline': '021-111-911-911',  # Example helpline
            'warning_signs': [
                'Severe abdominal pain',
                'Heavy bleeding',
                'Persistent vomiting',
                'Severe headache with vision changes',
                'Decreased fetal movement',
                'Signs of preterm labor'
            ]
        }
    
    def get_fallback_info(self, week: int, user_data: Dict) -> Dict[str, Any]:
        """Provide fallback information when main sources fail"""
        basic_info = {
            'week': week,
            'trimester': self.get_trimester(week),
            'main_content': f"""
            # Pregnancy Week {week} - Basic Information
            
            ## Baby Development
            Your baby is continuing to grow and develop. At week {week}, important developmental milestones are occurring.
            
            ## For You
            - Continue taking prenatal vitamins
            - Eat a balanced diet with plenty of fruits and vegetables
            - Stay hydrated
            - Get adequate rest
            - Gentle exercise as approved by your doctor
            
            ## Warning Signs
            Contact your healthcare provider if you experience:
            - Severe abdominal pain
            - Heavy bleeding
            - Persistent vomiting
            - Severe headache
            
            *Note: This is basic information. Please consult your healthcare provider for personalized advice.*
            """,
            'additional_info': {'recent_research': [], 'search_timestamp': datetime.now().isoformat()},
            'sources': ['Built-in knowledge base'],
            'weight_assessment': {'status': 'insufficient_data', 'message': 'Please provide complete information for weight assessment'},
            'next_appointment_reminder': self.get_appointment_reminder(week),
            'emergency_contacts': self.get_emergency_contacts()
        }
        
        return basic_info
    
    def save_user_data(self, user_id: int, data: Dict) -> bool:
        """Save user pregnancy data to database"""
        try:
            conn = sqlite3.connect('pregnancy_tracker.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO user_pregnancy_data 
                (user_id, last_period_date, due_date, current_week, current_weight, 
                 pre_pregnancy_weight, height, age, language_preference, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                data.get('last_period_date'),
                data.get('due_date'),
                data.get('current_week'),
                data.get('current_weight'),
                data.get('pre_pregnancy_weight'),
                data.get('height'),
                data.get('age'),
                data.get('language_preference', 'english'),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error saving user data: {e}")
            return False
    
    def get_user_data(self, user_id: int) -> Optional[Dict]:
        """Get user pregnancy data from database"""
        try:
            conn = sqlite3.connect('pregnancy_tracker.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM user_pregnancy_data WHERE user_id = ?
                ORDER BY updated_at DESC LIMIT 1
            ''', (user_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, row))
            return None
            
        except Exception as e:
            logger.error(f"Error getting user data: {e}")
            return None

# Flask routes
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')
tracker = PregnancyTracker()

@app.route('/pregnancy_tracker', methods=['GET', 'POST'])
def pregnancy_tracker():
    """Enhanced pregnancy tracker endpoint"""
    try:
        if request.method == 'POST':
            # Get user data from form or JSON
            if request.is_json:
                data = request.get_json()
            else:
                data = request.form.to_dict()
            
            user_id = data.get('user_id') or session.get('user_id')
            if not user_id:
                return jsonify({'error': 'User ID required'}), 400
            
            # Calculate current week if last period date is provided
            if 'last_period_date' in data:
                current_week = tracker.calculate_pregnancy_week(data['last_period_date'])
                data['current_week'] = current_week
                
                # Calculate due date
                lmp = datetime.strptime(data['last_period_date'], '%Y-%m-%d')
                due_date = lmp + timedelta(days=280)  # 40 weeks
                data['due_date'] = due_date.strftime('%Y-%m-%d')
            
            # Save user data
            tracker.save_user_data(user_id, data)
            
            # Get comprehensive pregnancy information
            pregnancy_info = tracker.get_comprehensive_pregnancy_info(
                data.get('current_week', 1), 
                data
            )
            
            if request.is_json:
                return jsonify({
                    'success': True,
                    'data': pregnancy_info
                })
            else:
                return render_template('pregnancy_tracker.html', 
                                     pregnancy_info=pregnancy_info, 
                                     user_data=data)
        
        else:  # GET request
            user_id = request.args.get('user_id') or session.get('user_id')
            if not user_id:
                return render_template('pregnancy_setup.html')
            
            # Get existing user data
            user_data = tracker.get_user_data(user_id)
            if not user_data:
                return render_template('pregnancy_setup.html')
            
            # Update current week based on last period date
            current_week = tracker.calculate_pregnancy_week(user_data['last_period_date'])
            user_data['current_week'] = current_week
            
            # Get pregnancy information
            pregnancy_info = tracker.get_comprehensive_pregnancy_info(current_week, user_data)
            
            return render_template('pregnancy_tracker.html', 
                                 pregnancy_info=pregnancy_info, 
                                 user_data=user_data)
    
    except Exception as e:
        logger.error(f"Error in pregnancy tracker: {e}")
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500

@app.route('/pregnancy_tracker/log_weekly', methods=['POST'])
def log_weekly_data():
    """Log weekly pregnancy data"""
    try:
        data = request.get_json()
        user_id = data.get('user_id') or session.get('user_id')
        
        if not user_id:
            return jsonify({'error': 'User ID required'}), 400
        
        conn = sqlite3.connect('pregnancy_tracker.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO weekly_logs (user_id, week, weight, symptoms, notes, mood_score)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            data.get('week'),
            data.get('weight'),
            json.dumps(data.get('symptoms', [])),
            data.get('notes'),
            data.get('mood_score')
        ))
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Weekly data logged successfully'})
        
    except Exception as e:
        logger.error(f"Error logging weekly data: {e}")
        return jsonify({'error': 'Failed to log data'}), 500

@app.route('/pregnancy_tracker/history/<int:user_id>')
def get_pregnancy_history(user_id):
    """Get pregnancy tracking history"""
    try:
        conn = sqlite3.connect('pregnancy_tracker.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM weekly_logs WHERE user_id = ? ORDER BY week ASC
        ''', (user_id,))
        
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        history = []
        for row in rows:
            log_data = dict(zip(columns, row))
            if log_data['symptoms']:
                log_data['symptoms'] = json.loads(log_data['symptoms'])
            history.append(log_data)
        
        conn.close()
        
        return jsonify({'success': True, 'history': history})
        
    except Exception as e:
        logger.error(f"Error getting pregnancy history: {e}")
        return jsonify({'error': 'Failed to get history'}), 500

if __name__ == '__main__':
    app.run(debug=True)