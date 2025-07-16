# app.py
from db import get_connection
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import psycopg2
import psycopg2.extras
import uuid
from datetime import datetime, timedelta
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Import the DukhtarAgent from main.py
from main import DukhtarAgent
from main import DUKHTAR_SYSTEM_PROMPT
from psycopg2.extras import RealDictCursor



app = Flask(__name__,static_url_path='/static', static_folder='C:\\Users\\rumai\\OneDrive\\Desktop\\Dukhtar\\static')
app.secret_key = 'your_secret_key'  # Used for session management

# Global agent instance
dukhtar_agent = DukhtarAgent()

# ========================================
# AUTHENTICATION DECORATOR
# ========================================

def login_required(f):
    """Decorator to require login for certain routes."""
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this feature.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

# ========================================
# AUTHENTICATION ROUTES
# ========================================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        phone = request.form['phone']
        children = request.form['children']
        age_of_last_child = request.form.get('age_of_last_child') or None
        current_weight = request.form.get('current_weight') or None
        diseases = request.form['diseases']
        info = request.form['info']

        hashed_password = generate_password_hash(password)

        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO users (name, email, password_hash, phone, number_of_children, age_of_last_child, current_weight, ongoing_diseases, additional_info)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (name, email, hashed_password, phone, children, age_of_last_child, current_weight, diseases, info))
            conn.commit()
            cur.close()
            conn.close()
            flash('Signup successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except psycopg2.Error as e:
            if 'conn' in locals():
                conn.rollback()
                conn.close()
            flash(f'Error: {e}', 'danger')

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
     if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute("SELECT user_id, name, password_hash FROM users WHERE email = %s", (email,))
            user = cur.fetchone()
            cur.close()
            conn.close()

            if user and check_password_hash(user[2], password):
                session['user_id'] = user[0]
                session['name'] = user[1]
                flash('Logged in successfully!', 'success')
                return redirect(url_for('home'))
            else:
                flash('Invalid credentials. Please try again.', 'danger')
        except psycopg2.Error as e:
            flash(f'Database error: {e}', 'danger')

     return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out.', 'info')
    return redirect(url_for('home'))

# ========================================
# AI CHAT ROUTES (REQUIRE LOGIN)
# ========================================

@app.route('/chat')
@login_required
def chat_page():
    """Chat page - requires login."""
    return render_template('test.html')

@app.route('/api/chat', methods=['POST'])
@login_required
def chat():
    """Text chat API - requires login."""
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({"error": "No message provided"}), 400
        
        response = dukhtar_agent.process_text(message)
        return jsonify({"response": response})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/process_voice', methods=['POST'])
@login_required
def process_voice():
    """Voice processing API - requires login."""
    try:
        audio_base64 = request.form.get('audio_base64')
        
        if not audio_base64:
            return jsonify({"error": "No audio data provided"}), 400
        
        print("Processing voice input...")
        result = dukhtar_agent.process_audio(audio_base64)
        
        print(f"Transcription: {result.get('transcription', 'No transcription')}")
        print(f"Text response: {result.get('text_response', 'No text response')}")
        print(f"Audio response length: {len(result.get('audio_response', '')) if result.get('audio_response') else 'No audio response'}")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in process_voice: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze_image', methods=['POST'])
@login_required
def analyze_image():
    """Image analysis API - requires login."""
    try:
        image_base64 = request.form.get('image_base64')
        query = request.form.get('query', '')
        
        if not image_base64:
            return jsonify({"error": "No image data provided"}), 400
        
        result = dukhtar_agent.process_image(image_base64, query)
        return jsonify({"result": result})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/clear_history', methods=['POST'])
@login_required
def clear_history():
    """Clear conversation history - requires login."""
    try:
        dukhtar_agent.clear_conversation()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/api/export_history')
@login_required
def export_history():
    """Export chat history - requires login."""
    try:
        history = dukhtar_agent.get_conversation_history()
        return jsonify({"messages": history})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/pregnancy_tracker', methods=['GET', 'POST'])
def pregnancy_tracker():
    """Pregnancy tracker page - allows users to track pregnancy progress."""
    
    print(f"Request method: {request.method}")  # Debug log
    
    if request.method == 'GET':
        # Render the form for user input
        return render_template('pregnancy_tracker_form.html')
    
    # Handle POST request with user data
    print("Processing POST request...")  # Debug log
    
    try:
        # Get user inputs from form
        pregnancy_week = request.form.get('pregnancy_week', type=int)
        trimester = request.form.get('trimester', type=int)
        current_weight = request.form.get('current_weight', type=float)
        pre_pregnancy_weight = request.form.get('pre_pregnancy_weight', type=float)
        height = request.form.get('height', type=float)  # in cm
        age = request.form.get('age', type=int)
        activity_level = request.form.get('activity_level', '')
        language = request.form.get('language', 'english')
        dietary_restrictions = request.form.get('dietary_restrictions', '')
        medical_conditions = request.form.get('medical_conditions', '')
        
        print(f"Form data received: week={pregnancy_week}, trimester={trimester}, weight={current_weight}")
        print(f"Height: {height}, Age: {age}")
        
        # Enhanced validation with specific error messages
        if not pregnancy_week:
            print("ERROR: Missing pregnancy week")
            return render_template('pregnancy_tracker_form.html', 
                                 error="Please provide pregnancy week")
        
        if not trimester:
            print("ERROR: Missing trimester")
            return render_template('pregnancy_tracker_form.html', 
                                 error="Please provide trimester")
        
        if not current_weight:
            print("ERROR: Missing current weight")
            return render_template('pregnancy_tracker_form.html', 
                                 error="Please provide current weight")
        
        if not pre_pregnancy_weight:
            print("ERROR: Missing pre-pregnancy weight")
            return render_template('pregnancy_tracker_form.html', 
                                 error="Please provide pre-pregnancy weight")
        
        if not height or height <= 0:
            print("ERROR: Invalid height")
            return render_template('pregnancy_tracker_form.html', 
                                 error="Please provide valid height")
        
        print("✓ All validations passed")
        
        # Calculate BMI and weight gain
        try:
            height_m = height / 100
            pre_pregnancy_bmi = pre_pregnancy_weight / (height_m ** 2)
            weight_gain = current_weight - pre_pregnancy_weight
            print(f"✓ BMI calculated: {pre_pregnancy_bmi:.2f}")
            print(f"✓ Weight gain calculated: {weight_gain:.2f} kg")
        except Exception as e:
            print(f"ERROR in BMI calculation: {e}")
            return render_template('pregnancy_tracker_form.html', 
                                 error="Error calculating BMI. Please check your inputs.")
        
        # Test API keys before proceeding
        print("Testing API keys...")
        tavily_key = os.getenv("TAVILY_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        
        if not tavily_key:
            print("ERROR: TAVILY_API_KEY not found")
            return render_template('pregnancy_tracker_form.html', 
                                 error="API configuration error. Please contact support.")
        
        if not openai_key:
            print("ERROR: OPENAI_API_KEY not found")
            return render_template('pregnancy_tracker_form.html', 
                                 error="API configuration error. Please contact support.")
        
        print("✓ API keys found")
        
        # Initialize Tavily search tool with error handling
        try:
            tavily_tool = TavilySearchResults(k=5, tavily_api_key=tavily_key)
            print("✓ Tavily tool initialized")
        except Exception as e:
            print(f"ERROR initializing Tavily: {e}")
            return render_template('pregnancy_tracker_form.html', 
                                 error="Search service unavailable. Please try again later.")
        
        # Create search queries
        search_queries = [
            f"pregnancy week {pregnancy_week} baby development fetal growth",
            f"pregnancy trimester {trimester} diet nutrition meal plan",
            f"pregnancy week {pregnancy_week} safe exercises physical activity",
            f"pregnancy weight gain week {pregnancy_week} normal range BMI",
            f"pregnancy week {pregnancy_week} symptoms what to expect"
        ]
        
        print(f"Starting search with {len(search_queries)} queries...")
        
        # Gather information from multiple sources
        all_search_results = []
        successful_searches = 0
        
        for i, query in enumerate(search_queries):
            try:
                print(f"Executing search {i+1}/{len(search_queries)}: {query}")
                results = tavily_tool.run(query)
                if results:
                    all_search_results.extend(results)
                    successful_searches += 1
                    print(f"✓ Search {i+1} successful - got {len(results)} results")
                else:
                    print(f"⚠ Search {i+1} returned no results")
            except Exception as e:
                print(f"ERROR in search {i+1}: {e}")
                continue
        
        print(f"✓ Completed {successful_searches}/{len(search_queries)} searches successfully")
        
        # Web scraping with better error handling
        url = f"https://www.whattoexpect.com//pregnancy//week-by-week//week-{pregnancy_week}/"
        print(f"Attempting to load: {url}")
        
        web_data = []
        try:
            loader = WebBaseLoader(url)
            web_data = loader.load()
            print(f"✓ Web scraping successful - got {len(web_data)} documents")
        except Exception as e:
            print(f"⚠ Web scraping failed: {e}")
            # Continue without web data
        
        # Combine all information
        all_content = ""
        for result in all_search_results:
            all_content += f"{result}\n\n"
        
        for doc in web_data:
            all_content += f"{doc.page_content}\n\n"
        
        print(f"✓ Combined content length: {len(all_content)} characters")
        
        # Create embeddings and vector store
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            
            if all_content.strip():
                splits = text_splitter.split_text(all_content)
                print(f"✓ Text split into {len(splits)} chunks")
            else:
                splits = ["General pregnancy information for comprehensive care."]
                print("⚠ Using fallback content - no search results available")
            
            print("Initializing embeddings...")
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=openai_key,
            )
            
            print("Creating vector store...")
            vectorstore = Chroma.from_texts(
                texts=splits,
                embedding=embeddings,
                persist_directory="./pregnancy_db"
            )
            print("✓ Vector store created")
            
        except Exception as e:
            print(f"ERROR creating vector store: {e}")
            return render_template('pregnancy_tracker_form.html', 
                                 error="Error processing information. Please try again.")
        
        # Initialize LLM
        try:
            print("Initializing LLM...")
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                openai_api_key=openai_key,
                temperature=0.7
            )
            print("✓ LLM initialized")
        except Exception as e:
            print(f"ERROR initializing LLM: {e}")
            return render_template('pregnancy_tracker_form.html', 
                                 error="AI service unavailable. Please try again later.")
        
        # Create retriever and QA chain
        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
            print("✓ Retriever created")
            
            # Create personalized prompt
            user_info = f"""
            Pregnancy Week: {pregnancy_week}
            Trimester: {trimester}
            Current Weight: {current_weight} kg
            Pre-pregnancy Weight: {pre_pregnancy_weight} kg
            Height: {height} cm
            Age: {age}
            Pre-pregnancy BMI: {pre_pregnancy_bmi:.1f}
            Weight Gain So Far: {weight_gain:.1f} kg
            Activity Level: {activity_level}
            Dietary Restrictions: {dietary_restrictions}
            Medical Conditions: {medical_conditions}
            Preferred Language: {language}
            """
            
            # Determine language-specific instructions
            if language.lower() == 'urdu':
                language_instruction = "Please provide the entire response in Urdu language with proper Urdu grammar and vocabulary."
            else:
                language_instruction = "Please provide the response in English."
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": ChatPromptTemplate.from_template(f"""
You are an expert pregnancy and maternal health advisor. Based on the user's specific information and the context provided, create a comprehensive, personalized article about their pregnancy journey.

User Information:
{user_info}

Context from medical sources: {{context}}

{language_instruction}

Please create a detailed, well-structured article that includes:

1. **BABY'S DEVELOPMENT** - Detailed information about fetal development at week {pregnancy_week}
2. **WEIGHT ANALYSIS** - Assessment of current weight gain (is it normal, too much, or too little?)
3. **PERSONALIZED DIET PLAN** - Specific meal recommendations considering their restrictions
4. **SAFE EXERCISE ROUTINE** - Appropriate exercises for their activity level and pregnancy stage
5. **SYMPTOMS TO EXPECT** - Common symptoms at this stage and when to contact doctor
6. **IMPORTANT REMINDERS** - Key things to remember and upcoming milestones
7. **HEALTH TIPS** - Specific advice for their situation

Make the article engaging, informative, and reassuring. Include specific examples and practical advice.

Question: Create a comprehensive pregnancy guide for week {pregnancy_week}.

Helpful Answer:""")
                }
            )
            print("✓ QA chain created")
            
        except Exception as e:
            print(f"ERROR creating QA chain: {e}")
            return render_template('pregnancy_tracker_form.html', 
                                 error="Error setting up AI assistant. Please try again.")
        
        # Get the response
        try:
            print("Generating AI response...")
            result = qa_chain.invoke({"query": f"pregnancy week {pregnancy_week} comprehensive guide"})
            article_content = result["result"]
            print(f"✓ AI response generated - length: {len(article_content)} characters")
            
            if not article_content or len(article_content) < 100:
                print("⚠ AI response seems too short, might be an error")
                
        except Exception as e:
            print(f"ERROR generating AI response: {e}")
            return render_template('pregnancy_tracker_form.html', 
                                 error="Error generating personalized guide. Please try again.")
        
        # Clean up the vector store
        try:
            vectorstore.delete_collection()
            print("✓ Vector store cleaned up")
        except Exception as e:
            print(f"⚠ Error cleaning up vector store: {e}")
            # Continue anyway
        
        # Additional personalization based on BMI and weight gain
        if pre_pregnancy_bmi < 18.5:
            weight_status = "underweight"
            recommended_gain = "12.5-18 kg"
        elif pre_pregnancy_bmi < 25:
            weight_status = "normal weight"
            recommended_gain = "11.5-16 kg"
        elif pre_pregnancy_bmi < 30:
            weight_status = "overweight"
            recommended_gain = "7-11.5 kg"
        else:
            weight_status = "obese"
            recommended_gain = "5-9 kg"
        
        # Prepare additional context
        additional_info = {
            'pregnancy_week': pregnancy_week,
            'trimester': trimester,
            'weight_status': weight_status,
            'recommended_gain': recommended_gain,
            'current_gain': weight_gain,
            'language': language
        }
        
        print("✓ Additional info prepared")
        

        # Save to database
        try:
            conn = get_connection()
            if conn:
                cur = conn.cursor()
                
                # Get tracking_id if exists (you may need to create this logic)
                tracking_id = None
                user_id= session.get('user_id')
                if user_id:
                    cur.execute("""
                        SELECT tracking_id FROM pregnancy_tracking 
                        WHERE user_id = %s AND is_active = TRUE 
                        ORDER BY created_at DESC LIMIT 1
                    """, (user_id,))
                    tracking_result = cur.fetchone()
                    if tracking_result:
                        tracking_id = tracking_result[0]
                
                # Insert pregnancy guide data
                insert_query = """
                    INSERT INTO pregnancy_guides (
                        user_id, tracking_id, pregnancy_week, trimester, 
                        current_weight, pre_pregnancy_weight, height_cm, age,
                        pre_pregnancy_bmi, weight_gain_kg, activity_level,
                        dietary_restrictions, medical_conditions, language,
                        generated_guide, weight_status, recommended_weight_gain,
                        search_queries_used, generation_status
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING guide_id
                """
                
                cur.execute(insert_query, (
                    user_id, tracking_id, pregnancy_week, trimester,
                    current_weight, pre_pregnancy_weight, height, age,
                    pre_pregnancy_bmi, weight_gain, activity_level,
                    dietary_restrictions, medical_conditions, language,
                    article_content, weight_status, recommended_gain,
                    search_queries, 'completed'
                ))
                
                guide_id = cur.fetchone()[0]
                conn.commit()
                
                print(f"✓ Pregnancy guide saved to database with ID: {guide_id}")
        except Exception as e:
            print(f"ERROR saving to database: {e}")
        print("✓ Rendering results template...")
        return render_template('pregnancy_results.html', 
                             article=article_content, 
                             user_info=additional_info)
        
    except Exception as e:
        print(f"FATAL ERROR in pregnancy tracker: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return render_template('pregnancy_tracker_form.html', 
                             error="An unexpected error occurred. Please try again or contact support.")


@app.route('/generate_speech', methods=['POST'])
def generate_speech():
    """Generate speech audio from text using OpenAI TTS"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        voice = data.get('voice', 'nova')  # Default voice
        language = data.get('language', 'english')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Clean the text for TTS (remove HTML tags, etc.)
        import re
        cleaned_text = re.sub(r'<[^>]+>', '', text)
        cleaned_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned_text)  # Remove bold markdown
        cleaned_text = re.sub(r'\*([^*]+)\*', r'\1', cleaned_text)  # Remove italic markdown
        cleaned_text = cleaned_text.replace('\n\n', '. ')  # Replace double newlines with periods
        cleaned_text = cleaned_text.strip()
        
        # Limit text length to avoid very long audio files
        if len(cleaned_text) > 4000:
            cleaned_text = cleaned_text[:4000] + "..."
        
        # Initialize OpenAI client
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Choose voice based on language
        if language.lower() == 'urdu':
            voice = 'nova'  # Nova works well for non-English languages
        else:
            voice = voice or 'nova'  # Default to nova for English
        
        # Generate speech
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=cleaned_text
        )
        
        # Create a unique filename
        import uuid
        filename = f"speech_{uuid.uuid4().hex[:8]}.mp3"
        filepath = os.path.join('static', 'audio', filename)
        
        # Ensure audio directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the audio file
        response.stream_to_file(filepath)
        
        # Return the audio file URL
        return jsonify({
            'audio_url': f'/static/audio/{filename}',
            'message': 'Speech generated successfully'
        })
        
    except Exception as e:
        print(f"Error generating speech: {e}")
        return jsonify({'error': 'Failed to generate speech'}), 500

@app.route('/cleanup_audio', methods=['POST'])
def cleanup_audio():
    """Clean up temporary audio files"""
    try:
        data = request.get_json()
        filename = data.get('filename', '')
        
        if filename:
            filepath = os.path.join('static', 'audio', filename)
            if os.path.exists(filepath):
                os.remove(filepath)
        
        return jsonify({'message': 'Audio cleaned up successfully'})
    
    except Exception as e:
        print(f"Error cleaning up audio: {e}")
        return jsonify({'error': 'Failed to cleanup audio'}), 500

# Helper function to calculate recommended weight gain
def get_weight_gain_recommendation(bmi, week):
    """Calculate recommended weight gain based on BMI and pregnancy week"""
    if bmi < 18.5:  # Underweight
        total_recommended = 16  # Average of 12.5-18
    elif bmi < 25:  # Normal weight
        total_recommended = 14  # Average of 11.5-16
    elif bmi < 30:  # Overweight
        total_recommended = 9   # Average of 7-11.5
    else:  # Obese
        total_recommended = 7   # Average of 5-9
    
    # Calculate expected gain by this week (most gain in 2nd/3rd trimester)
    if week <= 12:
        expected_gain = total_recommended * 0.1
    elif week <= 28:
        expected_gain = total_recommended * 0.4
    else:
        expected_gain = total_recommended * (week - 12) / 28
    
    return expected_gain







# ========================================
# DOCTOR & CONSULTATION PAGES (HTML)
# ========================================



@app.route('/doctors/<int:doctor_id>',methods=['GET', 'POST'])
def doctor_detail_page(doctor_id):
    """Page to show doctor details and reviews."""
    return render_template('doctor_detail.html',doctor_id=doctor_id)

@app.route('/doctors/<int:doctor_id>/book')
@login_required
def book_consultation_page(doctor_id):
    """Page to book a consultation with a doctor."""
    return render_template('book_consultation.html',doctor_id=doctor_id)
@app.route('/doctors')
def doctors_page():
    """Page to list all doctors with filters - Initial page load."""
    try:
        specialization = request.args.get('specialization')
        min_rating = request.args.get('min_rating', type=float)
        max_fee = request.args.get('max_fee', type=float)
        
        conn = get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        query = """
            SELECT doctor_id, full_name, specialization, qualification, 
                   experience_years, consultation_fee, profile_image_url,
                   bio, languages_spoken, availability_hours, rating, 
                   total_reviews, hospital_affiliation
            FROM doctors 
            WHERE is_active = TRUE AND is_verified = TRUE
        """
        params = []
        
        if specialization:
            query += " AND specialization ILIKE %s"
            params.append(f"%{specialization}%")
        
        if min_rating:
            query += " AND rating >= %s"
            params.append(min_rating)
            
        if max_fee:
            query += " AND consultation_fee <= %s"
            params.append(max_fee)
        
        query += " ORDER BY rating DESC, total_reviews DESC"
        
        cur.execute(query, params)
        doctors = cur.fetchall()
        
        cur.close()
        conn.close()
        
        # Render HTML template with data for initial load
        return render_template('doctors.html', 
                             doctors=doctors, 
                             total=len(doctors),
                             specialization=specialization or '',
                             min_rating=min_rating or '',
                             max_fee=max_fee or '')
        
    except Exception as e:
        # Render template with error message
        return render_template('doctors.html', 
                             doctors=[], 
                             total=0,
                             error=str(e),
                             specialization=specialization or '',
                             min_rating=min_rating or '',
                             max_fee=max_fee or '')

@app.route('/consultations')
@login_required
def consultations_page():
    """Page to list user's consultations."""
    return render_template('consultations.html')

@app.route('/consultations/<int:consultation_id>/chat')
@login_required
def consultation_chat_page(consultation_id):
    """Page for chat/messages during a consultation."""
    return render_template('consultation_chat.html',consultation_id=consultation_id)


# Route to get a specific doctor's details
@app.route('/api/doctors/<int:doctor_id>', methods=['GET'])
def get_doctor_details(doctor_id):
    """Get detailed information about a specific doctor"""
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Get doctor details
        cur.execute("""
            SELECT * FROM doctors 
            WHERE doctor_id = %s AND is_active = TRUE AND is_verified = TRUE
        """, (doctor_id,))

        doctor = cur.fetchone()

        if not doctor:
            return jsonify({
                'success': False,
                'error': 'Doctor not found'
            }), 404

        # Get doctor's availability
        cur.execute("""
            SELECT day_of_week, start_time, end_time, is_available
            FROM doctor_availability
            WHERE doctor_id = %s
            ORDER BY day_of_week
        """, (doctor_id,))

        availability = cur.fetchall()

        # Get recent reviews
        cur.execute("""
            SELECT rating, review_text, created_at, is_anonymous
            FROM doctor_reviews
            WHERE doctor_id = %s
            ORDER BY created_at DESC
            LIMIT 10
        """, (doctor_id,))

        reviews = cur.fetchall()

        cur.close()
        conn.close()

        return jsonify({
            'success': True,
            'doctor': dict(doctor),
            'availability': availability,
            'reviews': reviews
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/doctors')
def api_doctors():
    """API endpoint for AJAX requests - Returns JSON."""
    try:
        specialization = request.args.get('specialization')
        min_rating = request.args.get('min_rating', type=float)
        max_fee = request.args.get('max_fee', type=float)
        
        conn = get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        query = """
            SELECT doctor_id, full_name, specialization, qualification, 
                   experience_years, consultation_fee, profile_image_url,
                   bio, languages_spoken, availability_hours, rating, 
                   total_reviews, hospital_affiliation
            FROM doctors 
            WHERE is_active = TRUE AND is_verified = TRUE
        """
        params = []
        
        if specialization:
            query += " AND specialization ILIKE %s"
            params.append(f"%{specialization}%")
        
        if min_rating:
            query += " AND rating >= %s"
            params.append(min_rating)
            
        if max_fee:
            query += " AND consultation_fee <= %s"
            params.append(max_fee)
        
        query += " ORDER BY rating DESC, total_reviews DESC"
        
        cur.execute(query, params)
        doctors = cur.fetchall()
        
        cur.close()
        conn.close()
        
        # Return JSON response for AJAX
        return jsonify({
            'success': True,
            'doctors': [dict(doctor) for doctor in doctors],
            'total': len(doctors)
        })
        
    except Exception as e:
        # Return JSON error response
        return jsonify({
            'success': False,
            'error': str(e),
            'doctors': [],
            'total': 0
        }), 500

# Route to book a consultation
@app.route('/api/consultations/book', methods=['POST'])
def book_consultation():
    """Book a consultation with a doctor"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['doctor_id', 'appointment_date', 'appointment_time', 'consultation_type']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Assume user_id is stored in session
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({
                'success': False,
                'error': 'User not authenticated'
            }), 401
        
        conn = get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Check if doctor exists and is available
        cur.execute("""
            SELECT consultation_fee FROM doctors 
            WHERE doctor_id = %s AND is_active = TRUE AND is_verified = TRUE
        """, (data['doctor_id'],))
        
        doctor = cur.fetchone()
        if not doctor:
            return jsonify({
                'success': False,
                'error': 'Doctor not found or not available'
            }), 404
        
        # Check if the time slot is available
        cur.execute("""
            SELECT consultation_id FROM consultations
            WHERE doctor_id = %s AND appointment_date = %s AND appointment_time = %s
            AND status NOT IN ('cancelled', 'completed')
        """, (data['doctor_id'], data['appointment_date'], data['appointment_time']))
        
        existing_consultation = cur.fetchone()
        if existing_consultation:
            return jsonify({
                'success': False,
                'error': 'This time slot is already booked'
            }), 400
        
        # Generate meeting link for video consultations
        meeting_link = None
        if data['consultation_type'] == 'video':
            meeting_link = f"https://meet.dukhtar.com/room/{uuid.uuid4()}"
        
        # Insert consultation booking
        cur.execute("""
            INSERT INTO consultations (user_id, doctor_id, appointment_date, appointment_time,
                                     consultation_type, consultation_fee, meeting_link, duration_minutes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING consultation_id
        """, (
            user_id,
            data['doctor_id'],
            data['appointment_date'],
            data['appointment_time'],
            data['consultation_type'],
            doctor['consultation_fee'],
            meeting_link,
            data.get('duration_minutes', 30)
        ))
        
        consultation_id = cur.fetchone()['consultation_id']
        
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'consultation_id': consultation_id,
            'meeting_link': meeting_link,
            'message': 'Consultation booked successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Route to get user's consultations
@app.route('/api/consultations', methods=['GET'])
def get_user_consultations():
    """Get all consultations for the current user"""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({
                'success': False,
                'error': 'User not authenticated'
            }), 401
        
        status = request.args.get('status')  # Filter by status
        
        conn = get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        query = """
            SELECT c.consultation_id, c.appointment_date, c.appointment_time,
                   c.consultation_type, c.status, c.duration_minutes,
                   c.consultation_fee, c.payment_status, c.meeting_link,
                   d.full_name as doctor_name, d.specialization,
                   d.profile_image_url as doctor_image
            FROM consultations c
            JOIN doctors d ON c.doctor_id = d.doctor_id
            WHERE c.user_id = %s
        """
        params = [user_id]
        
        if status:
            query += " AND c.status = %s"
            params.append(status)
        
        query += " ORDER BY c.appointment_date DESC, c.appointment_time DESC"
        
        cur.execute(query, params)
        consultations = cur.fetchall()
        
        # Convert time objects to strings for JSON serialization
        for consultation in consultations:
            if consultation['appointment_time']:
                consultation['appointment_time'] = consultation['appointment_time'].strftime('%H:%M')
            if consultation['appointment_date']:
                consultation['appointment_date'] = consultation['appointment_date'].isoformat()
        
        cur.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'consultations': consultations
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Route to send message during consultation
# @app.route('/api/consultations/<int:consultation_id>/messages', methods=['POST'])
# def send_message(consultation_id):
#     """Send a message during consultation"""
#     try:
#         data = request.get_json()
#         user_id = session.get('user_id')
        
#         if not user_id:
#             return jsonify({
#                 'success': False,
#                 'error': 'User not authenticated'
#             }), 401
        
#         # Validate that user is part of this consultation
#         conn = get_connection()
#         cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
#         cur.execute("""
#             SELECT consultation_id FROM consultations
#             WHERE consultation_id = %s AND user_id = %s
#         """, (consultation_id, user_id))
        
#         consultation = cur.fetchone()
#         if not consultation:
#             return jsonify({
#                 'success': False,
#                 'error': 'Consultation not found or access denied'
#             }), 404
        
#         # Insert message
#         cur.execute("""
#             INSERT INTO consultation_messages (consultation_id, sender_type, sender_id,
#                                              message_text, message_type, file_url)
#             VALUES (%s, %s, %s, %s, %s, %s)
#             RETURNING message_id
#         """, (
#             consultation_id,
#             'user',
#             user_id,
#             data.get('message_text'),
#             data.get('message_type', 'text'),
#             data.get('file_url')
#         ))
        
#         message_id = cur.fetchone()['message_id']
        
#         conn.commit()
#         cur.close()
#         conn.close()
        
#         return jsonify({
#             'success': True,
#             'message_id': message_id,
#             'message': 'Message sent successfully'
#         })
        
#     except Exception as e:
#         return jsonify({
#             'success': False,
#             'error': str(e)
#         }), 500

# Add a new route to get doctor's response and add it to history
@app.route('/api/consultations/<int:consultation_id>/doctor_response', methods=['POST'])
def add_doctor_response_to_history(consultation_id):
    """Add doctor's response to DukhtarAgent history (this would be called when doctor responds)"""
    try:
        data = request.get_json()
        user_id = session.get('user_id')
        
        if not user_id:
            return jsonify({
                'success': False,
                'error': 'User not authenticated'
            }), 401
        
        # Get consultation details
        conn = get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cur.execute("""
            SELECT c.consultation_id, d.full_name as doctor_name, d.specialization
            FROM consultations c
            JOIN doctors d ON c.doctor_id = d.doctor_id
            WHERE c.consultation_id = %s AND c.user_id = %s
        """, (consultation_id, user_id))
        
        consultation = cur.fetchone()
        if not consultation:
            return jsonify({
                'success': False,
                'error': 'Consultation not found'
            }), 404
        
        cur.close()
        conn.close()
        
        # Add doctor's response to DukhtarAgent history
        doctor_name = consultation['doctor_name']
        specialization = consultation['specialization']
        
        doctor_response = f"Dr. {doctor_name} ({specialization}) responded: {data.get('message_text', '')}"
        
        
        return jsonify({
            'success': True,
            'message': 'Doctor response added to history'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Add a new route to get consultation summary and add to history
@app.route('/api/consultations/<int:consultation_id>/summary', methods=['POST'])
def add_consultation_summary_to_history(consultation_id):
    """Add consultation summary to DukhtarAgent history when consultation ends"""
    try:
        user_id = session.get('user_id')
        
        if not user_id:
            return jsonify({
                'success': False,
                'error': 'User not authenticated'
            }), 401
        
        conn = get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get consultation details and all messages
        cur.execute("""
            SELECT c.consultation_id, c.appointment_date, c.appointment_time,
                   c.consultation_type, c.duration_minutes,
                   d.full_name as doctor_name, d.specialization
            FROM consultations c
            JOIN doctors d ON c.doctor_id = d.doctor_id
            WHERE c.consultation_id = %s AND c.user_id = %s
        """, (consultation_id, user_id))
        
        consultation = cur.fetchone()
        if not consultation:
            return jsonify({
                'success': False,
                'error': 'Consultation not found'
            }), 404
        
        # Get all messages from the consultation
        cur.execute("""
            SELECT sender_type, message_text, sent_at
            FROM consultation_messages
            WHERE consultation_id = %s
            ORDER BY sent_at ASC
        """, (consultation_id,))
        
        messages = cur.fetchall()
        
        cur.close()
        conn.close()
        
        # Create consultation summary
        doctor_name = consultation['doctor_name']
        specialization = consultation['specialization']
        appointment_date = consultation['appointment_date']
        consultation_type = consultation['consultation_type']
        
        # Summarize the conversation
        user_messages = [msg['message_text'] for msg in messages if msg['sender_type'] == 'user']
        doctor_messages = [msg['message_text'] for msg in messages if msg['sender_type'] == 'doctor']
        
        summary = f"""
        Consultation Summary:
        Doctor: Dr. {doctor_name} ({specialization})
        Date: {appointment_date}
        Type: {consultation_type}
        Duration: {consultation.get('duration_minutes', 'N/A')} minutes
        
        Key Discussion Points:
        User Questions/Concerns: {' | '.join(user_messages[:3])}
        Doctor's Advice: {' | '.join(doctor_messages[:3])}
        """
       
        
        
        return jsonify({
            'success': True,
            'message': 'Consultation summary added to history'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Route to get consultation messages
@app.route('/api/consultations/<int:consultation_id>/messages', methods=['GET'])
def get_consultation_messages(consultation_id):
    """Get all messages for a consultation"""
    try:
        user_id = session.get('user_id')
        
        if not user_id:
            return jsonify({
                'success': False,
                'error': 'User not authenticated'
            }), 401
        
        conn = get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Validate access to consultation
        cur.execute("""
            SELECT consultation_id FROM consultations
            WHERE consultation_id = %s AND user_id = %s
        """, (consultation_id, user_id))
        
        consultation = cur.fetchone()
        if not consultation:
            return jsonify({
                'success': False,
                'error': 'Consultation not found or access denied'
            }), 404
        
        # Get messages
        cur.execute("""
            SELECT message_id, sender_type, sender_id, message_text,
                   message_type, file_url, sent_at, is_read
            FROM consultation_messages
            WHERE consultation_id = %s
            ORDER BY sent_at ASC
        """, (consultation_id,))
        
        messages = cur.fetchall()
        
        cur.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'messages': messages
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Route to submit review for a doctor
@app.route('/api/doctors/<int:doctor_id>/reviews', methods=['POST'])
def submit_review(doctor_id):
    """Submit a review for a doctor after consultation"""
    try:
        data = request.get_json()
        user_id = session.get('user_id')
        
        if not user_id:
            return jsonify({
                'success': False,
                'error': 'User not authenticated'
            }), 401
        
        # Validate required fields
        if 'consultation_id' not in data or 'rating' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required fields: consultation_id and rating'
            }), 400
        
        if not (1 <= data['rating'] <= 5):
            return jsonify({
                'success': False,
                'error': 'Rating must be between 1 and 5'
            }), 400
        
        conn = get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Verify consultation exists and is completed
        cur.execute("""
            SELECT consultation_id FROM consultations
            WHERE consultation_id = %s AND user_id = %s AND doctor_id = %s
            AND status = 'completed'
        """, (data['consultation_id'], user_id, doctor_id))
        
        consultation = cur.fetchone()
        if not consultation:
            return jsonify({
                'success': False,
                'error': 'Consultation not found or not completed'
            }), 404
        
        # Insert review
        cur.execute("""
            INSERT INTO doctor_reviews (doctor_id, user_id, consultation_id, rating,
                                      review_text, is_anonymous)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (consultation_id) DO UPDATE SET
                rating = EXCLUDED.rating,
                review_text = EXCLUDED.review_text,
                is_anonymous = EXCLUDED.is_anonymous
        """, (
            doctor_id,
            user_id,
            data['consultation_id'],
            data['rating'],
            data.get('review_text'),
            data.get('is_anonymous', False)
        ))
        
        # Update doctor's average rating
        cur.execute("""
            UPDATE doctors SET 
                rating = (SELECT ROUND(AVG(rating), 2) FROM doctor_reviews WHERE doctor_id = %s),
                total_reviews = (SELECT COUNT(*) FROM doctor_reviews WHERE doctor_id = %s)
            WHERE doctor_id = %s
        """, (doctor_id, doctor_id, doctor_id))
        
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Review submitted successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ========================================
# MAIN EXECUTION
# ========================================

if __name__ == '__main__':
    print("Starting Dukhtar AI Assistant with Authentication...")
    print("Flask app will be available at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
    

