# Dukhtar - AI-Powered Pregnancy Assistant

Dukhtar is a comprehensive AI-powered pregnancy and women's health assistant that combines user authentication with advanced conversational AI capabilities. The system provides personalized support for expecting mothers through text chat, voice interaction, and medical image analysis.

## Features

### 🔐 User Authentication
- Secure user registration and login system
- PostgreSQL database integration
- Session management
- User profile management

### 🤖 AI Chat Assistant
- **Text Chat**: Natural language conversations about pregnancy and women's health
- **Voice Input**: Speech-to-text with support for English and Urdu
- **Voice Output**: Text-to-speech responses in multiple languages
- **Image Analysis**: Medical document and prescription analysis
- **Search Integration**: Real-time information from trusted medical sources

### 🎯 Specialized Capabilities
- Pregnancy anxiety and stress management
- Family planning guidance
- Child spacing advice
- Mental wellbeing support
- Medical document interpretation
- 24/7 availability

## Technology Stack

### Backend
- **Flask**: Web framework
- **PostgreSQL**: User database
- **LangGraph**: AI agent orchestration
- **LangChain**: LLM integration
- **OpenAI**: GPT-4 and Whisper models
- **gTTS**: Text-to-speech for Urdu/Hindi
- **OpenCV**: Image processing
- **Tavily**: Web search integration

### Frontend
- **HTML5/CSS3**: Modern responsive design
- **JavaScript**: Interactive chat interface
- **Font Awesome**: Icons
- **Google Fonts**: Typography

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Dukhtar
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   TAVILY_API_KEY=your_tavily_api_key
   DATABASE_URL=your_postgresql_connection_string
   ```

4. **Set up PostgreSQL database**
   ```sql
   CREATE TABLE users (
       user_id SERIAL PRIMARY KEY,
       name VARCHAR(100) NOT NULL,
       email VARCHAR(100) UNIQUE NOT NULL,
       password_hash VARCHAR(255) NOT NULL,
       phone VARCHAR(20),
       number_of_children INTEGER,
       age_of_last_child INTEGER,
       current_weight DECIMAL(5,2),
       ongoing_diseases TEXT,
       additional_info TEXT,
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );
   ```

5. **Configure database connection**
   Update `config.py` with your PostgreSQL credentials:
   ```python
   def config():
       return {
           'host': 'localhost',
           'database': 'dukhtar_db',
           'user': 'your_username',
           'password': 'your_password'
       }
   ```

## Usage

### Starting the Application
```bash
python app.py
```

The application will be available at `http://localhost:5000`

### User Flow

1. **Registration/Login**: Users must create an account or log in to access AI features
2. **Home Page**: Personalized dashboard showing user-specific content
3. **Chat Interface**: Access the AI assistant through `/chat` route
4. **Features Available**:
   - Text conversations
   - Voice input (English/Urdu)
   - Image upload and analysis
   - Chat history management
   - Export conversations

### API Endpoints

#### Authentication (Public)
- `GET /` - Home page
- `GET /signup` - Registration page
- `POST /signup` - User registration
- `GET /login` - Login page
- `POST /login` - User authentication
- `GET /logout` - User logout

#### AI Chat (Requires Authentication)
- `GET /chat` - Chat interface
- `POST /api/chat` - Text chat
- `POST /api/process_voice` - Voice processing
- `POST /api/analyze_image` - Image analysis
- `POST /api/clear_history` - Clear chat history
- `GET /api/export_history` - Export chat history

## Security Features

- **Authentication Required**: All AI features require user login
- **Session Management**: Secure session handling
- **Password Hashing**: bcrypt password encryption
- **Input Validation**: Sanitized user inputs
- **Error Handling**: Comprehensive error management

## AI Capabilities

### Language Support
- **Input Languages**: English, Urdu, Hindi
- **Output Languages**: English, Urdu, Hindi
- **Auto-detection**: Automatic language detection for voice input

### Medical Image Analysis
- Handwritten prescription reading
- Medical document interpretation
- Multiple preprocessing techniques
- Confidence scoring for accuracy

### Voice Processing
- Real-time speech recognition
- Multi-language support
- Audio response generation
- Visible audio controls

## File Structure

```
Dukhtar/
├── app.py                 # Main Flask application
├── db.py                  # Database connection
├── config.py             # Database configuration
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables
├── templates/
│   ├── index.html       # Home page
│   ├── login.html       # Login page
│   ├── signup.html      # Registration page
│   └── test.html        # Chat interface
└── static/              # Static assets
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions, please contact the development team or create an issue in the repository.

## Privacy

- User data is stored securely in PostgreSQL
- Chat conversations are processed through secure APIs
- No personal health information is stored permanently
- Users can export and delete their chat history

---

**Dukhtar** - Your compassionate pregnancy and women's health assistant 🤱💕 