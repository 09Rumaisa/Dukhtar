import os
import base64
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from dataclasses import dataclass
from io import BytesIO
import tempfile
import requests
from PIL import Image, ImageEnhance, ImageFilter

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

# LangChain imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

# Audio processing
from langdetect import detect
from gtts import gTTS
import speech_recognition as sr

# Load environment variables
from dotenv import load_dotenv
load_dotenv()
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
project_name = "dukhtar_chatbot"
os.environ["LANGCHAIN_PROJECT"] = project_name

# ========================================
# STATE DEFINITION
# ========================================

class DukhtarState(TypedDict):
    """State for the Dukhtar conversational agent"""
    messages: Annotated[List[HumanMessage | AIMessage], add_messages]
    current_language: str
    search_context: str
    audio_file_path: Optional[str]
    image_analysis_result: Optional[str]
    user_query: str
    tavily_links: List[Dict[str, str]]
    next_action: str

# ========================================
# SYSTEM PROMPT
# ========================================

DUKHTAR_SYSTEM_PROMPT = """
You are Dukhtar, a kind, compassionate, culturally-aware virtual therapist.
You help women deal with pregnancy anxiety, stress, spacing between children, family planning, and mental wellbeing.
You can also answer general FAQs about women's health during pregnancy.

Key guidelines:
- Always be supportive, non-judgmental, and use simple, short language
- If the user speaks Urdu, reply in Urdu. If they speak English, reply in English
- If unsure about language, gently ask a follow-up question
- If the user asks about topics other than pregnancy or family planning, politely redirect: "I am here to help with pregnancy-related questions. Please ask me about that."
- Use any search context provided to give accurate, up-to-date information
- If analyzing medical images, provide clear, simple explanations
- Always prioritize the user's emotional wellbeing and safety
"""

# ========================================
# TOOLS DEFINITION
# ========================================

@tool
def search_pregnancy_info(query: str) -> Dict[str, Any]:
    """Search for pregnancy and family planning related information using Tavily."""
    try:
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            return {"error": "Tavily API key not configured", "context": "", "links": []}
        
        tavily_url = "https://api.tavily.com/search"
        headers = {"Authorization": f"Bearer {tavily_api_key}", "Content-Type": "application/json"}
        payload = {
            "query": f"pregnancy family planning women health {query}",
            "search_depth": "basic",
            "max_results": 3
        }
        
        response = requests.post(tavily_url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])

        context_text = ""
        link_list = []

        for i, result in enumerate(results):
            context_text += f"\nTitle: {result.get('title')}\nSnippet: {result.get('snippet')}\nURL: {result.get('url')}\n"
            link_list.append({
                "title": result.get("title"),
                "url": result.get("url")
            })

        return {
            "context": context_text,
            "links": link_list,
            "error": None
        }

    except Exception as e:
        return {"error": f"Search failed: {str(e)}", "context": "", "links": []}

@tool
def analyze_medical_image(image_base64: str) -> str:
    """Analyze a medical image (prescription, report) and provide explanation in simple terms."""
    try:
        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(image_data)
            temp_image_path = tmp_file.name
        
        try:
            # Process image with multiple methods
            processed_images = preprocess_for_handwriting(temp_image_path)
            
            # Initialize OpenAI client
            client = ChatOpenAI(
                model="gpt-4o",
                api_key=os.getenv("OPENAI_API_KEY")
            )
            
            best_extraction = ""
            best_confidence = 0
            
            for processed_path, method_name in processed_images:
                with open(processed_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode()

                # Extract text using vision model
                extraction_prompt = """Extract ALL text from this medical document. This may contain handwritten prescriptions or reports with poor handwriting. 
                
                Focus on:
                - Patient name and details
                - Diagnosis/condition
                - Medication names (even if spelling seems off)
                - Dosages and frequencies
                - Doctor's instructions
                - Any medical abbreviations
                
                If text is unclear, make your best guess and indicate uncertainty with [?]. 
                Include even partially readable text."""

                messages = [
                    SystemMessage(content=extraction_prompt),
                    HumanMessage(content=[
                        {"type": "text", "text": "Please extract all text from this medical document:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ])
                ]
                
                response = client.invoke(messages)
                extracted_text = response.content
                
                # Calculate confidence
                confidence = calculate_extraction_confidence(extracted_text)
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_extraction = extracted_text
            
            # Generate explanation
            explanation_prompt = """You are a medical assistant helping patients understand their prescriptions and reports. 
            
            The extracted text may contain:
            - Handwritten medical abbreviations
            - Unclear medication names
            - Common prescription shorthand
            
            Please:
            1. Interpret medical abbreviations (e.g., "bid" = twice daily, "prn" = as needed)
            2. Suggest likely medication names if spelling is unclear
            3. Explain diagnosis in simple terms
            4. Provide dosage instructions clearly
            5. Mention any important warnings
            6. If text is unclear, acknowledge uncertainty but provide best interpretation
            
            Keep explanations simple and supportive, especially for pregnant women."""

            explanation_messages = [
                SystemMessage(content=explanation_prompt),
                HumanMessage(content=f"This was extracted from a medical document (possibly handwritten):\n\n{best_extraction}\n\nPlease explain the diagnosis and medications in simple terms, interpreting any medical abbreviations and unclear handwriting.")
            ]
            
            explanation_response = client.invoke(explanation_messages)
            
            # Clean up temporary files
            os.unlink(temp_image_path)
            for processed_path, _ in processed_images:
                if processed_path != temp_image_path and os.path.exists(processed_path):
                    os.unlink(processed_path)
            
            return explanation_response.content

        except Exception as e:
            # Clean up on error
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
            raise e

    except Exception as e:
        return f"⚠️ Failed to process the medical document. Error: {str(e)}"

@tool
def transcribe_audio(audio_base64: str) -> str:
    """Transcribe audio file to text using OpenAI Whisper."""
    try:
        # Decode base64 audio
        audio_data = base64.b64decode(audio_base64)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_file:
            tmp_file.write(audio_data)
            temp_audio_path = tmp_file.name
        
        try:
            # Initialize OpenAI client
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            # Transcribe audio with auto language detection
            with open(temp_audio_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-1"
                    # No language parameter = auto-detect
                )
            
            print(f"Transcription result: {transcript.text}")
            
            # Clean up
            os.unlink(temp_audio_path)
            
            return transcript.text.strip()
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            raise e

    except Exception as e:
        print(f"Transcription error: {str(e)}")
        return f"Failed to transcribe audio: {str(e)}"

@tool
def generate_audio_response(text: str, language: str = "en") -> str:
    """Generate audio response using TTS."""
    try:
        print(f"Generating audio for text: {text[:100]}...")
        print(f"Language: {language}")
        
        # Detect language if not provided
        if language == "auto":
            detected_lang = detect(text)
            language = detected_lang if detected_lang in ["ur", "hi", "en"] else "en"
            print(f"Detected language: {detected_lang}, using: {language}")
        
        if language in ["ur", "hi"]:
            # Use gTTS for Urdu/Hindi
            print("Using gTTS for Urdu/Hindi...")
            tts = gTTS(text, lang='ur')
            
            # Create temporary file with proper cleanup
            tmp_file = None
            try:
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                tmp_file_path = tmp_file.name
                tmp_file.close()  # Close the file handle
                
                # Save the TTS audio
                tts.save(tmp_file_path)
                
                # Read the audio content
                with open(tmp_file_path, "rb") as f:
                    audio_content = f.read()
                
                print(f"gTTS audio generated: {len(audio_content)} bytes")
                
            except Exception as e:
                print(f"gTTS error: {str(e)}")
                raise e
            finally:
                # Clean up the temporary file
                if tmp_file and os.path.exists(tmp_file_path):
                    try:
                        os.unlink(tmp_file_path)
                    except Exception as e:
                        print(f"Warning: Could not delete temp file {tmp_file_path}: {e}")
        else:
            # Use OpenAI TTS for English
            print("Using OpenAI TTS for English...")
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            speech = client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text
            )
            audio_content = speech.content
            print(f"OpenAI TTS audio generated: {len(audio_content)} bytes")
        
        # Return base64 encoded audio
        base64_audio = base64.b64encode(audio_content).decode()
        print(f"Base64 audio length: {len(base64_audio)}")
        return base64_audio
        
    except Exception as e:
        print(f"Error generating audio: {str(e)}")
        # Return empty string instead of error message to avoid confusion
        return ""

# ========================================
# IMAGE PROCESSING HELPERS
# ========================================

def preprocess_for_handwriting(image_path: str) -> List[tuple]:
    """Apply multiple preprocessing techniques for handwritten medical documents."""
    processed_images = []
    base_name = os.path.splitext(image_path)[0]
    
    # Load original image
    img = cv2.imread(image_path)
    if img is None:
        return [(image_path, "Original")]
    
    # Method 1: Enhanced contrast and sharpening
    method1 = enhance_for_handwriting(img.copy())
    path1 = f"{base_name}_method1.jpg"
    cv2.imwrite(path1, method1)
    processed_images.append((path1, "Enhanced Contrast"))
    
    # Method 2: Binarization
    method2 = apply_advanced_binarization(img.copy())
    path2 = f"{base_name}_method2.jpg"
    cv2.imwrite(path2, method2)
    processed_images.append((path2, "Advanced Binarization"))
    
    # Method 3: Morphological operations
    method3 = apply_morphological_enhancement(img.copy())
    path3 = f"{base_name}_method3.jpg"
    cv2.imwrite(path3, method3)
    processed_images.append((path3, "Morphological Enhancement"))
    
    return processed_images

def enhance_for_handwriting(img):
    """Specialized enhancement for handwritten text."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(filtered)
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
    unsharp = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
    return cv2.cvtColor(unsharp, cv2.COLOR_GRAY2BGR)

def apply_advanced_binarization(img):
    """Apply adaptive thresholding for better text separation."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    if binary.mean() < 127:
        binary = cv2.bitwise_not(binary)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

def apply_morphological_enhancement(img):
    """Apply morphological operations to clean up handwritten text."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    kernel2 = np.ones((2, 2), np.uint8)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2)
    enhanced = cv2.equalizeHist(closing)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

def calculate_extraction_confidence(extracted_text):
    """Calculate confidence score for extracted text based on medical keywords."""
    if not extracted_text:
        return 0
    
    medical_keywords = [
        'patient', 'diagnosis', 'prescription', 'medication', 'dose', 'dosage',
        'mg', 'ml', 'tablet', 'capsule', 'daily', 'twice', 'morning', 'evening',
        'before', 'after', 'meal', 'bid', 'tid', 'qid', 'prn', 'stat',
        'doctor', 'dr', 'hospital', 'clinic', 'treatment', 'therapy',
        'blood', 'pressure', 'sugar', 'diabetes', 'hypertension', 'infection'
    ]
    
    text_lower = extracted_text.lower()
    keyword_count = sum(1 for keyword in medical_keywords if keyword in text_lower)
    
    base_confidence = min(len(extracted_text) / 100, 50)
    keyword_confidence = keyword_count * 10
    
    return base_confidence + keyword_confidence

# ========================================
# AGENT NODES
# ========================================

def should_search(state: DukhtarState) -> bool:
    """Determine if we should search for additional information."""
    user_query = state.get("user_query", "")
    
    # Search triggers
    search_keywords = [
        "recent", "latest", "new", "current", "update", "research", "study",
        "statistics", "data", "news", "guidelines", "recommendations"
    ]
    
    return any(keyword in user_query.lower() for keyword in search_keywords)

def input_classifier(state: DukhtarState) -> DukhtarState:
    """Classify the input type and determine next action."""
    messages = state.get("messages", [])
    if not messages:
        state["next_action"] = "respond"
        return state
    
    last_message = messages[-1]
    
    # Check if we have search context to use
    if state.get("search_context"):
        state["next_action"] = "respond"
        return state
    
    # Check if we have image analysis result
    if state.get("image_analysis_result"):
        state["next_action"] = "respond"
        return state
    
    # Check if we should search
    if should_search(state):
        state["next_action"] = "search"
        return state
    
    state["next_action"] = "respond"
    return state

def search_node(state: DukhtarState) -> DukhtarState:
    """Search for relevant information."""
    user_query = state.get("user_query", "")
    
    # Perform search
    search_tool = search_pregnancy_info
    search_result = search_tool.invoke({"query": user_query})
    
    # Update state
    state["search_context"] = search_result.get("context", "")
    state["tavily_links"] = search_result.get("links", [])
    
    return state

def respond_node(state: DukhtarState) -> DukhtarState:
    """Generate response using the LLM."""
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Build context
    context_parts = []
    
    # Add search context if available
    if state.get("search_context"):
        context_parts.append(f"Relevant search information:\n{state['search_context']}")
    
    # Add image analysis if available
    if state.get("image_analysis_result"):
        context_parts.append(f"Medical image analysis:\n{state['image_analysis_result']}")
    
    # Prepare messages
    messages = [SystemMessage(content=DUKHTAR_SYSTEM_PROMPT)]
    
    # Add conversation history
    messages.extend(state.get("messages", []))
    
    # Add context if available
    if context_parts:
        context_message = HumanMessage(content=f"Additional context:\n\n{chr(10).join(context_parts)}")
        messages.append(context_message)
    
    # Generate response
    response = llm.invoke(messages)
    
    # Detect language
    try:
        detected_lang = detect(response.content)
        state["current_language"] = detected_lang
    except:
        state["current_language"] = "en"
    
    # Add response to messages
    state["messages"].append(AIMessage(content=response.content))
    
    return state

# ========================================
# GRAPH CONSTRUCTION
# ========================================

def create_dukhtar_agent():
    """Create the Dukhtar LangGraph agent."""
    
    # Create tools
    tools = [
        search_pregnancy_info,
        analyze_medical_image,
        transcribe_audio,
        generate_audio_response
    ]
    
    # Create tool node
    tool_node = ToolNode(tools)
    
    # Create graph
    workflow = StateGraph(DukhtarState)
    
    # Add nodes
    workflow.add_node("classifier", input_classifier)
    workflow.add_node("search", search_node)
    workflow.add_node("respond", respond_node)
    workflow.add_node("tools", tool_node)
    
    # Add edges
    workflow.add_edge(START, "classifier")
    
    workflow.add_conditional_edges(
        "classifier",
        lambda state: state["next_action"],
        {
            "search": "search",
            "respond": "respond"
        }
    )
    
    workflow.add_edge("search", "respond")
    workflow.add_edge("respond", END)
    
    # Add memory
    memory = MemorySaver()
    
    # Compile graph
    app = workflow.compile(checkpointer=memory)
    
    return app

# ========================================
# MAIN AGENT CLASS
# ========================================

class DukhtarAgent:
    """Main Dukhtar agent class for easy interaction."""
    
    def __init__(self):
        self.agent = create_dukhtar_agent()
        self.config = {"configurable": {"thread_id": "dukhtar_session"}}
    
    def process_text(self, text: str) -> str:
        """Process text input and return response."""
        initial_state = {
            "messages": [HumanMessage(content=text)],
            "user_query": text,
            "current_language": "en",
            "search_context": "",
            "audio_file_path": None,
            "image_analysis_result": None,
            "tavily_links": [],
            "next_action": ""
        }
        
        result = self.agent.invoke(initial_state, config=self.config)
        
        # Return the last AI message
        ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
        if ai_messages:
            return ai_messages[-1].content
        return "I'm here to help with pregnancy-related questions. How can I assist you?"
    
    def process_audio(self, audio_base64: str) -> Dict[str, Any]:
        """Process audio input and return text + audio response."""
        # Transcribe audio
        transcription = transcribe_audio.invoke({"audio_base64": audio_base64})
        print(f"Transcription: {transcription}")
        
        # Detect language of transcription
        try:
            detected_lang = detect(transcription)
            print(f"Detected language: {detected_lang}")
        except:
            detected_lang = "en"
            print("Language detection failed, defaulting to English")
        
        # Process text
        text_response = self.process_text(transcription)
        print(f"Text response: {text_response}")
        
        # Generate audio response with detected language
        audio_response = generate_audio_response.invoke({
            "text": text_response,
            "language": detected_lang if detected_lang in ["ur", "hi", "en"] else "en"
        })
        
        print(f"Audio response generated: {len(audio_response) if audio_response else 0} characters")
        
        return {
           "transcription": transcription,
            "text_response": text_response,
            "audio_response": audio_response,
            "detected_language": detected_lang
        }
    
    def process_image(self, image_base64: str, query: str = "") -> str:
        """Process medical image and return analysis."""
        # Analyze image
        analysis = analyze_medical_image.invoke({"image_base64": image_base64})
        
        # If there's an additional query, process it with the image context
        if query:
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "user_query": query,
                "current_language": "en",
                "search_context": "",
                "audio_file_path": None,
                "image_analysis_result": analysis,
                "tavily_links": [],
                "next_action": ""
            }
            
            result = self.agent.invoke(initial_state, config=self.config)
            
            # Return the last AI message
            ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
            if ai_messages:
                return ai_messages[-1].content
        
        return analysis
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        try:
            # Get state from memory
            state = self.agent.get_state(config=self.config)
            messages = state.values.get("messages", [])
            
            history = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    history.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    history.append({"role": "assistant", "content": msg.content})
            
            return history
        except:
            return []
    
    def clear_conversation(self):
        """Clear conversation history."""
        # Create new config with different thread_id
        import uuid
        self.config = {"configurable": {"thread_id": str(uuid.uuid4())}}

# ========================================
# MODULE EXPORTS
# ========================================

__all__ = ['DukhtarAgent', 'DUKHTAR_SYSTEM_PROMPT']