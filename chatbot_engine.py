import pandas as pd
import os
import google.generativeai as genai
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# Load environment variables from .env file
load_dotenv()

class FAQEngine:
    """
    Handles structured, repetitive questions using a combination of TF-IDF and Sentence Transformers.
    """
    def __init__(self, filepath):
        """
        Initializes the FAQ Engine by loading and preparing the FAQ data.
        """
        try:
            self.df = pd.read_csv(filepath)
            self.questions = self.df['Question'].tolist()
            
            # TF-IDF Model
            self.tfidf_vectorizer = TfidfVectorizer().fit(self.questions)
            self.question_vectors_tfidf = self.tfidf_vectorizer.transform(self.questions)
            
            # Sentence Transformer Model for semantic search
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.question_embeddings_semantic = self.semantic_model.encode(self.questions, convert_to_tensor=True)
            
        except FileNotFoundError:
            print(f"Error: The file at {filepath} was not found.")
            self.df = pd.DataFrame(columns=['Question', 'Answer']) # Create empty dataframe
            self.questions = []

    def get_response(self, user_query, semantic_threshold=0.65):
        """
        Finds the best matching FAQ for a user query.
        First uses semantic search, then verifies with TF-IDF if needed.
        """
        if not self.questions:
            return None, 0.0

        # --- Semantic Search ---
        query_embedding = self.semantic_model.encode(user_query, convert_to_tensor=True)
        semantic_scores = util.pytorch_cos_sim(query_embedding, self.question_embeddings_semantic)[0]
        best_match_index_semantic = semantic_scores.argmax().item()
        best_score_semantic = semantic_scores[best_match_index_semantic].item()

        if best_score_semantic > semantic_threshold:
            # Return the answer if semantic score is high enough
            return self.df['Answer'].iloc[best_match_index_semantic], best_score_semantic
        else:
            # If semantic score is low, maybe it's a keyword match. Let's check TF-IDF.
            query_vector_tfidf = self.tfidf_vectorizer.transform([user_query])
            cosine_similarities = cosine_similarity(query_vector_tfidf, self.question_vectors_tfidf).flatten()
            best_match_index_tfidf = cosine_similarities.argmax()
            best_score_tfidf = cosine_similarities[best_match_index_tfidf]
            
            # Use a lower threshold for TF-IDF as a fallback
            if best_score_tfidf > 0.5:
                 return self.df['Answer'].iloc[best_match_index_tfidf], best_score_tfidf
            else:
                 # If both fail, return no answer
                 return None, 0.0

class GeminiEngine:
    """
    Handles open-ended or complex queries using the Gemini API.
    """
    def __init__(self):
        """
        Initializes the Gemini model.
        """
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables.")
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash-latest')
        except Exception as e:
            print(f"Error initializing Gemini Engine: {e}")
            self.model = None

    def get_response(self, user_query):
        """
        Sends a query to the Gemini API and gets a response.
        """
        if not self.model:
            return "Sorry, the Gemini service is currently unavailable."
        
        try:
            # Add a bit of context for the model
            prompt = f"You are Vislona's customer support chatbot. Answer the following user query concisely and helpfully: '{user_query}'"
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Sorry, I encountered an error while processing your request: {e}"

class HybridChatbot:
    """
    A dual-engine chatbot that uses the FAQ Engine first and falls back to the Gemini Engine.
    """
    def __init__(self, faq_filepath):
        """
        Initializes both the FAQ and Gemini engines.
        """
        self.faq_engine = FAQEngine(faq_filepath)
        self.gemini_engine = GeminiEngine()
        self.confidence_threshold = 0.7 # Threshold for FAQ engine confidence

    def get_response(self, user_query, mode='Hybrid'):
        """
        Provides a response based on the selected mode.
        """
        if mode == 'FAQ Only':
            faq_answer, score = self.faq_engine.get_response(user_query)
            if faq_answer:
                return faq_answer
            else:
                return "I'm sorry, I couldn't find an answer to that in my knowledge base. Please try rephrasing your question."

        elif mode == 'Gemini Only':
            return self.gemini_engine.get_response(user_query)

        elif mode == 'Hybrid':
            # First, try the FAQ engine
            faq_answer, score = self.faq_engine.get_response(user_query)
            
            if faq_answer and score >= self.confidence_threshold:
                # If confidence is high, return the FAQ answer
                return faq_answer
            else:
                # Otherwise, fall back to the Gemini engine
                return self.gemini_engine.get_response(user_query)