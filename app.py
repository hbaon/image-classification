#!/usr/bin/env python3
"""
AI Image Classification System - Simple Version
Uses ONLY Hugging Face API (no local PyTorch required)
"""

import streamlit as st
from PIL import Image
import requests
import time
import io
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Image Classification",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 3rem;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .confidence-bar {
        background-color: #e0e0e0;
        height: 25px;
        border-radius: 12px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    .confidence-fill {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        height: 100%;
        border-radius: 12px;
        transition: width 1s ease;
    }
    .upload-area {
        border: 3px dashed #ccc;
        border-radius: 15px;
        padding: 3rem;
        text-align: center;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    .upload-area:hover {
        border-color: #1f77b4;
        background-color: #f8f9fa;
    }
                .metric-card {
                background: white;
                padding: 1rem;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                text-align: center;
            }
            
            /* Sidebar text size optimization - Stronger CSS */
            .css-1d391kg, .css-1d391kg * {
                font-size: 0.75rem !important;
            }
            
            /* Metric text size - Override Streamlit defaults */
            .stMetric, .stMetric * {
                font-size: 0.7rem !important;
            }
            
            /* Header text size */
            .css-1d391kg h3, .css-1d391kg h4 {
                font-size: 0.9rem !important;
                margin-bottom: 0.3rem !important;
                font-weight: 600 !important;
            }
            
            /* Metric labels and values */
            .stMetric > div > div:first-child {
                font-size: 0.65rem !important;
            }
            
            .stMetric > div > div:last-child {
                font-size: 0.8rem !important;
                font-weight: 500 !important;
            }
            
            /* Info text size */
            .stAlert, .stAlert * {
                font-size: 0.7rem !important;
                padding: 0.3rem !important;
            }
            
            /* Compact sidebar layout */
            .stMetric {
                margin-bottom: 0.3rem !important;
            }
            
            .stMetric > div {
                padding: 0.2rem !important;
            }
            
            /* Force smaller text everywhere in sidebar */
            [data-testid="stSidebar"] * {
                font-size: 0.75rem !important;
            }
            
            /* Responsive sidebar */
            @media (max-width: 768px) {
                [data-testid="stSidebar"] * {
                    font-size: 0.7rem !important;
                }
            }
            
            /* Button styling for example images */
            .stButton > button {
                font-size: 0.8rem !important;
                line-height: 1.0 !important;
                padding: 0.5rem 0.3rem !important;
                text-align: center !important;
                min-height: 50px !important;
                height: 50px !important;
                width: 100px !important;
                display: flex !important;
                justify-content: center !important;
                align-items: center !important;
                font-weight: 500 !important;
                margin: 0 auto !important;
            }
            
            /* Center images and align with buttons */
            .stImage {
                text-align: center !important;
                margin: 0 auto !important;
                width: 100% !important;
            }
            
            .stImage > img {
                display: block !important;
                margin: 0 auto !important;
                width: 100px !important;
                height: auto !important;
            }
            
            /* Column alignment - Force center */
            [data-testid="column"] {
                text-align: center !important;
                display: flex !important;
                flex-direction: column !important;
                align-items: center !important;
                justify-content: flex-start !important;
            }
            
            [data-testid="column"] > div {
                width: 100% !important;
                text-align: center !important;
                display: flex !important;
                flex-direction: column !important;
                align-items: center !important;
                justify-content: flex-start !important;
            }
            
            /* Force all content in columns to center */
            [data-testid="column"] * {
                text-align: center !important;
                margin-left: auto !important;
                margin-right: auto !important;
            }
            
            /* Perfect alignment for images and buttons */
            [data-testid="column"] > div {
                display: flex !important;
                flex-direction: column !important;
                align-items: center !important;
                justify-content: flex-start !important;
                width: 100% !important;
            }
            
            /* Remove any extra margins/padding that might cause misalignment */
            .stImage, .stButton {
                margin: 0 !important;
                padding: 0 !important;
                width: 100px !important;
            }
            
            /* Ensure buttons are exactly below images */
            .stButton {
                margin-top: 10px !important;
            }
    .api-status {
        padding: 0.5rem 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .api-status.success {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .api-status.warning {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)

# Hugging Face API configuration
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/microsoft/resnet-50"
API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "")

def test_api_connection():
    """Test Hugging Face API connection"""
    try:
        headers = {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}
        response = requests.get(HUGGINGFACE_API_URL, headers=headers, timeout=10)
        
        if response.status_code == 200:
            return True, "✅ API connection successful!"
        elif response.status_code == 401:
            return False, "⚠️ API token required. Please add HUGGINGFACE_API_TOKEN to .env file"
        else:
            return False, f"⚠️ API Error: {response.status_code}"
    except Exception as e:
        return False, f"❌ Connection failed: {str(e)}"

def classify_image(image):
    """Classify image using Hugging Face API"""
    try:
        # Resize image to 224x224 (ResNet-50 requirement)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((224, 224))
        
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Prepare headers
        headers = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "image/jpeg"} if API_TOKEN else {"Content-Type": "image/jpeg"}
        
        # Make API request
        response = requests.post(
            HUGGINGFACE_API_URL,
            data=img_byte_arr,
            headers=headers,
            timeout=30
        )
        
        # Track API calls
        if 'api_calls' not in st.session_state:
            st.session_state.api_calls = 0
        st.session_state.api_calls += 1
        
        # Check rate limit
        rate_limit_remaining = response.headers.get('X-RateLimit-Remaining', 'Unknown')
        if rate_limit_remaining != 'Unknown':
            st.info(f"📊 Remaining API calls: {rate_limit_remaining}")
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            st.error("⚠️ Rate limit exceeded. Please wait a few minutes before trying again.")
            return None
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Error calling Hugging Face API: {e}")
        return None

def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">🖼️ AI Image Classification</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload any image and let our AI classify it using Hugging Face API</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This AI system uses **ResNet50** architecture via Hugging Face API.
        
        **Features:**
        - 🚀 Free Hugging Face API
        - 🎯 1000+ object categories
        - 📱 Mobile-friendly interface
        - 🔒 No local model required
        
        **Built by:** [Nguyen Hoang Bao](https://github.com/hbaon)
        """)
        
        st.header("🔧 Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold (%)",
            min_value=0,
            max_value=100,
            value=10,
            help="Only show predictions above this confidence level"
        )
        
        st.header("📊 Model Info")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model", "ResNet50")
            st.metric("Dataset", "ImageNet")
        with col2:
            st.metric("Classes", "1000+")
            st.metric("Inference", "API")
        
        # API Usage Tracking
        st.header("📈 API Usage")
        api_calls = st.session_state.get('api_calls', 0)
        st.metric("Total Calls", api_calls)
        
        # Rate limit info
        if api_calls > 0:
            remaining_free = max(0, 30000 - api_calls)
            st.info(f"🆓 Free tier: ~{remaining_free:,} calls remaining this month")
        
        # API Status
        st.header("🔌 API Status")
        api_working, api_message = test_api_connection()
        if api_working:
            st.success(api_message)
        else:
            st.warning(api_message)
        
        if not API_TOKEN:
            st.warning("⚠️ Add HUGGINGFACE_API_TOKEN to .env file for API access")
        else:
            st.success("✅ API token configured")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file to classify",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
            help="Supported formats: PNG, JPG, JPEG, GIF, BMP"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            st.image(uploaded_file, caption="Uploaded Image", width='stretch')
            
            # File info
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.2f} KB",
                "File type": uploaded_file.type
            }
            st.json(file_details)
            
            # Classify button
            if st.button("🔍 Classify Image", type="primary", use_container_width=True):
                if api_working:
                    classify_image_api(uploaded_file, confidence_threshold)
                else:
                    st.error("❌ API not available. Please check connection.")
        
        st.divider()
        
        # Example images section
        st.subheader("🖼️ Example Images")
        example_images = [f for f in os.listdir('examples') if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        
        if example_images:
            # Display example images in columns
            st.write(f"📸 **{len(example_images)} example images available**")
            
            # Calculate number of rows needed
            num_cols = 4
            num_rows = (len(example_images) + num_cols - 1) // num_cols
            
            for row in range(num_rows):
                cols = st.columns(num_cols)
                for col in range(num_cols):
                    idx = row * num_cols + col
                    if idx < len(example_images):
                        img_name = example_images[idx]
                        with cols[col]:
                            img_path = os.path.join('examples', img_name)
                            st.image(img_path, width=100)
                            # Simple button text - only "Test"
                            if st.button("Test", key=f"test_{idx}", use_container_width=True):
                                # Load and classify example image
                                with open(img_path, 'rb') as f:
                                    # Create a proper mock object
                                    class MockUploadedFile:
                                        def __init__(self, file_path, file_name):
                                            self.file_path = file_path
                                            self.name = file_name
                                            self.size = os.path.getsize(file_path)
                                            self.type = 'image/jpeg'
                                        
                                        def read(self):
                                            with open(self.file_path, 'rb') as f:
                                                return f.read()
                                        
                                        def seek(self, pos):
                                            pass  # Mock seek method
                                    
                                    mock_file = MockUploadedFile(img_path, img_name)
                                    classify_image_api(mock_file, confidence_threshold)
    
    with col2:
        st.header("📊 Results")
        
        # Results placeholder
        if 'results' not in st.session_state:
            st.info("👆 Upload an image and click 'Classify' to see results")
        else:
            # Show image preview at top of results
            if 'current_image' in st.session_state:
                st.subheader("🖼️ Image Being Classified")
                if hasattr(st.session_state.current_image, 'file_path'):
                    # Example image
                    st.image(st.session_state.current_image.file_path, caption=st.session_state.current_image.name, width=300)
                else:
                    # Uploaded image
                    st.image(st.session_state.current_image, caption=st.session_state.current_image.name, width=300)
                st.divider()
            
            display_results(st.session_state.results, confidence_threshold)

def classify_image_api(uploaded_file, confidence_threshold):
    """Handle image classification via API"""
    
    # Show processing
    with st.spinner("🤖 AI is analyzing your image via Hugging Face API..."):
        start_time = time.time()
        
        # Load image - handle both real and mock files
        try:
            if hasattr(uploaded_file, 'file_path'):
                # Mock file from examples
                image = Image.open(uploaded_file.file_path)
            else:
                # Real uploaded file
                image = Image.open(uploaded_file)
        except Exception as e:
            st.error(f"❌ Error loading image: {e}")
            return
        
        # Classify via API
        results = classify_image(image)
        
        if results:
            # Convert HF format to our format
            predictions = []
            for result in results:
                predictions.append({
                    'class': result['label'].replace('_', ' ').title(),
                    'confidence': result['score'],
                    'probability': round(result['score'] * 100, 2)
                })
            
            # Sort by confidence
            predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Store results and current image
            processing_time = round(time.time() - start_time, 3)
            st.session_state.results = {
                'predictions': predictions,
                'processing_time': processing_time,
                'method': 'Hugging Face API'
            }
            st.session_state.current_image = uploaded_file
            
            st.success(f"✅ Classification completed in {processing_time} seconds!")
            st.rerun()
        else:
            st.error("❌ Classification failed. Please try again.")

def display_results(results, confidence_threshold):
    """Display classification results"""
    
    if 'predictions' not in results:
        return
    
    # Method used
    st.info(f"🔧 Inference Method: {results.get('method', 'Unknown')}")
    
    # Processing time
    st.metric("⏱️ Processing Time", f"{results['processing_time']} seconds")
    
    # Filter predictions by confidence threshold
    filtered_predictions = [
        pred for pred in results['predictions'] 
        if pred['probability'] >= confidence_threshold
    ]
    
    if not filtered_predictions:
        st.warning(f"⚠️ No predictions above {confidence_threshold}% confidence threshold")
        return
    
    # Display predictions
    st.subheader(f"🎯 Top {len(filtered_predictions)} Predictions")
    
    for i, pred in enumerate(filtered_predictions):
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 2])
            
            with col1:
                st.markdown(f"**{i+1}. {pred['class']}**")
            
            with col2:
                st.metric("Confidence", f"{pred['probability']:.1f}%")
            
            with col3:
                # Confidence bar
                st.markdown(f"""
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {pred['probability']}%"></div>
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
    
    # Performance metrics
    st.subheader("📈 Performance Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Predictions", len(results['predictions']))
    
    with col2:
        st.metric("Filtered Results", len(filtered_predictions))
    
    with col3:
        avg_confidence = sum(p['probability'] for p in filtered_predictions) / len(filtered_predictions)
        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")

if __name__ == "__main__":
    main()
