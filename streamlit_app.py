#!/usr/bin/env python3
"""
AI Image Classification System - Streamlit Version
Optimized for Hugging Face Spaces deployment
"""

import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time
import io

# Page configuration
st.set_page_config(
    page_title="AI Image Classification",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .confidence-bar {
        background-color: #e0e0e0;
        height: 20px;
        border-radius: 10px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    .confidence-fill {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease;
    }
    .upload-area {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .upload-area:hover {
        border-color: #1f77b4;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load ResNet50 model with caching"""
    with st.spinner("Loading AI model... This may take a few minutes on first run."):
        model = ResNet50(weights='imagenet')
        st.success("Model loaded successfully! 🎉")
        return model

def preprocess_image(img):
    """Preprocess image for model input"""
    try:
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to 224x224
        img = img.resize((224, 224))
        
        # Convert to array and preprocess
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

def classify_image(model, img_array):
    """Classify image using ResNet50 model"""
    try:
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        
        # Decode predictions
        decoded_predictions = decode_predictions(predictions, top=5)[0]
        
        # Format results
        results = []
        for _, label, confidence in decoded_predictions:
            results.append({
                'class': label.replace('_', ' ').title(),
                'confidence': float(confidence),
                'probability': round(float(confidence) * 100, 2)
            })
        
        return results
    except Exception as e:
        st.error(f"Error classifying image: {e}")
        return None

def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">🖼️ AI Image Classification</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload any image and let our AI classify it using advanced Convolutional Neural Networks</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This AI system uses **ResNet50** architecture pre-trained on ImageNet dataset.
        
        **Features:**
        - 🚀 Real-time classification
        - 🎯 1000+ object categories
        - 📱 Mobile-friendly interface
        - 🔒 Privacy-focused (local processing)
        
        **Built by:** [Nguyen Hoang Bao](https://github.com/hbaon)
        """)
        
        st.header("🔧 Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold (%)",
            min_value=0,
            max_value=100,
            value=50,
            help="Only show predictions above this confidence level"
        )
        
        st.header("📊 Model Info")
        st.metric("Model", "ResNet50")
        st.metric("Dataset", "ImageNet")
        st.metric("Classes", "1000+")
        st.metric("Architecture", "CNN")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
            help="Supported formats: PNG, JPG, JPEG, GIF, BMP"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            # File info
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.2f} KB",
                "File type": uploaded_file.type
            }
            st.json(file_details)
            
            # Classify button
            if st.button("🔍 Classify Image", type="primary", use_container_width=True):
                classify_button_clicked(uploaded_file, confidence_threshold)
    
    with col2:
        st.header("📊 Results")
        
        # Results placeholder
        if 'results' not in st.session_state:
            st.info("👆 Upload an image and click 'Classify' to see results")
        else:
            display_results(st.session_state.results, confidence_threshold)

def classify_button_clicked(uploaded_file, confidence_threshold):
    """Handle classification button click"""
    
    # Load model
    model = load_model()
    
    # Show processing
    with st.spinner("🤖 AI is analyzing your image..."):
        start_time = time.time()
        
        # Preprocess image
        img = Image.open(uploaded_file)
        img_array = preprocess_image(img)
        
        if img_array is not None:
            # Classify image
            results = classify_image(model, img_array)
            
            if results:
                # Calculate processing time
                processing_time = round(time.time() - start_time, 3)
                
                # Store results in session state
                st.session_state.results = {
                    'predictions': results,
                    'processing_time': processing_time,
                    'image': img
                }
                
                st.success(f"✅ Classification completed in {processing_time} seconds!")
                st.rerun()
            else:
                st.error("❌ Classification failed. Please try again.")
        else:
            st.error("❌ Error processing image. Please try a different image.")

def display_results(results, confidence_threshold):
    """Display classification results"""
    
    if 'predictions' not in results:
        return
    
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
        avg_confidence = np.mean([p['probability'] for p in filtered_predictions])
        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")

if __name__ == "__main__":
    main()
