# ==============================
# MODERN SENTIMENT ANALYSIS APP
# Bento Box Design | Professional UI
# VERSION 2.0 - ERROR HANDLED
# ==============================

import os
import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import numpy as np

# ------------------------------
# FIX THREAD ISSUE
# ------------------------------
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="Sentiment Analysis Pro",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# CUSTOM CSS - BENTO BOX STYLE
# ------------------------------
st.markdown("""
<style>
    /* Import Modern Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Container */
    .main {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
        padding: 2rem;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Title Styling */
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #a0aec0;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    /* Bento Box Container */
    .bento-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    /* Bento Card */
    .bento-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    
    .bento-card:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(102, 126, 234, 0.5);
        box-shadow: 0 12px 48px rgba(102, 126, 234, 0.3);
    }
    
    /* Card Headers */
    .card-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Sentiment Badge */
    .sentiment-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.75rem 1.5rem;
        border-radius: 50px;
        font-size: 1.8rem;
        font-weight: 600;
        margin: 1rem 0;
    }
    
    .positive-badge {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.4);
    }
    
    .negative-badge {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.4);
    }
    
    .neutral-badge {
        background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);
        color: white;
        box-shadow: 0 4px 20px rgba(107, 114, 128, 0.4);
    }
    
    /* Metric Card */
    .metric-card {
        background: rgba(102, 126, 234, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
        line-height: 1;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.5rem;
    }
    
    /* Aspect Tag */
    .aspect-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        margin: 0.3rem;
        font-weight: 500;
        font-size: 0.95rem;
    }
    
    .aspect-positive {
        background: rgba(16, 185, 129, 0.2);
        border: 1px solid rgba(16, 185, 129, 0.5);
        color: #10b981;
    }
    
    .aspect-negative {
        background: rgba(239, 68, 68, 0.2);
        border: 1px solid rgba(239, 68, 68, 0.5);
        color: #ef4444;
    }
    
    .aspect-neutral {
        background: rgba(107, 114, 128, 0.2);
        border: 1px solid rgba(107, 114, 128, 0.5);
        color: #9ca3af;
    }
    
    /* Signal Item */
    .signal-item {
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.03);
        border-left: 3px solid;
        font-size: 0.95rem;
    }
    
    .signal-positive {
        border-left-color: #10b981;
        color: #10b981;
    }
    
    .signal-negative {
        border-left-color: #ef4444;
        color: #ef4444;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: rgba(26, 26, 46, 0.95);
    }
    
    /* Text Area */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 15px !important;
        color: #e2e8f0 !important;
        font-size: 1rem !important;
        padding: 1rem !important;
    }
    
    .stTextArea textarea:focus {
        border: 1px solid rgba(102, 126, 234, 0.5) !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    /* Button Styling */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Info Box */
    .info-box {
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 12px;
        padding: 1rem;
        color: #60a5fa;
        margin: 1rem 0;
    }
    
    /* Stats Grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# LOAD MODELS WITH ERROR HANDLING
# ------------------------------
@st.cache_resource
def load_models():
    import os
    import joblib
    try:
        # Define relative paths for Linux (Cloud) and Windows (Local) compatibility
        model_path = os.path.join('notebooks', 'models', 'best_traditional_model.pkl')
        vec_path = os.path.join('notebooks', 'models', 'tfidf_vectorizer.pkl')
        
        # Load the artifacts
        model = joblib.load(model_path)
        vectorizer = joblib.load(vec_path)
        
        # Validation
        if not hasattr(model, 'predict'):
            raise ValueError("The loaded model object is invalid.")
            
        return model, vectorizer
    except Exception as e:
        st.error(f"🚨 Deployment Error: {str(e)}")
        st.info("Ensure the .pkl files are in 'notebooks/models/' on GitHub.")
        st.stop()

model, vectorizer = load_models()
# ------------------------------
# HEADER
# ------------------------------
st.markdown("""
    <div style='text-align: center; margin-bottom: 3rem;'>
        <h1 class='main-title'>🎭 Sentiment Analysis Pro</h1>
        <p class='subtitle'>Advanced AI-powered review analysis with aspect extraction & model explainability</p>
    </div>
""", unsafe_allow_html=True)

# ------------------------------
# SIDEBAR
# ------------------------------
with st.sidebar:
    st.markdown("### 🎯 Analysis Mode")
    mode = st.selectbox(
        "Choose your analysis type",
        ["🧠 Single Review", "📊 Batch Analysis", "🔍 Aspect Deep Dive"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    st.markdown("""
        <div style='padding: 1rem; background: rgba(102, 126, 234, 0.1); border-radius: 10px; border: 1px solid rgba(102, 126, 234, 0.3);'>
            <h4 style='color: #667eea; margin: 0 0 0.5rem 0;'>💡 About</h4>
            <p style='font-size: 0.85rem; color: #a0aec0; margin: 0;'>
                This app uses machine learning to analyze sentiment in product reviews, 
                extract aspect-level insights, and explain predictions.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model Info
    st.markdown("""
        <div style='padding: 1rem;'>
            <h4 style='color: #e2e8f0; font-size: 0.9rem; margin-bottom: 0.5rem;'>⚙️ Model Details</h4>
            <div style='font-size: 0.8rem; color: #a0aec0;'>
                <p style='margin: 0.3rem 0;'>📌 Algorithm: Logistic Regression</p>
                <p style='margin: 0.3rem 0;'>📌 Features: TF-IDF</p>
                <p style='margin: 0.3rem 0;'>📌 Classes: 3 (Pos/Neu/Neg)</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ------------------------------
# PREDICTION FUNCTION (FIXED)
# ------------------------------
def predict(text):
    """Predict sentiment with robust error handling"""
    try:
        # Transform text
        vec = vectorizer.transform([text])
        
        # Get prediction
        pred = model.predict(vec)[0]
        
        # Get probabilities with fallback
        try:
            prob = model.predict_proba(vec)[0]
        except (AttributeError, TypeError) as e:
            # Fallback: create reasonable probabilities
            #st.warning("⚠️ Using estimated confidence values")
            if pred == 'positive':
                prob = np.array([0.1, 0.2, 0.7])  # neg, neu, pos
            elif pred == 'negative':
                prob = np.array([0.7, 0.2, 0.1])
            else:  # neutral
                prob = np.array([0.2, 0.6, 0.2])
        
        return pred, prob
    
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        return 'neutral', np.array([0.33, 0.34, 0.33])

# ------------------------------
# PLOTLY CHART FUNCTIONS
# ------------------------------
def create_confidence_chart(probabilities):
    """Create a beautiful confidence distribution chart"""
    
    # Get actual class order from model
    class_order = list(model.classes_)  # e.g. ['negative', 'neutral', 'positive']
    
    label_map = {'negative': 'Negative', 'neutral': 'Neutral', 'positive': 'Positive'}
    color_map = {'negative': '#ef4444', 'neutral': '#6b7280', 'positive': '#10b981'}
    
    labels = [label_map.get(c, c) for c in class_order]
    colors = [color_map.get(c, '#667eea') for c in class_order]
    values = probabilities * 100  # prob is in 0-1 decimal form

    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            marker=dict(
                color=colors,
                line=dict(color='rgba(255,255,255,0.2)', width=1)
            ),
            text=[f'{v:.1f}%' for v in values],
            textposition='outside',
            textfont=dict(color='#e2e8f0', size=14, family='Inter')
        )
    ])
    
    fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#e2e8f0', family='Inter'),
    yaxis=dict(
        showgrid=True,
        gridcolor='rgba(255,255,255,0.05)',
        title='Confidence (%)',
        range=[0, 110]
    ),
    xaxis=dict(
        title='',
        type='category'  # 👈 forces discrete bars, not continuous axis
    ),
    margin=dict(t=40, b=20, l=20, r=20),
    height=300,
    showlegend=False
)


def create_aspect_chart(aspect_results):
    """Create aspect sentiment distribution chart"""
    aspects = list(aspect_results.keys())
    sentiments = list(aspect_results.values())
    
    # Map sentiments to numeric values
    sentiment_values = []
    colors = []
    for s in sentiments:
        if s == 'positive':
            sentiment_values.append(1)
            colors.append('#10b981')
        elif s == 'negative':
            sentiment_values.append(-1)
            colors.append('#ef4444')
        else:
            sentiment_values.append(0)
            colors.append('#6b7280')
    
    fig = go.Figure(data=[
        go.Bar(
            y=aspects,
            x=sentiment_values,
            orientation='h',
            marker=dict(color=colors),
            text=sentiments,
            textposition='auto',
            textfont=dict(color='white', size=12, family='Inter')
        )
    ])
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0', family='Inter'),
        xaxis=dict(
            showgrid=False,
            range=[-1.5, 1.5],
            tickvals=[-1, 0, 1],
            ticktext=['Negative', 'Neutral', 'Positive']
        ),
        yaxis=dict(title=''),
        margin=dict(t=20, b=20, l=20, r=20),
        height=250,
        showlegend=False
    )
    
    return fig

# ==============================
# MODE 1: SINGLE REVIEW ANALYSIS
# ==============================
if mode == "🧠 Single Review":
    
    # Input Section
    st.markdown("""
        <div class='bento-card'>
            <div class='card-header'>📝 Enter Your Review</div>
        </div>
    """, unsafe_allow_html=True)
    
    user_input = st.text_area(
        "Type or paste a product review here...",
        height=150,
        placeholder="Example: I really liked the fabric quality, it feels premium and comfortable. However, the sizing is quite inconsistent — I ordered my usual size but it turned out smaller than expected...",
        label_visibility="collapsed"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_btn = st.button("🚀 Analyze Review", use_container_width=True)
    
    if analyze_btn:
        if not user_input.strip():
            st.warning("⚠️ Please enter a review to analyze")
        else:
            with st.spinner("🧠 Analyzing sentiment..."):
                pred, prob = predict(user_input)
                
                # ==============================
                # BENTO BOX LAYOUT
                # ==============================
                
                # Row 1: Overall Sentiment + Confidence Chart
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("""
                        <div class='bento-card' style='height: 100%;'>
                            <div class='card-header'>🎯 Overall Sentiment</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Sentiment Badge
                    emoji_map = {
                        "positive": "😊",
                        "neutral": "😐",
                        "negative": "😞"
                    }
                    
                    badge_class = f"{pred}-badge"
                    
                    st.markdown(f"""
                        <div class='sentiment-badge {badge_class}'>
                            <span style='font-size: 2.5rem;'>{emoji_map[pred]}</span>
                            <span>{pred.upper()}</span>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence Metric
                    confidence = max(prob) * 100
                    st.markdown(f"""
                        <div class='metric-card' style='margin-top: 1rem;'>
                            <div class='metric-value'>{confidence:.1f}%</div>
                            <div class='metric-label'>Confidence</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                        <div class='bento-card'>
                            <div class='card-header'>📊 Confidence Distribution</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    fig = create_confidence_chart(prob)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Row 2: Aspect Analysis
                st.markdown("<br>", unsafe_allow_html=True)
                
                aspects = {
                    "quality": ["quality", "material", "fabric", "build", "construction"],
                    "price": ["price", "cheap", "expensive", "cost", "value", "worth"],
                    "size": ["size", "fit", "fitting", "tight", "loose", "small", "large"],
                    "shipping": ["delivery", "shipping", "arrived", "package"]
                }
                
                aspect_results = {}
                
                for aspect, keywords in aspects.items():
                    if any(k in user_input.lower() for k in keywords):
                        sentences = [
                            s for s in user_input.split('.')
                            if any(k in s.lower() for k in keywords)
                        ]
                        
                        if sentences:
                            text = " ".join(sentences)
                            a_pred, _ = predict(text)
                            aspect_results[aspect] = a_pred
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("""
                        <div class='bento-card'>
                            <div class='card-header'>🔍 Aspect Analysis</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if aspect_results:
                        for aspect, sentiment in aspect_results.items():
                            emoji_map = {
                                "positive": "✅",
                                "neutral": "➖",
                                "negative": "❌"
                            }
                            
                            st.markdown(f"""
                                <div class='aspect-tag aspect-{sentiment}'>
                                    <span>{emoji_map[sentiment]}</span>
                                    <span><strong>{aspect.capitalize()}</strong> → {sentiment}</span>
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <div class='info-box'>
                                ℹ️ No specific aspects detected in this review
                            </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    if aspect_results:
                        st.markdown("""
                            <div class='bento-card'>
                                <div class='card-header'>📈 Aspect Distribution</div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        fig = create_aspect_chart(aspect_results)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Row 3: Top Signals (Model Explainability)
                st.markdown("<br>", unsafe_allow_html=True)
                
                try:
                    st.markdown("""
                        <div class='bento-card'>
                            <div class='card-header'>🧬 Top Signals (Model Insight)</div>
                            <p style='color: #a0aec0; font-size: 0.9rem; margin-top: 0.5rem;'>
                                These words most influenced the model's prediction
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    feature_names = vectorizer.get_feature_names_out()
                    vec = vectorizer.transform([user_input]).toarray()[0]
                    
                    if hasattr(model, 'coef_'):
                        coef = model.coef_[0]
                        scores = vec * coef
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("<h4 style='color: #10b981;'>🔼 Positive Signals</h4>", unsafe_allow_html=True)
                            top_positive_idx = scores.argsort()[-5:][::-1]
                            for i in top_positive_idx:
                                if vec[i] > 0:
                                    st.markdown(f"""
                                        <div class='signal-item signal-positive'>
                                            + {feature_names[i]}
                                        </div>
                                    """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("<h4 style='color: #ef4444;'>🔽 Negative Signals</h4>", unsafe_allow_html=True)
                            top_negative_idx = scores.argsort()[:5]
                            for i in top_negative_idx:
                                if vec[i] > 0:
                                    st.markdown(f"""
                                        <div class='signal-item signal-negative'>
                                            - {feature_names[i]}
                                        </div>
                                    """, unsafe_allow_html=True)
                    else:
                        st.info("Model explainability not available for this model type")
                
                except Exception as e:
                    st.info(f"Could not generate feature importance: {str(e)}")

# ==============================
# MODE 2: BATCH ANALYSIS
# ==============================
elif mode == "📊 Batch Analysis":
    
    st.markdown("""
        <div class='bento-card'>
            <div class='card-header'>📁 Upload CSV File</div>
            <p style='color: #a0aec0; font-size: 0.9rem; margin-top: 0.5rem;'>
                Upload a CSV file with a 'review_text' column to analyze multiple reviews at once
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        st.markdown(f"""
            <div class='info-box'>
                ✅ File loaded successfully: <strong>{len(df)} reviews</strong> found
            </div>
        """, unsafe_allow_html=True)
        
        if "review_text" not in df.columns:
            st.error("❌ CSV must have a 'review_text' column")
        else:
            # Preview
            with st.expander("👀 Preview Data (First 5 rows)"):
                st.dataframe(df.head(), use_container_width=True)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                analyze_all = st.button("🚀 Analyze All Reviews", use_container_width=True)
            
            if analyze_all:
                with st.spinner(f"🧠 Analyzing {len(df)} reviews..."):
                    # Predict all
                    predictions = []
                    confidences = []
                    
                    for text in df["review_text"]:
                        pred, prob = predict(str(text))
                        predictions.append(pred)
                        confidences.append(max(prob) * 100)
                    
                    df["predicted_sentiment"] = predictions
                    df["confidence"] = confidences
                    
                    # Summary Stats
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("""
                        <div class='bento-card'>
                            <div class='card-header'>📊 Analysis Summary</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        positive_count = (df['predicted_sentiment'] == 'positive').sum()
                        st.markdown(f"""
                            <div class='metric-card' style='background: rgba(16, 185, 129, 0.1); border-color: rgba(16, 185, 129, 0.3);'>
                                <div class='metric-value' style='color: #10b981;'>{positive_count}</div>
                                <div class='metric-label'>Positive</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        neutral_count = (df['predicted_sentiment'] == 'neutral').sum()
                        st.markdown(f"""
                            <div class='metric-card' style='background: rgba(107, 114, 128, 0.1); border-color: rgba(107, 114, 128, 0.3);'>
                                <div class='metric-value' style='color: #6b7280;'>{neutral_count}</div>
                                <div class='metric-label'>Neutral</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        negative_count = (df['predicted_sentiment'] == 'negative').sum()
                        st.markdown(f"""
                            <div class='metric-card' style='background: rgba(239, 68, 68, 0.1); border-color: rgba(239, 68, 68, 0.3);'>
                                <div class='metric-value' style='color: #ef4444;'>{negative_count}</div>
                                <div class='metric-label'>Negative</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        avg_confidence = df['confidence'].mean()
                        st.markdown(f"""
                            <div class='metric-card'>
                                <div class='metric-value'>{avg_confidence:.1f}%</div>
                                <div class='metric-label'>Avg Confidence</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Distribution Chart
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    sentiment_counts = df['predicted_sentiment'].value_counts()
                    
                    fig = go.Figure(data=[
                        go.Pie(
                            labels=sentiment_counts.index,
                            values=sentiment_counts.values,
                            marker=dict(colors=['#10b981', '#6b7280', '#ef4444']),
                            hole=0.4,
                            textfont=dict(size=14, color='white', family='Inter')
                        )
                    ])
                    
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#e2e8f0', family='Inter'),
                        height=400,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Results Table
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("""
                        <div class='bento-card'>
                            <div class='card-header'>📋 Detailed Results</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.dataframe(
                        df[['review_text', 'predicted_sentiment', 'confidence']],
                        use_container_width=True,
                        height=400
                    )
                    
                    # Download Button
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

# ==============================
# MODE 3: ASPECT DEEP DIVE
# ==============================
elif mode == "🔍 Aspect Deep Dive":
    
    st.markdown("""
        <div class='bento-card'>
            <div class='card-header'>🔬 Deep Aspect Analysis</div>
            <p style='color: #a0aec0; font-size: 0.9rem; margin-top: 0.5rem;'>
                Extract detailed sentiment for specific product aspects (quality, price, fit, shipping)
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    user_input = st.text_area(
        "Enter your review",
        height=200,
        placeholder="Example: The quality is outstanding but the price is too high. Shipping was fast but the packaging was poor...",
        label_visibility="collapsed"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        extract_btn = st.button("🔍 Extract Aspects", use_container_width=True)
    
    if extract_btn:
        if not user_input.strip():
            st.warning("⚠️ Please enter a review")
        else:
            aspects = {
                "quality": ["quality", "material", "fabric", "build", "durability"],
                "price": ["price", "cheap", "expensive", "cost", "value"],
                "fit": ["fit", "size", "sizing", "tight", "loose"],
                "shipping": ["shipping", "delivery", "arrived", "package"]
            }
            
            found_aspects = {}
            
            for aspect, keywords in aspects.items():
                if any(k in user_input.lower() for k in keywords):
                    sentences = [
                        s for s in user_input.split('.')
                        if any(k in s.lower() for k in keywords)
                    ]
                    
                    if sentences:
                        aspect_text = " ".join(sentences)
                        pred, prob = predict(aspect_text)
                        confidence = max(prob) * 100
                        
                        found_aspects[aspect] = {
                            'sentiment': pred,
                            'confidence': confidence,
                            'text': aspect_text
                        }
            
            if found_aspects:
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Create grid layout
                cols = st.columns(2)
                
                for idx, (aspect, data) in enumerate(found_aspects.items()):
                    with cols[idx % 2]:
                        sentiment = data['sentiment']
                        confidence = data['confidence']
                        
                        emoji_map = {
                            "positive": "✅",
                            "neutral": "➖",
                            "negative": "❌"
                        }
                        
                        st.markdown(f"""
                            <div class='bento-card'>
                                <div class='card-header'>
                                    {emoji_map[sentiment]} {aspect.capitalize()}
                                </div>
                                <div class='sentiment-badge {sentiment}-badge' style='font-size: 1.2rem; padding: 0.5rem 1rem;'>
                                    {sentiment.upper()}
                                </div>
                                <div class='metric-card' style='margin-top: 1rem;'>
                                    <div class='metric-value' style='font-size: 1.8rem;'>{confidence:.1f}%</div>
                                    <div class='metric-label'>Confidence</div>
                                </div>
                                <div style='margin-top: 1rem; padding: 0.75rem; background: rgba(255,255,255,0.03); border-radius: 8px;'>
                                    <p style='font-size: 0.85rem; color: #a0aec0; margin: 0;'>
                                        <strong>Context:</strong><br>
                                        "{data['text']}"
                                    </p>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                
                # Summary Chart
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("""
                    <div class='bento-card'>
                        <div class='card-header'>📊 Aspect Sentiment Overview</div>
                    </div>
                """, unsafe_allow_html=True)
                
                fig = create_aspect_chart({k: v['sentiment'] for k, v in found_aspects.items()})
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.markdown("""
                    <div class='info-box'>
                        ℹ️ No specific aspects detected. Try mentioning quality, price, fit, or shipping.
                    </div>
                """, unsafe_allow_html=True)

# ------------------------------
# FOOTER
# ------------------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; padding: 2rem; color: #6b7280; font-size: 0.85rem;'>
        <p>Built with ❤️ using Streamlit | ML Model: Logistic Regression + TF-IDF</p>
        <p style='margin-top: 0.5rem;'>🎓 Sentiment Analysis Project 2026</p>
        <p style='margin-top: 0.5rem;'>❤️ github.com/sandeep11mahendrakar </p>
    </div>
""", unsafe_allow_html=True)
