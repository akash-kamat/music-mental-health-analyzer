import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# 🎛️ Page Setup
st.set_page_config(
    page_title="Music & Mental Health", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🎧 Music & Mental Health: Depression Dashboard & Predictor</h1></div>', unsafe_allow_html=True)

# 📦 Load & process data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("mxmh.csv")
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        df = df.dropna(subset=["depression", "fav_genre", "hours_per_day", "age"])
        df["depression"] = df["depression"].astype(float)
        df["depression_binary"] = df["depression"].apply(lambda x: 1 if x > 4 else 0)
        df["age_group"] = pd.cut(df["age"], bins=[10, 18, 25, 35, 50, 100],
                                 labels=["<=18", "19-25", "26-35", "36-50", "51+"])
        if "primary_streaming_service" in df.columns:
            df["primary_streaming_service"] = df["primary_streaming_service"].fillna("Unknown")
        return df
    except FileNotFoundError:
        st.error("❌ CSV file 'mxmh.csv' not found. Please ensure the file is in the same directory.")
        st.stop()

df = load_data()

# Display dataset info in sidebar
with st.sidebar:
    st.header("📊 Dataset Information")
    st.markdown(f"""
    <div style="
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        color: #333333;
    ">
        <h4 style="color: #333333; margin-top: 0;">Dataset Overview</h4>
        <p style="color: #333333;"><strong>Total Records:</strong> {len(df)}</p>
        <p style="color: #333333;"><strong>Features:</strong> {len(df.columns)}</p>
        <p style="color: #333333;"><strong>Depression Cases:</strong> {df['depression_binary'].sum()}</p>
        <p style="color: #333333;"><strong>Non-Depression:</strong> {len(df) - df['depression_binary'].sum()}</p>
    </div>
    """, unsafe_allow_html=True)

# 🎵 Encode genres
le_genre = LabelEncoder()
df["fav_genre_encoded"] = le_genre.fit_transform(df["fav_genre"])

# 🤖 Train model and calculate performance metrics once
@st.cache_resource
def train_model_and_get_metrics():
    from sklearn.model_selection import cross_val_score
    
    X = df[["fav_genre_encoded", "hours_per_day", "age"]]
    y = df["depression_binary"]
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X, y)
    
    # Calculate performance metrics once
    cv_scores = cross_val_score(model, X, y, cv=5)
    
    return model, cv_scores

model, cv_scores = train_model_and_get_metrics()

# Pre-calculate statistics to avoid recalculation
@st.cache_data
def get_dataset_statistics():
    stats = {
        'avg_hours_depressed': df[df['depression_binary'] == 1]['hours_per_day'].mean(),
        'avg_hours_not_depressed': df[df['depression_binary'] == 0]['hours_per_day'].mean(),
        'overall_depression_rate': df['depression_binary'].mean(),
        'genre_depression_rates': df.groupby('fav_genre')['depression_binary'].mean().to_dict()
    }
    return stats

dataset_stats = get_dataset_statistics()

# Function to create uniform plots
def create_uniform_plot(plot_type, **kwargs):
    """Create plots with uniform sizing and styling"""
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set consistent styling
    sns.set_palette("husl")
    
    if plot_type == "countplot":
        plot = sns.countplot(ax=ax, **kwargs)
        # Rotate x-axis labels to prevent overlap
        plt.xticks(rotation=45, ha='right')
        
    elif plot_type == "boxplot":
        plot = sns.boxplot(ax=ax, **kwargs)
        
    # Improve layout
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#fafafa')
    
    # Adjust layout to prevent text overlap
    plt.tight_layout()
    
    # Add value labels on bars for countplots
    if plot_type == "countplot":
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', fontsize=9)
    
    return fig, ax

# 📊 Tabs
tab1, tab2 = st.tabs(["📈 Analysis", "🔮 Predict"])

# ------------------- TAB 1: ANALYSIS -------------------
with tab1:
    st.subheader("📊 Depression Analysis from Music Survey")
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        depression_rate = (df['depression_binary'].sum() / len(df)) * 100
        st.metric("Depression Rate", f"{depression_rate:.1f}%", delta=None)
    
    with col2:
        avg_hours = df['hours_per_day'].mean()
        st.metric("Avg Daily Hours", f"{avg_hours:.1f}h", delta=None)
    
    with col3:
        most_popular_genre = df['fav_genre'].mode()[0]
        st.metric("Most Popular Genre", most_popular_genre, delta=None)
    
    with col4:
        avg_age = df['age'].mean()
        st.metric("Average Age", f"{avg_age:.0f} years", delta=None)

    st.markdown("---")

    # First Row of Graphs
    st.subheader("🎵 Music Preferences & Depression")
    
    # Genre Analysis - Full Width
    fig1, ax1 = create_uniform_plot("countplot", 
                                   data=df, 
                                   x="fav_genre", 
                                   hue="depression_binary",
                                   order=df["fav_genre"].value_counts().index)
    ax1.set_title("🎵 Favorite Genre vs Depression Status", fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel("Music Genre", fontsize=12)
    ax1.set_ylabel("Number of People", fontsize=12)
    ax1.legend(title="Depression", labels=['No Depression', 'Depression'], loc='upper right')
    st.pyplot(fig1)
    plt.close()

    # Second Row - Two columns
    col1, col2 = st.columns(2)
    
    with col1:
        fig2, ax2 = create_uniform_plot("boxplot", 
                                       data=df, 
                                       x="depression_binary", 
                                       y="hours_per_day")
        ax2.set_title("⏱️ Daily Music Hours vs Depression", fontsize=14, fontweight='bold', pad=20)
        ax2.set_xlabel("Depression Status", fontsize=12)
        ax2.set_ylabel("Hours per Day", fontsize=12)
        ax2.set_xticklabels(['No Depression', 'Depression'])
        st.pyplot(fig2)
        plt.close()

    with col2:
        fig3, ax3 = create_uniform_plot("countplot", 
                                       data=df, 
                                       x="age_group", 
                                       hue="depression_binary")
        ax3.set_title("👥 Age Groups vs Depression", fontsize=14, fontweight='bold', pad=20)
        ax3.set_xlabel("Age Group", fontsize=12)
        ax3.set_ylabel("Number of People", fontsize=12)
        ax3.legend(title="Depression", labels=['No Depression', 'Depression'])
        st.pyplot(fig3)
        plt.close()

    # Third Row - Streaming Platform Analysis
    if "primary_streaming_service" in df.columns:
        st.subheader("📱 Streaming Platform Analysis")
        
        # Limit to top streaming services to avoid overcrowding
        top_services = df['primary_streaming_service'].value_counts().head(8).index
        df_filtered = df[df['primary_streaming_service'].isin(top_services)]
        
        fig4, ax4 = create_uniform_plot("countplot", 
                                       data=df_filtered, 
                                       x="primary_streaming_service", 
                                       hue="depression_binary")
        ax4.set_title("📱 Top Streaming Platforms vs Depression", fontsize=14, fontweight='bold', pad=20)
        ax4.set_xlabel("Streaming Platform", fontsize=12)
        ax4.set_ylabel("Number of People", fontsize=12)
        ax4.legend(title="Depression", labels=['No Depression', 'Depression'])
        st.pyplot(fig4)
        plt.close()
    
    # Additional Insights
    st.markdown("---")
    st.subheader("🔍 Key Insights")
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.markdown("""
        **📈 Depression Patterns:**
        - Higher depression rates observed in certain age groups
        - Music listening hours vary between depressed and non-depressed individuals
        - Genre preferences show interesting correlations with mental health
        """)
    
    with insights_col2:
        # Calculate some statistics
        depressed_avg_hours = df[df['depression_binary'] == 1]['hours_per_day'].mean()
        non_depressed_avg_hours = df[df['depression_binary'] == 0]['hours_per_day'].mean()
        
        st.markdown(f"""
        **📊 Quick Statistics:**
        - Depressed individuals listen to music: **{depressed_avg_hours:.1f}h/day** on average
        - Non-depressed individuals: **{non_depressed_avg_hours:.1f}h/day** on average
        - Total survey participants: **{len(df)}** people
        """)

# ------------------- TAB 2: PREDICTION -------------------
with tab2:
    st.subheader("🧠 Predict Depression Based on Music Habits")
    
    # Model Performance Section
    with st.expander("📊 Model Performance & Information", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Accuracy", f"{cv_scores.mean():.2%}")
        with col2:
            st.metric("Standard Deviation", f"±{cv_scores.std():.2%}")
        with col3:
            st.metric("Training Samples", f"{len(df)}")
        
        st.info("ℹ️ This model uses Random Forest algorithm with music genre, daily listening hours, and age as features.")
    
    st.markdown("---")
    
    # Prediction Interface
    st.subheader("🎯 Make a Prediction")
    
    # Create two columns for better layout
    input_col, result_col = st.columns([1, 1])
    
    with input_col:
        st.markdown("**Enter your information:**")
        
        # Genre selection with better formatting
        genre_list = sorted(list(le_genre.classes_))
        user_genre = st.selectbox(
            "🎵 What's your favorite music genre?", 
            genre_list,
            help="Select the genre you listen to most often"
        )
        
        # Hours slider with better description
        user_hours = st.slider(
            "🎧 How many hours do you listen to music daily?", 
            0.0, 24.0, 2.0, step=0.5,
            help="Include all forms of music listening (streaming, radio, etc.)"
        )
        
        # Age input
        user_age = st.number_input(
            "🎂 What's your age?", 
            min_value=10, max_value=100, value=25,
            help="Enter your current age"
        )
        
        # Prediction button
        predict_button = st.button("🔮 Predict Depression Risk", type="primary", use_container_width=True)
    
    with result_col:
        st.markdown("**Prediction Results:**")
        
        if predict_button:
            # Make prediction
            genre_encoded = le_genre.transform([user_genre])[0]
            input_df = pd.DataFrame([[genre_encoded, user_hours, user_age]],
                                    columns=["fav_genre_encoded", "hours_per_day", "age"])

            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0]
            
            # Display results
            if prediction == 1:
                st.error("😟 **Higher Risk of Depression**")
                risk_level = "High"
                confidence = prediction_proba[1] * 100
            else:
                st.success("😊 **Lower Risk of Depression**")
                risk_level = "Low"
                confidence = prediction_proba[0] * 100
            
            # Show confidence and additional info
            st.metric("Confidence Level", f"{confidence:.1f}%")
            
            # Risk factors analysis
            st.markdown("**Risk Factor Analysis:**")
            
            # Compare with dataset averages (using pre-calculated stats)
            if user_hours > dataset_stats['avg_hours_depressed']:
                st.warning(f"⚠️ Your listening time ({user_hours}h) is above average for depressed individuals ({dataset_stats['avg_hours_depressed']:.1f}h)")
            elif user_hours < dataset_stats['avg_hours_not_depressed']:
                st.info(f"ℹ️ Your listening time ({user_hours}h) is below average for non-depressed individuals ({dataset_stats['avg_hours_not_depressed']:.1f}h)")
            
            # Genre analysis (using pre-calculated stats)
            genre_depression_rate = dataset_stats['genre_depression_rates'].get(user_genre, dataset_stats['overall_depression_rate'])
            
            if genre_depression_rate > dataset_stats['overall_depression_rate']:
                st.warning(f"⚠️ {user_genre} listeners have a {genre_depression_rate:.1%} depression rate (vs {dataset_stats['overall_depression_rate']:.1%} overall)")
            else:
                st.info(f"ℹ️ {user_genre} listeners have a {genre_depression_rate:.1%} depression rate (vs {dataset_stats['overall_depression_rate']:.1%} overall)")
        
        else:
            st.info("👆 Fill in your information and click 'Predict' to see results")
    
    # Disclaimer
    st.markdown("---")
    
    # Additional Resources
    with st.expander("🆘 Mental Health Resources", expanded=False):
        st.markdown("""
        **If you need help:**
        - **National Suicide Prevention Lifeline:** 988 (US)
        - **Crisis Text Line:** Text HOME to 741741
        - **International Association for Suicide Prevention:** https://www.iasp.info/resources/Crisis_Centres/
        
        **Remember:** Seeking help is a sign of strength, not weakness. 💪
        """)

