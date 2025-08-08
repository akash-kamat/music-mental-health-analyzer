# 🎧 Music & Mental Health Analyzer

An interactive Streamlit dashboard that analyzes the relationship between music listening habits and mental health using machine learning. This project explores how music preferences, listening duration, and demographics correlate with depression indicators.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Features

### 📊 **Interactive Data Analysis**
- **Genre Analysis**: Visualize depression rates across different music genres
- **Listening Habits**: Analyze daily music consumption patterns
- **Demographics**: Explore age group correlations with mental health
- **Platform Analysis**: Compare streaming service usage patterns

### 🔮 **ML-Powered Prediction**
- **Depression Risk Assessment**: Predict likelihood based on music habits
- **Confidence Scoring**: Get probability scores for predictions
- **Risk Factor Analysis**: Understand contributing factors
- **Real-time Insights**: Compare individual patterns with dataset averages

### 🎨 **Modern UI/UX**
- **Responsive Design**: Works seamlessly on desktop and mobile
- **Interactive Visualizations**: Hover effects and dynamic charts
- **Professional Styling**: Clean, modern interface with gradient themes
- **Performance Optimized**: Fast loading with cached computations

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/music-mental-health-analyzer.git
   cd music-mental-health-analyzer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   - Navigate to `http://localhost:8501`
   - Start exploring the dashboard!

## 📦 Dependencies

```txt
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
seaborn>=0.12.0
matplotlib>=3.6.0
```

## 📊 Dataset

The project uses the **Music & Mental Health Survey Dataset** which includes:
- **Demographics**: Age, primary streaming service
- **Music Preferences**: Favorite genres, listening hours
- **Mental Health**: Depression indicators and severity scores
- **Sample Size**: 700+ survey responses

### Data Features
- `age`: Participant age
- `fav_genre`: Preferred music genre
- `hours_per_day`: Daily music listening hours
- `depression`: Depression severity score (0-10)
- `primary_streaming_service`: Main music platform used

## 🤖 Machine Learning Model

### Algorithm: Random Forest Classifier
- **Features**: Genre (encoded), listening hours, age
- **Target**: Binary depression classification (threshold: >4)
- **Performance**: ~75-80% accuracy with cross-validation
- **Validation**: 5-fold cross-validation for robust evaluation

### Model Pipeline
1. **Data Preprocessing**: Handle missing values, encode categorical variables
2. **Feature Engineering**: Create age groups, binary depression labels
3. **Model Training**: Random Forest with 100 estimators
4. **Performance Evaluation**: Cross-validation and metrics calculation

## 📱 Application Structure

```
📁 project/
├── 📄 app.py                 # Main Streamlit application
├── 📄 mxmh.csv              # Dataset file
├── 📄 README.md             # Project documentation
└── 📄 requirements.txt      # Python dependencies
```

## 🎯 Usage Guide

### 📈 Analysis Tab
1. **Overview Metrics**: View key dataset statistics
2. **Genre Analysis**: Explore music preferences vs depression
3. **Listening Patterns**: Analyze daily music consumption
4. **Demographics**: Understand age-related trends
5. **Platform Insights**: Compare streaming service usage

### 🔮 Prediction Tab
1. **Input Your Data**: Select genre, hours, and age
2. **Get Prediction**: Click predict for risk assessment
3. **View Analysis**: See detailed risk factor breakdown
4. **Compare Patterns**: Understand how you compare to dataset averages

## 🔍 Key Insights

- **Genre Correlation**: Certain music genres show higher depression correlations
- **Listening Duration**: Extreme listening hours (very high/low) may indicate risk factors
- **Age Patterns**: Different age groups show varying depression rates
- **Platform Usage**: Streaming service preferences correlate with demographics

## ⚠️ Important Disclaimer

This tool is for **educational and research purposes only**. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. If you're experiencing symptoms of depression, please consult with a qualified healthcare professional.

## 🆘 Mental Health Resources

- **National Suicide Prevention Lifeline**: 988 (US)
- **Crisis Text Line**: Text HOME to 741741
- **International Association for Suicide Prevention**: [IASP Resources](https://www.iasp.info/resources/Crisis_Centres/)
- **Mental Health America**: [Find Help](https://www.mhanational.org/finding-help)

## 🛠️ Technical Details

### Performance Optimizations
- **Caching**: Model training and statistics pre-calculated
- **Lazy Loading**: Expensive computations cached with `@st.cache_resource`
- **Memory Management**: Proper plot cleanup to prevent memory leaks
- **Responsive Design**: Optimized for various screen sizes

### Code Quality
- **Modular Functions**: Reusable plotting and calculation functions
- **Error Handling**: Graceful handling of missing files and data issues
- **Documentation**: Comprehensive inline comments and docstrings
- **Best Practices**: Following Python and Streamlit conventions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

## 🙏 Acknowledgments

- Music & Mental Health Survey dataset contributors
- Streamlit community for excellent documentation
- Scikit-learn team for robust ML algorithms
- Open source community for inspiration and tools

## 📈 Future Enhancements

- [ ] Add more ML models (SVM, Neural Networks)
- [ ] Implement feature importance visualization
- [ ] Add data export functionality
- [ ] Include more mental health indicators
- [ ] Deploy to cloud platform (Heroku/Streamlit Cloud)
- [ ] Add user authentication and data persistence

---

⭐ **If you found this project helpful, please give it a star!** ⭐

*Made with ❤️ and Python*