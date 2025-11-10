# AI Digital Twin Creator

An intelligent system designed to develop a virtual replica of a user that learns, predicts, and adapts to individual behavioral patterns over time. This system uses AI and ML to analyze user data such as daily routines, preferences, and contextual inputs.

## 🎯 Features

- **Habit Detection**: Uses KMeans clustering to identify behavioral patterns and habits
- **Activity Prediction**: Time-series forecasting (Prophet) to predict future activities
- **Natural Language Processing**: spaCy-based NLP for understanding user input
- **Personalized Recommendations**: Intelligent recommendation engine providing suggestions, reminders, and productivity insights
- **Interactive Dashboard**: Streamlit-based UI with visualizations and insights
- **Feedback Loop**: Continuous learning system that improves recommendations based on user feedback

## 📋 Requirements

- Python 3.8 or higher
- See `requirements.txt` for all dependencies

## 🚀 Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download spaCy Language Model

```bash
python -m spacy download en_core_web_sm
```

### 3. Run the Application

```bash
python -m streamlit run app.py
```

**Note:** If `streamlit` command doesn't work, use `python -m streamlit` instead.

The application will open in your default web browser at `http://localhost:8501`

## 📁 Project Structure

```
.
├── app.py                      # Main Streamlit application
├── data_processing.py          # Data collection and preprocessing
├── model_training.py           # ML models (clustering, time-series, NLP)
├── recommendation_engine.py    # Recommendation system
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🔧 Module Descriptions

### `data_processing.py`
- Handles data collection and preprocessing
- Generates sample data for demonstration
- Extracts features for ML models
- Analyzes user preferences

### `model_training.py`
- **HabitDetector**: KMeans clustering for habit detection
- **ActivityPredictor**: Prophet time-series forecasting for activity prediction
- **NLPProcessor**: spaCy-based natural language processing

### `recommendation_engine.py`
- Generates personalized recommendations
- Provides reminders and suggestions
- Implements feedback loop for continuous learning
- Processes user input and generates contextual responses

### `app.py`
- Main Streamlit dashboard
- Interactive visualizations
- Real-time insights and recommendations
- Chat interface with digital twin

## 📊 Dashboard Features

### 1. Dashboard Tab
- Key metrics (total activities, unique activities, average duration, energy level)
- Activity distribution visualizations
- Time-based activity patterns
- User profile summary

### 2. Recommendations Tab
- Personalized recommendations based on behavior patterns
- Reminders and suggestions
- Feedback system (thumbs up/down)
- Feedback summary statistics

### 3. Predictions Tab
- Activity prediction probabilities
- 7-day forecast for top activities
- Visualizations of predicted patterns

### 4. Habits Tab
- Detected habit clusters
- Pattern analysis for each cluster
- Habit distribution visualization

### 5. Chat Tab
- Natural language interface
- Ask questions about your digital twin
- Get recommendations and insights
- Interactive conversation history

## 🎨 Example Output

### Sample Recommendations

1. **Morning Exercise Reminder**
   - Type: Reminder
   - Message: "Based on your routine, consider morning exercise for better energy."
   - Priority: 7/10
   - Category: Health

2. **Activity Prediction**
   - Type: Prediction
   - Message: "Based on your patterns, you're likely to do 'Coding Session' soon (probability: 25%)."
   - Priority: 8/10
   - Category: Prediction

3. **Work-Life Balance Insight**
   - Type: Insight
   - Message: "You spend 60% of your time on work. Consider more leisure activities."
   - Priority: 6/10
   - Category: Balance

### Sample Predictions

- **Coding Session**: 25% probability (next 7 days)
- **Reading**: 18% probability
- **Gym**: 15% probability
- **Coffee Break**: 12% probability

### Detected Habits

- **Cluster 1**: Work-focused activities (Coding Session, Work Meeting)
  - Average Duration: 75 minutes
  - Preferred Time: Afternoon
  - Common Location: Office

- **Cluster 2**: Health activities (Morning Exercise, Gym)
  - Average Duration: 45 minutes
  - Preferred Time: Morning
  - Common Location: Gym

## 🔄 How It Works

1. **Data Collection**: The system collects or simulates user activity data (activities, timestamps, duration, energy levels, locations)

2. **Data Processing**: Raw data is preprocessed to extract features for ML models

3. **Model Training**:
   - **Clustering**: KMeans identifies behavioral patterns and habits
   - **Time-Series**: Prophet models predict future activity occurrences
   - **NLP**: spaCy processes natural language input

4. **Recommendation Generation**: The recommendation engine combines:
   - Time-based patterns
   - Activity predictions
   - Detected habits
   - Productivity insights

5. **Feedback Loop**: User feedback improves future recommendations

6. **Visualization**: Interactive dashboard displays insights and predictions

## 🛠️ Customization

### Adjust Number of Clusters

In `app.py`, modify the sidebar slider:
```python
n_clusters = st.slider("Number of Habit Clusters", min_value=3, max_value=10, value=5)
```

### Change Prediction Period

In `recommendation_engine.py`, modify the prediction periods:
```python
predictions = self.activity_predictor.predict_next_activities(periods=7)  # Change 7 to desired days
```

### Add Custom Activities

In `data_processing.py`, modify the `_generate_sample_data` method to include your custom activities.

## 📝 Notes

- The system uses simulated data by default for demonstration purposes
- For production use, replace the sample data generator with real data collection
- spaCy model (`en_core_web_sm`) must be downloaded separately
- Prophet model may take a few seconds to train on first run

## 🔒 Privacy & Ethics

- Users can view and control their virtual profile
- All data processing is transparent
- Feedback loop allows users to improve recommendations
- System is designed with privacy and transparency in mind

## 🐛 Troubleshooting

### Issue: spaCy model not found
**Solution**: Run `python -m spacy download en_core_web_sm`

### Issue: Prophet installation errors
**Solution**: Ensure you have the latest pip: `pip install --upgrade pip` then `pip install prophet`

### Issue: Streamlit not opening
**Solution**: Check if port 8501 is available, or specify a different port: `streamlit run app.py --server.port 8502`

## 📚 Technologies Used

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **scikit-learn**: Machine learning (KMeans clustering)
- **Prophet**: Time-series forecasting
- **spaCy**: Natural language processing
- **Plotly**: Interactive visualizations
- **Matplotlib/Seaborn**: Additional plotting

## 🤝 Contributing

This is a demonstration project. Feel free to extend it with:
- Real data collection mechanisms
- Additional ML models
- Enhanced NLP capabilities
- More sophisticated recommendation algorithms
- User authentication and data persistence

## 📄 License

This project is provided as-is for educational and demonstration purposes.

## 🎓 Learning Outcomes

This project demonstrates:
- Machine learning model integration (clustering, time-series, NLP)
- Recommendation system design
- Interactive dashboard development
- Data processing and feature engineering
- Feedback loop implementation
- User-centric AI system design

---

**Built with ❤️ for AI/ML enthusiasts**

