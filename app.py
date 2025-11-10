"""
Main Application for AI Digital Twin Creator

Streamlit-based interactive dashboard for the AI Digital Twin Creator system.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

from data_processing import DataProcessor
from model_training import ModelTrainer
from recommendation_engine import RecommendationEngine

# Page configuration
st.set_page_config(
    page_title="AI Digital Twin Creator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation-card {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)


def load_and_process_data():
    """Load and process user data."""
    processor = DataProcessor()
    
    # Check if user has entered data
    if 'user_activities' in st.session_state and len(st.session_state.user_activities) > 0:
        # Convert user activities to DataFrame
        data = pd.DataFrame(st.session_state.user_activities)
        # Ensure timestamp is datetime
        data['timestamp'] = pd.to_datetime(data['timestamp'])
    else:
        # Use sample data if no user data
        data = processor.load_data()
    
    processed_data = processor.preprocess_data(data)
    features = processor.extract_features(processed_data)
    preferences = processor.get_user_preferences(processed_data)
    
    return processor, processed_data, features, preferences


@st.cache_resource
def train_models(processed_data, features):
    """Train ML models (cached resource)."""
    trainer = ModelTrainer()
    
    # Adjust clusters based on data size
    n_samples = len(features)
    n_clusters = trainer.habit_detector.n_clusters
    
    # Ensure clusters don't exceed samples
    if n_samples < n_clusters:
        trainer.habit_detector.n_clusters = max(2, n_samples)
    
    training_results = trainer.train_all(processed_data, features)
    return trainer, training_results


def main():
    """Main application function."""
    st.markdown('<div class="main-header">🤖 AI Digital Twin Creator</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Data status
        if 'user_activities' in st.session_state and len(st.session_state.user_activities) > 0:
            st.success(f"✅ Using **{len(st.session_state.user_activities)}** of your activities!")
        else:
            st.info("📊 Using **sample data** for demonstration")
            st.caption("Add your activities in the 'Add Activities' tab")
        
        st.markdown("---")
        
        # Data options
        st.subheader("Data Options")
        days_of_data = st.slider("Days of Data", min_value=7, max_value=90, value=30, disabled=True)
        st.caption("(Only applies to sample data)")
        
        # Model options
        st.subheader("Model Options")
        n_clusters = st.slider("Number of Habit Clusters", min_value=2, max_value=10, value=5)
        
        # Refresh button
        if st.button("🔄 Refresh Data & Models", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
    
    # Load and process data
    with st.spinner("Loading and processing data..."):
        processor, processed_data, features, preferences = load_and_process_data()
    
    # Train models (with automatic cluster adjustment)
    with st.spinner("Training ML models..."):
        trainer, training_results = train_models(processed_data, features)
    
    # Initialize recommendation engine
    recommendation_engine = RecommendationEngine(
        activity_predictor=trainer.activity_predictor,
        habit_detector=trainer.habit_detector,
        nlp_processor=trainer.nlp_processor
    )
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "➕ Add Activities", "📊 Dashboard", "🎯 Recommendations", "🔮 Predictions", "🧠 Habits", "💬 Chat"
    ])
    
    # Tab 0: Add Activities (Data Input)
    with tab1:
        st.header("➕ Add Your Activities")
        st.info("💡 **Enter your daily activities here!** The more activities you add, the better your digital twin will understand your patterns.")
        
        # Initialize session state for user activities
        if 'user_activities' not in st.session_state:
            st.session_state.user_activities = []
        
        # Activity input form
        with st.form("activity_form", clear_on_submit=True):
            st.subheader("📝 Add New Activity")
            
            col1, col2 = st.columns(2)
            
            with col1:
                activity_name = st.text_input(
                    "Activity Name *",
                    placeholder="e.g., Morning Exercise, Coding Session, Lunch",
                    help="What did you do? (e.g., Morning Exercise, Coding Session, Lunch)"
                )
                
                activity_date = st.date_input(
                    "Date *",
                    value=datetime.now().date(),
                    help="When did this activity happen?"
                )
                
                activity_time = st.time_input(
                    "Time *",
                    value=datetime.now().time(),
                    help="What time did you do this?"
                )
            
            with col2:
                category = st.selectbox(
                    "Category *",
                    options=["Work", "Health", "Meal", "Leisure", "Learning", "Errands", "Transport", "Break", "Rest"],
                    help="What type of activity is this?"
                )
                
                duration = st.number_input(
                    "Duration (minutes) *",
                    min_value=1,
                    max_value=480,
                    value=30,
                    help="How long did this activity take?"
                )
                
                energy_level = st.slider(
                    "Energy Level (1-10) *",
                    min_value=1,
                    max_value=10,
                    value=5,
                    help="How energetic did you feel? (1 = very tired, 10 = very energetic)"
                )
            
            location = st.selectbox(
                "Location *",
                options=["Home", "Office", "Gym", "Cafe", "Outdoors", "Other"],
                help="Where did this activity happen?"
            )
            
            notes = st.text_area(
                "Notes (Optional)",
                placeholder="Any additional details about this activity...",
                help="Add any extra information you want to remember"
            )
            
            submitted = st.form_submit_button("➕ Add Activity", use_container_width=True)
            
            if submitted:
                if activity_name:
                    # Combine date and time
                    timestamp = datetime.combine(activity_date, activity_time)
                    
                    # Create activity entry
                    activity_entry = {
                        'timestamp': timestamp,
                        'activity': activity_name,
                        'category': category,
                        'duration': int(duration),
                        'energy_level': int(energy_level),
                        'location': location,
                        'notes': notes if notes else ''
                    }
                    
                    # Add to session state
                    st.session_state.user_activities.append(activity_entry)
                    st.success(f"✅ Activity '{activity_name}' added successfully!")
                    st.rerun()
                else:
                    st.error("⚠️ Please enter an activity name!")
        
        st.markdown("---")
        
        # Display current activities
        st.subheader("📋 Your Activities")
        
        if len(st.session_state.user_activities) > 0:
            # Create DataFrame for display
            activities_df = pd.DataFrame(st.session_state.user_activities)
            activities_df['timestamp'] = pd.to_datetime(activities_df['timestamp'])
            activities_df = activities_df.sort_values('timestamp', ascending=False)
            
            # Display summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Activities", len(st.session_state.user_activities))
            with col2:
                st.metric("Unique Activities", activities_df['activity'].nunique())
            with col3:
                st.metric("Date Range", f"{activities_df['timestamp'].min().date()} to {activities_df['timestamp'].max().date()}")
            
            # Display activities table
            display_df = activities_df.copy()
            display_df['Date'] = display_df['timestamp'].dt.date
            display_df['Time'] = display_df['timestamp'].dt.strftime('%H:%M')
            display_df = display_df[['Date', 'Time', 'activity', 'category', 'duration', 'energy_level', 'location', 'notes']]
            display_df.columns = ['Date', 'Time', 'Activity', 'Category', 'Duration (min)', 'Energy', 'Location', 'Notes']
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Quick actions
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🗑️ Clear All Activities", use_container_width=True):
                    st.session_state.user_activities = []
                    st.success("All activities cleared!")
                    st.rerun()
            
            with col2:
                if st.button("🔄 Refresh Dashboard", use_container_width=True):
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.rerun()
            
            with col3:
                # Download as CSV
                csv = activities_df.to_csv(index=False)
                st.download_button(
                    label="💾 Download as CSV",
                    data=csv,
                    file_name=f"activities_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.warning("📝 **No activities added yet!** Start adding activities above to create your digital twin.")
            st.info("💡 **Tip:** Add at least 10-20 activities to get meaningful insights and recommendations.")
    
    # Tab 2: Dashboard
    with tab2:
        st.header("📊 Digital Twin Dashboard")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Activities", len(processed_data))
        
        with col2:
            st.metric("Unique Activities", processed_data['activity'].nunique())
        
        with col3:
            st.metric("Avg Duration (min)", f"{processed_data['duration'].mean():.1f}")
        
        with col4:
            st.metric("Avg Energy Level", f"{processed_data['energy_level'].mean():.1f}/10")
        
        st.markdown("---")
        
        # Activity distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Activity Distribution by Category")
            category_counts = processed_data['category'].value_counts()
            fig = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Activity Categories"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Top Activities")
            activity_counts = processed_data['activity'].value_counts().head(10)
            fig = px.bar(
                x=activity_counts.values,
                y=activity_counts.index,
                orientation='h',
                title="Most Frequent Activities",
                labels={'x': 'Count', 'y': 'Activity'}
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Time-based analysis
        st.subheader("Activity Patterns Over Time")
        
        # Daily activity count
        processed_data['date'] = pd.to_datetime(processed_data['timestamp']).dt.date
        daily_counts = processed_data.groupby('date').size().reset_index(name='count')
        daily_counts['date'] = pd.to_datetime(daily_counts['date'])
        
        fig = px.line(
            daily_counts,
            x='date',
            y='count',
            title="Daily Activity Count",
            labels={'date': 'Date', 'count': 'Number of Activities'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Hourly activity distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Activity Distribution by Hour")
            hourly_counts = processed_data.groupby('hour').size()
            fig = px.bar(
                x=hourly_counts.index,
                y=hourly_counts.values,
                title="Activities by Hour of Day",
                labels={'x': 'Hour', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Activity Distribution by Day of Week")
            if 'day_of_week' in processed_data.columns:
                day_counts = processed_data.groupby('day_of_week').size()
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                day_counts = day_counts.reindex([d for d in day_order if d in day_counts.index])
                fig = px.bar(
                    x=day_counts.index,
                    y=day_counts.values,
                    title="Activities by Day of Week",
                    labels={'x': 'Day', 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Day of week data not available. Please ensure timestamps are properly formatted.")
        
        # User preferences summary
        st.subheader("👤 User Profile Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Favorite Activities:**")
            for activity, count in list(preferences['favorite_activities'].items())[:5]:
                st.write(f"- {activity}: {count} times")
        
        with col2:
            st.write("**Preferences:**")
            st.write(f"- Most Common Location: {preferences['most_common_location']}")
            st.write(f"- Average Duration: {preferences['average_duration']:.1f} minutes")
            st.write(f"- Peak Energy Hours: {', '.join(map(str, preferences['peak_energy_hours']))}")
    
    # Tab 3: Recommendations
    with tab3:
        st.header("🎯 Personalized Recommendations")
        
        # Generate recommendations
        recommendations = recommendation_engine.generate_recommendations(
            processed_data, preferences, datetime.now()
        )
        
        if recommendations:
            st.info(f"Found {len(recommendations)} personalized recommendations for you!")
            
            for i, rec in enumerate(recommendations):
                with st.container():
                    st.markdown(f"""
                        <div class="recommendation-card">
                            <h4>{rec.get('title', 'Recommendation')}</h4>
                            <p>{rec.get('message', '')}</p>
                            <small>Type: {rec.get('type', 'N/A')} | Priority: {rec.get('priority', 0)}/10 | Category: {rec.get('category', 'N/A')}</small>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Feedback buttons
                    col1, col2, col3 = st.columns([1, 1, 3])
                    with col1:
                        if st.button("👍", key=f"like_{i}"):
                            recommendation_engine.add_feedback(f"rec_{i}", "positive", 5)
                            st.success("Thanks for your feedback!")
                    with col2:
                        if st.button("👎", key=f"dislike_{i}"):
                            recommendation_engine.add_feedback(f"rec_{i}", "negative", 2)
                            st.info("Feedback recorded. We'll improve!")
                    
                    st.markdown("---")
        else:
            st.warning("No recommendations available at this time.")
        
        # Feedback summary
        feedback_summary = recommendation_engine.get_feedback_summary()
        if feedback_summary['total_feedback'] > 0:
            st.subheader("📊 Feedback Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Feedback", feedback_summary['total_feedback'])
            with col2:
                st.metric("Average Rating", f"{feedback_summary['average_rating']:.2f}/5")
            with col3:
                st.metric("Positive Feedback", feedback_summary['positive_feedback'])
    
    # Tab 4: Predictions
    with tab4:
        st.header("🔮 Activity Predictions")
        
        if trainer.activity_predictor.models:
            st.info("Predicting your next activities based on historical patterns...")
            
            # Get predictions
            predictions = trainer.activity_predictor.predict_next_activities(periods=7)
            
            if predictions:
                # Display predictions
                st.subheader("Predicted Activities (Next 7 Days)")
                
                # Sort by probability
                sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
                
                for activity, probability in sorted_predictions:
                    st.progress(probability, text=f"{activity}: {probability:.1%} probability")
                
                # Visualization
                fig = px.bar(
                    x=[p[1] for p in sorted_predictions],
                    y=[p[0] for p in sorted_predictions],
                    orientation='h',
                    title="Activity Prediction Probabilities",
                    labels={'x': 'Probability', 'y': 'Activity'}
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed forecast for top activity
                if sorted_predictions:
                    top_activity = sorted_predictions[0][0]
                    st.subheader(f"Detailed Forecast: {top_activity}")
                    
                    try:
                        forecast = trainer.activity_predictor.predict(top_activity, periods=7)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=forecast['ds'],
                            y=forecast['yhat'],
                            mode='lines',
                            name='Predicted',
                            line=dict(color='blue')
                        ))
                        fig.add_trace(go.Scatter(
                            x=forecast['ds'],
                            y=forecast['yhat_lower'],
                            mode='lines',
                            name='Lower Bound',
                            line=dict(color='lightblue', dash='dash'),
                            showlegend=False
                        ))
                        fig.add_trace(go.Scatter(
                            x=forecast['ds'],
                            y=forecast['yhat_upper'],
                            mode='lines',
                            name='Upper Bound',
                            line=dict(color='lightblue', dash='dash'),
                            fill='tonexty',
                            fillcolor='rgba(173, 216, 230, 0.2)'
                        ))
                        fig.update_layout(
                            title=f"7-Day Forecast for {top_activity}",
                            xaxis_title="Date",
                            yaxis_title="Expected Count"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error generating forecast: {e}")
            else:
                st.warning("No predictions available. Need more data.")
        else:
            st.warning("Prediction models not trained. Please ensure sufficient data.")
    
    # Tab 5: Habits
    with tab5:
        st.header("🧠 Detected Habits")
        
        habits = training_results.get('habits', {})
        n_samples = training_results.get('n_samples', 0)
        n_clusters_used = training_results.get('n_clusters_used', 0)
        
        if n_samples < 2:
            st.warning("⚠️ **Insufficient data for habit detection.** Please add at least 2 activities to detect habits.")
            st.info("💡 **Tip:** Add more activities in the 'Add Activities' tab to enable habit detection.")
        elif habits:
            st.info(f"Detected {len(habits)} distinct habit clusters from your behavior patterns (using {n_clusters_used} clusters from {n_samples} activities).")
            
            for cluster_id, habit_info in habits.items():
                with st.expander(f"**Habit Cluster {cluster_id + 1}** ({habit_info['size']} activities)"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Common Activities:**")
                        for activity, count in habit_info.get('common_activities', {}).items():
                            st.write(f"- {activity}: {count} times")
                        
                        st.write("**Common Categories:**")
                        for category, count in habit_info.get('common_categories', {}).items():
                            st.write(f"- {category}: {count} times")
                    
                    with col2:
                        st.write("**Pattern Details:**")
                        st.write(f"- Average Duration: {habit_info.get('avg_duration', 0):.1f} minutes")
                        
                        st.write("**Preferred Times:**")
                        for time, count in habit_info.get('common_times', {}).items():
                            st.write(f"- {time}: {count} times")
                        
                        st.write("**Common Locations:**")
                        for location, count in habit_info.get('common_locations', {}).items():
                            st.write(f"- {location}: {count} times")
            
            # Habit visualization
            st.subheader("Habit Cluster Visualization")
            
            # Create cluster distribution chart
            cluster_sizes = {f"Cluster {k+1}": v['size'] for k, v in habits.items()}
            fig = px.pie(
                values=list(cluster_sizes.values()),
                names=list(cluster_sizes.keys()),
                title="Distribution of Activities Across Habit Clusters"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No habits detected. Need more data for habit detection.")
    
    # Tab 6: Chat
    with tab6:
        st.header("💬 Chat with Your Digital Twin")
        
        st.info("Ask questions or give instructions to your digital twin. It understands natural language!")
        
        # Chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.chat_message("user").write(message['content'])
            else:
                st.chat_message("assistant").write(message['content'])
        
        # User input
        user_input = st.chat_input("Type your message here...")
        
        if user_input:
            # Add user message to history
            st.session_state.chat_history.append({'role': 'user', 'content': user_input})
            
            # Process input
            response = recommendation_engine.process_user_input(user_input)
            
            # Generate response
            if response['intent'] == 'query':
                reply = f"I understand you want to see information. Here's what I found:\n\n"
                reply += f"- Your favorite activity: {list(preferences['favorite_activities'].keys())[0] if preferences['favorite_activities'] else 'N/A'}\n"
                reply += f"- Most common location: {preferences['most_common_location']}\n"
                reply += f"- Average activity duration: {preferences['average_duration']:.1f} minutes"
            elif response['intent'] == 'recommendation':
                recs = recommendation_engine.generate_recommendations(processed_data, preferences)
                if recs:
                    reply = f"Here are some recommendations:\n\n"
                    for i, rec in enumerate(recs[:3], 1):
                        reply += f"{i}. {rec['title']}: {rec['message']}\n"
                else:
                    reply = "I'm analyzing your patterns to provide recommendations. Please check back soon!"
            else:
                reply = response.get('response', "I understand. How can I help you?")
            
            # Add assistant response to history
            st.session_state.chat_history.append({'role': 'assistant', 'content': reply})
            
            # Rerun to display new messages
            st.rerun()
        
        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()


if __name__ == "__main__":
    main()

