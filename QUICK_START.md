# Quick Start Guide

## ✅ Installation Complete!

All dependencies have been installed successfully. You're ready to run the project!

## 🚀 How to Run the Project

### Method 1: Using Streamlit (Recommended)

1. **Open a terminal/command prompt**

2. **Navigate to the project directory:**
   ```bash
   cd "C:\jovi pro"
   ```

3. **Run the application:**
   ```bash
   python -m streamlit run app.py
   ```
   
   **Note:** If `streamlit` command doesn't work, always use `python -m streamlit` instead.

4. **The app will automatically open in your browser** at `http://localhost:8501`

   If it doesn't open automatically, you can manually navigate to:
   - `http://localhost:8501` in your web browser

### Alternative: Using Python directly (Recommended if streamlit command doesn't work)

```bash
cd "C:\jovi pro"
python -m streamlit run app.py
```

**This is the recommended method** if you get "'streamlit' is not recognized" error.

## 📱 Using the Application

Once the app is running, you'll see:

1. **Dashboard Tab** - View your activity metrics and visualizations
2. **Recommendations Tab** - Get personalized suggestions and reminders
3. **Predictions Tab** - See predicted future activities
4. **Habits Tab** - Explore detected behavior patterns
5. **Chat Tab** - Interact with your digital twin using natural language

## 🔄 Stopping the Application

- Press `Ctrl + C` in the terminal to stop the server
- Or close the terminal window

## ⚠️ Troubleshooting

### If the app doesn't start:
- Make sure you're in the correct directory: `C:\jovi pro`
- Check that all dependencies are installed: `pip list | findstr streamlit`
- Try running: `python -m streamlit run app.py`

### If you see import errors:
- Reinstall dependencies: `pip install -r requirements.txt`
- Make sure spaCy model is downloaded: `python -m spacy download en_core_web_sm`

### If port 8501 is already in use:
- Streamlit will automatically try the next available port (8502, 8503, etc.)
- Check the terminal output for the actual URL

## 📝 Notes

- The app uses **sample data** by default for demonstration
- First run may take a few seconds to generate data and train models
- All visualizations are interactive (hover, zoom, etc.)
- You can provide feedback on recommendations using 👍/👎 buttons

---

**Enjoy exploring your AI Digital Twin!** 🤖

