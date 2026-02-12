# Twitter Sentiment Analysis

## Project Overview
This project analyzes sentiments expressed in tweets related to various topics. Utilizing Natural Language Processing and machine learning techniques, it can categorize sentiments as positive, negative, or neutral based on the content of the tweets.

## Features
- Real-time sentiment analysis of tweets.
- Visual representation of sentiments over time.
- Integration with Twitter API for data acquisition.
- Support for multiple languages.
- User-friendly interface for non-technical users.

## Technologies
- **Programming Language:** Python
- **Libraries:** Tweepy, Pandas, NumPy, Scikit-learn, Matplotlib, NLTK, Flask
- **Database:** SQLite / PostgreSQL (depending on project requirements)
- **Deployment:** Heroku / AWS (for cloud deployment)

## Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Naniyarram/final_year_project.git
   cd final_year_project
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file and add your Twitter API keys:
   ```bash
   TWITTER_API_KEY=your_api_key
   TWITTER_API_SECRET=your_api_secret
   ```
4. Run the application:
   ```bash
   python app.py
   ```

## Workflow
1. The application connects to the Twitter API and fetches tweets based on specified criteria.
2. The collected tweets are preprocessed (cleaning, tokenization).
3. The model performs sentiment analysis on the processed tweets.
4. Results are stored in the database and visualized on the dashboard.

## Model Performance
- **Accuracy:** 85%
- **Precision:** 83%
- **Recall:** 84%
- **F1 Score:** 83%

## Use Cases
- Brand monitoring to gauge public sentiment towards products.
- Analyzing public opinion on political events or social movements.
- Tracking customer feedback for service improvement.

## Future Improvements
- Enhance the model with more training data.
- Implement real-time sentiment monitoring with alerts.
- Expand language support for global analysis.