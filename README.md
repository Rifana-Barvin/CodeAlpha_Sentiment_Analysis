
# Sentiment Analysis Using Machine Learning

## Project Overview
This project performs sentiment analysis on tweets using Natural Language Processing (NLP) and Machine Learning. It classifies tweets into positive and negative sentiment categories.

This project was developed as part of the CodeAlpha Data Analytics Internship.

## Objectives
- Analyze text data from tweets.
- Clean and preprocess text.
- Convert text into numerical features using TF-IDF.
- Train a Logistic Regression classification model.
- Evaluate model performance.
- Visualize results using a confusion matrix and accuracy chart.

## Dataset
The project uses a balanced sample derived from the Sentiment140 dataset.

- Original dataset: 1.6 million tweets.
- Sample dataset: 19,930 tweets before text preprocessing.
- Negative tweets: 9,976.
- Positive tweets: 9,954.

The sample CSV is stored at `data/sentiment_data.csv`. The original large dataset is excluded from GitHub.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK
- Scikit-learn

## Methodology

1. Load the tweet dataset.
2. Convert text to lowercase.
3. Remove URLs, mentions, hashtags symbols, and non-alphabetic characters.
4. Split the data into training and testing sets.
5. Apply TF-IDF vectorization with unigrams and bigrams.
6. Train a Logistic Regression model.
7. Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.
8. Save visualizations in the `outputs` folder.

## Model Performance

- Algorithm: Logistic Regression
- Feature extraction: TF-IDF
- Training samples: 15,898
- Testing samples: 3,975
- Test accuracy: 74.04%

The model achieved 74.04% accuracy on the held-out test set. Performance may vary with different datasets and preprocessing choices.

## Sample Predictions

| Tweet | Predicted Sentiment |
|---|---|
| I love this product, it is amazing! | Positive |
| This is the worst experience ever. | Negative |
| I am happy with the service. | Positive |

## Visualizations

The project generates:
- `outputs/confusion_matrix.png`
- `outputs/model_accuracy.png`

## Project Structure

```text
CodeAlpha_Sentiment_Analysis/
├── data/
│   └── sentiment_data.csv
├── outputs/
│   ├── confusion_matrix.png
│   └── model_accuracy.png
├── create_sample.py
├── sentiment_analysis.py
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run

1. Clone this repository.
2. Open the project folder in VS Code.
3. Create and activate a virtual environment.
4. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Run the analysis:

   ```bash
   python sentiment_analysis.py
   ```

## Limitations

- The model classifies only positive and negative sentiment.
- Sarcasm, slang, and context can affect predictions.
- The results are based on a sample of the original dataset.
- The current model is a baseline and could be improved further.

## Internship Task

CodeAlpha Data Analytics Internship — Task 4: Sentiment Analysis.
