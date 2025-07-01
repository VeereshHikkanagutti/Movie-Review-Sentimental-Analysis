# Movie Review Sentiment Analysis

This project performs sentiment analysis on IMDB movie reviews using various machine learning models. The goal is to classify reviews as positive or negative.

## Dataset

- **Source:** [IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- **File:** `IMDB Dataset.csv`
- The dataset contains 50,000 movie reviews labeled as positive or negative.

## Project Structure

```
Movie-Review-Sentimental-Analysis/
│
├── IMDB Dataset.csv
├── Movie Review - Sentimental Analysis.ipynb
└── README.md
```

## Main Steps

1. **Data Loading:** Load and sample the IMDB dataset.
2. **Text Cleaning:** Remove stopwords, punctuation, and apply stemming.
3. **Feature Extraction:** Use `CountVectorizer` to convert text to numerical features.
4. **Model Training:** Train and evaluate the following models:
    - Weighted Words
    - Multinomial Naive Bayes
    - Support Vector Machine (SVM)
    - K-Nearest Neighbour (KNN)
    - Logistic Regression
5. **Evaluation:** Compare models using accuracy, precision, recall, and F-score.

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- nltk
- matplotlib

Install dependencies with:
```bash
pip install pandas numpy scikit-learn nltk matplotlib
```

## Usage

1. Download or clone this repository.
2. Place `IMDB Dataset.csv` in the project folder.
3. Open `Movie Review - Sentimental Analysis.ipynb` in Jupyter Notebook or VS Code.
4. Run the notebook cells to execute the analysis.

## Results

The notebook compares the performance of different models and visualizes their metrics.

---


