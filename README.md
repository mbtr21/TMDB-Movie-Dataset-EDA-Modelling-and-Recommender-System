# Movie Profitability Recommender System

## Introduction

This project encompasses a sophisticated recommender system designed to predict movie profitability. It integrates an interactive Exploratory Data Analysis (EDA) dashboard created with Dash and Plotly, offering users insights into various datasets through visualization. The core of this system lies in its machine learning capabilities, utilizing four models: Linear Regression, Random Forest, Decision Tree, and XGBoost. These models analyze and predict a movie's profitability based on selected features. Additionally, the system includes a recommendation engine that employs TF-IDF (Term Frequency-Inverse Document Frequency) for suggesting movies. This README outlines the project structure, installation process, and guidelines for utilization.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Dependencies](#dependencies)
- [Configuration](#configuration)


## Installation

Before installation, ensure you have Python installed on your system. This project is built using Python, so it's a prerequisite.

1. Clone the repository to your local machine:

```bash
git clone https://github.com/mbtr21/TMDB-Movie-Dataset-EDA-Modelling-and-Recommender-System.git
``` 
Navigate to the project directory:
cd movie-profitability-recommender-system
pip install -r requirements.txt

# Usage
Before every thing please run the reformat_data.py

And remember the project need run celery

To run the EDA dashboard and the machine learning dashboard, follow these steps:
```bash
python reformat_data.py
```
After reformat  data
run celery 
```bash
celery -A tasks  beat --loglevel INFO
```
After this run the main.ipynb
```bash
main.ipynb
```
Open your web browser and go to http://127.0.0.1:8050/.

Here, you can interact with the EDA dashboard, select features for the machine learning model, and view the performance (F1 score, accuracy, recall, precision) of each model via a bar chart.

For movie recommendations:

Enter a movie name in the designated box.
The system will display recommended movies based on TF-IDF analysis.

# Features
.Interactive EDA Dashboard: Visualize data using Dash and Plotly for in-depth analysis.

.Machine Learning Models: Utilize Linear Regression, Random Forest, Decision Tree, and XGBoost to predict movie profitability.

.Feature Selection: Users can select different features to tailor the prediction models.

.Model Performance Evaluation: View F1 score, accuracy, recall, and precision of models in a bar chart.

.Movie Recommendation: Enter a movie name to receive recommendations based on TF-IDF.

# Dependencies
Python 3.x

Dash

Plotly

Scikit-learn

Pandas

XGBoost

The requirements.txt file includes all necessary libraries.

# Configuration
No additional configuration is required to run the project beyond the installation steps provided.