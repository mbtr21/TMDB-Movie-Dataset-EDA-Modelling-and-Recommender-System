import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class MachineLearningClassifier:
    # Constructor initializes the classifier with the dataset (data_frame) and the target variable.
    def __init__(self, data_frame, target):
        self.data_frame = data_frame  # The features dataset.
        self.target = target  # The target variable dataset.
        # Placeholder attributes for training and testing sets.
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.model = None  # Placeholder for the model.
        self.predictions = None  # Placeholder for model predictions.
        self.name = None  # Placeholder for the name of the model.

    # Splits the dataset into training and testing sets.
    def train_test_split(self, test_size=0.5, random_state=42):
        # Utilizes sklearn's train_test_split function to divide the data.
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data_frame, self.target,
                                                                                test_size=test_size,
                                                                                random_state=random_state)

    # Fits an XGBoost classification model to the training data.
    def fit_xgboost_classifier(self, objective='binary:logistic', n_estimators=100, seed=42):
        self.name = 'xgboost'  # Sets the model name.
        # Initializes and fits the XGBoost classifier to the training data.
        self.model = xgb.XGBClassifier(objective=objective, n_estimators=n_estimators, seed=seed)
        self.model.fit(self.X_train, self.y_train)
        # Predicts on the test set.
        self.predictions = self.model.predict(self.X_test)

    # Fits a Decision Tree classifier to the training data.
    def fit_decision_tree_classifier(self, random_state=42, min_samples_leaf=0.2):
        self.name = 'decision_tree'  # Sets the model name.
        # Initializes and fits the Decision Tree classifier to the training data.
        self.model = DecisionTreeClassifier(random_state=random_state, min_samples_leaf=min_samples_leaf)
        self.model.fit(self.X_train, self.y_train)
        # Predicts on the test set.
        self.predictions = self.model.predict(self.X_test)

    # Fits a Random Forest classifier to the training data.
    def fit_random_forest_classifier(self, max_depth=2, min_samples_leaf=0.2):
        self.name = 'random_forest'  # Sets the model name.
        # Initializes and fits the Random Forest classifier to the training data.
        self.model = RandomForestClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        self.model.fit(self.X_train, self.y_train)
        # Predicts on the test set.
        self.predictions = self.model.predict(self.X_test)

    # Fits a Logistic Regression model to the training data.
    def fit_logistic_regression_classifier(self, random_state=42):
        self.name = 'logistic_regression'  # Sets the model name.
        # Initializes and fits the Logistic Regression model to the training data.
        self.model = LogisticRegression(random_state=random_state)
        self.model.fit(self.X_train, self.y_train)
        # Predicts on the test set.
        self.predictions = self.model.predict(self.X_test)

    # Computes and returns performance metrics for the model.
    def metrics(self):
        # Calculates accuracy, precision, recall, and F1 score of the model predictions.
        accuracy = accuracy_score(self.y_test, self.predictions)
        precision = precision_score(self.y_test, self.predictions)
        recall = recall_score(self.y_test, self.predictions)
        f1 = f1_score(self.y_test, self.predictions)
        # Returns a dictionary with the model name and performance metrics.
        return {'name': self.name, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1_score': f1}