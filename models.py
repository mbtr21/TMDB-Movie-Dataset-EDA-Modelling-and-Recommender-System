import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class MachineLearningClassifier:
    def __init__(self, data_frame, target):
        self.data_frame = data_frame
        self.target = target
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.model = None
        self.predictions = None

    def train_test_split(self, test_size=0.5, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data_frame, self.target,
                                                                                test_size=test_size,
                                                                                random_state=random_state)

    def fit_xgboost_classifier(self, objective='binary:logistic', n_estimators=100, seed=42):
        self.model = xgb.XGBClassifier(objective=objective, n_estimators=n_estimators, seed=seed)
        self.model.fit(self.X_train, self.y_train)
        self.predictions = self.model.predict(self.X_test)

    def fit_decision_tree_classifier(self, random_state=42, min_samples_leaf=0.2):
        self.model = DecisionTreeClassifier(random_state=random_state, min_samples_leaf=min_samples_leaf)
        self.model.fit(self.X_train, self.y_train)
        self.predictions = self.model.predict(self.X_test)

    def fit_random_forest_classifier(self, max_depth=2, min_samples_leaf=0.2):
        self.model = RandomForestClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        self.model.fit(self.X_train, self.y_train)
        self.predictions = self.model.predict(self.X_test)

    def fit_logistic_regression_classifier(self, random_state=42):
        self.model = LogisticRegression(random_state=random_state)
        self.model.fit(self.X_train, self.y_train)
        self.predictions = self.model.predict(self.X_test)

    def metrics(self):
        accuracy = accuracy_score(self.y_test, self.predictions)
        precision = precision_score(self.y_test, self.predictions)
        recall = recall_score(self.y_test, self.predictions)
        f1 = f1_score(self.y_test, self.predictions)
        return accuracy, precision, recall, f1

