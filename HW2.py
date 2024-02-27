import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')


# preprocessing

# Handling missing values
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

# Encoding categorical variables
train_data = pd.get_dummies(train_data, columns=['Sex', 'Embarked'])

# Dropping irrelevant columns
train_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Splitting the data into features and target variable
X = train_data.drop('Survived', axis=1)
y = train_data['Survived']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#DT

# Initialize the decision tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Define hyperparameters for tuning
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(dt_classifier, param_grid, cv=5)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Initialize the decision tree classifier with the best parameters
best_dt_classifier = DecisionTreeClassifier(random_state=42, **best_params)

# Fit the model on the training data
best_dt_classifier.fit(X_train, y_train)

# Plot the decision tree
plt.figure(figsize=(15,10))
plot_tree(best_dt_classifier, feature_names=X.columns, class_names=['Not Survived', 'Survived'], filled=True)
plt.show()


#Apply the five-fold cross validation of your fine-tuned decision tree learning
#model to the Titanic training data to extract average classification accuracy

# Perform five-fold cross-validation
cv_scores = cross_val_score(best_dt_classifier, X_train, y_train, cv=5)

# Calculate average classification accuracy
average_accuracy = cv_scores.mean()

print("Average Classification Accuracy:", average_accuracy)


# Define features and target variable
X = train_data.drop('Survived', axis=1)
y = train_data['Survived']


# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Define hyperparameters for tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(rf_classifier, param_grid, cv=5)

# Fit the grid search to the data
grid_search.fit(X, y)

# Get the best parameters
best_params = grid_search.best_params_
#print("Best Parameters:", best_params)

# Initialize the Random Forest classifier with the best parameters
best_rf_classifier = RandomForestClassifier(random_state=42, **best_params)

# Perform five-fold cross-validation
cv_scores = cross_val_score(best_rf_classifier, X, y, cv=5)

# Calculate average classification accuracy
average_accuracy = cv_scores.mean()

print("Average Classification Accuracy:", average_accuracy)