from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score
import pandas as pd

# Load the mushroom dataset from a text file
try:
    mushrooms = pd.read_csv("mushrooms.txt", delimiter=",", header=None)
except FileNotFoundError:
    print("Error: could not find file 'mushrooms.txt'")

# Convert all non-numeric values to numeric values using the LabelEncoder
le = LabelEncoder()
for column in mushrooms.columns:
    mushrooms[column] = le.fit_transform(mushrooms[column])

# Add column names to the dataframe
mushrooms.columns = [
    "class",
    "cap_shape",
    "cap_surface",
    "cap_color",
    "bruises",
    "odor",
    "gill_attachment",
    "gill_spacing",
    "gill_size",
    "gill_color",
    "stalk_shape",
    "stalk_root",
    "stalk_surface_above_ring",
    "stalk_surface_below_ring",
    "stalk_color_above_ring",
    "stalk_color_below_ring",
    "veil_type",
    "veil_color",
    "ring_number",
    "ring_type",
    "spore_print_color",
    "population",
    "habitat"
]

# Split the data into training and testing sets
X = mushrooms.drop('class', axis=1)
y = mushrooms['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the decision tree classifier using the ID3 algorithm
try:
    dt = DecisionTreeClassifier(criterion="entropy", random_state=42)
    dt.fit(X_train, y_train)
except ValueError as e:
    print("Error: could not train decision tree classifier:", e)

# Predict the class of new mushrooms
y_pred = dt.predict(X_test)

# Define k-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Perform k-fold cross-validation
scores = cross_val_score(dt, X, y, cv=kf)

# Evaluate the performance of the model
try:
    score = dt.score(X_test, y_test)
    print("Accuracy: {:.2f}%".format(score * 100))
    print("Accuracy: {:.2f}% (+/- {:.2f}%)".format(scores.mean() * 100, scores.std() * 100))
except ValueError as e:
    print("Error: could not evaluate performance of model:", e)
