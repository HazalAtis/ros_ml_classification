#!/usr/bin/env python3

import rospy
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

def main():
    rospy.init_node('fruit_classifier_node', anonymous=True)
    rospy.loginfo("🍎🍊 Starting Fruit Classification using Linear Classifier")

    # Load dataset
    path = '/home/ghazaleh/fruits_weight_sphercity.csv'
    df = pd.read_csv(path)

    rospy.loginfo(f"Dataset columns: {df.columns.tolist()}")
    rospy.loginfo(f"Original dataset shape: {df.shape}")

    # Drop rows with missing values
    df = df.dropna()

    # Encode color strings to numeric values manually
    color_map = {
        'Red': 80,
        'Orange': 20,
        'Green': 40,
        'Greenish yellow': 60,
        'Reddish yellow': 70
    }
    df['Color'] = df['Color'].map(color_map)

    # Encode labels
    df['labels'] = df['labels'].map({'apple': 0, 'orange': 1})

    # Drop any rows that failed to convert
    df = df.dropna()

    # Ensure correct types
    df = df.astype({'Color': float, 'Weight': float, 'Sphericity': float, 'labels': int})

    # Features and label
    X = df[['Color', 'Weight', 'Sphericity']]
    y = df['labels']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    rospy.loginfo(f"Accuracy: {accuracy}")
    rospy.loginfo("\n" + classification_report(y_test, y_pred))

    # Visualization
    plt.scatter(X_test['Weight'], X_test['Sphericity'], c=y_pred, cmap='coolwarm', label='Predicted')
    plt.xlabel('Weight')
    plt.ylabel('Sphericity')
    plt.title('Fruit Classification: Weight vs Sphericity')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

