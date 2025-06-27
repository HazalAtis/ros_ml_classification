#!/usr/bin/env python3

import rospy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def main():
    rospy.init_node('iris_classifier_node', anonymous=True)
    rospy.loginfo("🌸 Starting Iris Classification using KNN, Decision Tree, and Random Forest")

    # Load the dataset
    df = pd.read_csv('/home/ghazaleh/Iris.csv')

    rospy.loginfo(f"📊 Dataset columns: {list(df.columns)}")
    rospy.loginfo(f"📏 Dataset shape: {df.shape}")

    # Drop 'Id' column
    df = df.drop(columns=['Id'])

    # Encode species names to numbers
    label_encoder = LabelEncoder()
    df['Species'] = label_encoder.fit_transform(df['Species'])

    # Separate features and labels
    X = df.drop(columns=['Species'])
    y = df['Species']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifiers = {
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42)
    }

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        rospy.loginfo(f"\n🧠 {name} Classifier Accuracy: {acc:.4f}")
        rospy.loginfo(f"\n{classification_report(y_test, y_pred)}")

    # Visualize SepalLength vs SepalWidth
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X['SepalLengthCm'], X['SepalWidthCm'], c=y, cmap='viridis')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')
    plt.title('Iris Dataset Sepal Length vs Width')
    plt.grid(True)
    plt.colorbar(scatter, label='Encoded Species')
    plt.show()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

