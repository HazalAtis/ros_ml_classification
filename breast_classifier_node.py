#!/usr/bin/env python3

import rospy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def main():
    rospy.init_node('breast_classifier_node', anonymous=True)
    rospy.loginfo("🎗️ Starting Breast Cancer Classification using SVM Linear Classifier")

    # Load dataset
    df = pd.read_csv('/home/ghazaleh/Breast_Cancer.csv')
    rospy.loginfo(f"📊 Dataset shape: {df.shape}")
    rospy.loginfo(f"🧾 Columns: {list(df.columns)}")

    # Drop ID column if present
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    # Encode diagnosis column: M=1, B=0
    df['diagnosis'] = LabelEncoder().fit_transform(df['diagnosis'])

    # Features and target
    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train linear SVM
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation
    rospy.loginfo("✅ Classification Report:\n" + classification_report(y_test, y_pred))

    # Plot first two features from test set
    plt.figure()
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm', edgecolor='k')
    plt.title("Breast Cancer Prediction (SVM)")
    plt.xlabel(df.columns[1])
    plt.ylabel(df.columns[2])
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

