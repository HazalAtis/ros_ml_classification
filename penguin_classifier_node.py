#!/usr/bin/env python3

import rospy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def main():
    rospy.init_node('penguin_classifier_node', anonymous=True)
    rospy.loginfo("🐧 Starting Penguin Classification using SVM")

    # Load dataset
    df = pd.read_csv('/home/ghazaleh/Penguin.csv')
    rospy.loginfo(f"📄 Dataset shape: {df.shape}")
    rospy.loginfo(f"📄 Columns: {list(df.columns)}")

    # Drop rows with missing values
    df = df.dropna()

    # Encode species labels (target)
    label_encoder = LabelEncoder()
    df['species'] = label_encoder.fit_transform(df['species'])

    # Encode sex and island
    df['sex'] = label_encoder.fit_transform(df['sex'])
    df['island'] = label_encoder.fit_transform(df['island'])

    # Select features and target
    features = ['culmen_length_mm', 'culmen_depth_mm']
    X = df[features]
    y = df['species']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train the model
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Classification report
    rospy.loginfo("\n✅ Classification Report:\n" + classification_report(y_test, y_pred))

    # Plotting
    plt.figure()
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', edgecolors='k')
    plt.title("Penguin Species Prediction (SVM)")
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.grid(True)
    plt.colorbar(label='Encoded Species')
    plt.show()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

