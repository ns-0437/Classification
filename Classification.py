import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    st.title("Random Forest Classifier Example with Streamlit")

    # Load the train and test datasets
    train_data = pd.read_excel(r"C:\Users\Hp\Downloads\train.xlsx")
    test_data = pd.read_excel(r"C:\Users\Hp\Downloads\test.xlsx")

    # Display the train and test dataframes
    st.subheader("Train Data")
    st.write(train_data.head())

    st.subheader("Test Data")
    st.write(test_data.head())

    # Split the train data into features and target variable
    X_train = train_data.drop(columns=['target'])
    y_train = train_data['target']

    # Preprocess the data (scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train the classifier (Random Forest Classifier as an example)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)

    # Evaluate the model accuracy on the train set
    train_predictions = clf.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, train_predictions)
    st.subheader("Train Accuracy:")
    st.write(train_accuracy)

    # Preprocess the test data
    X_test_scaled = scaler.transform(test_data)

    # Predict the target values for the test set
    test_predictions = clf.predict(X_test_scaled)

    # Save the predictions to a file
    test_data['predicted_target'] = test_predictions
    st.subheader("Predicted Test Data:")
    st.write(test_data.head())

    # Save predictions to CSV
    test_data.to_csv('predicted_test.csv', index=False)
    st.success("Predictions saved to predicted_test.csv")

if __name__ == "__main__":
    main()
