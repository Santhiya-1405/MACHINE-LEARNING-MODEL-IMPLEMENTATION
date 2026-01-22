# MACHINE-LEARNING-MODEL-IMPLEMENTATION

Company : CODTECH IT SOLUTION 

Name : Santhiya E 

Intern ID : CTIS1228

Domain : python Programming

Duration : 4 Weeks

Mentor : Neela Santhosh

Description :

This project demonstrates the implementation of a Machine Learning–based Spam Email Detection system using Python and the Scikit-learn library. The main objective of the program is to classify emails as either spam or ham (not spam) by analyzing the textual content of the messages. It highlights how Natural Language Processing (NLP) techniques combined with machine learning algorithms can effectively solve real-world text classification problems.
The program starts by creating a small sample dataset containing email messages along with their corresponding labels. This dataset is organized using a Pandas DataFrame, which allows efficient handling and processing of text data. The email content is used as the input feature, while the label represents the output class for classification.
To transform the raw text into numerical data suitable for machine learning, the TF-IDF (Term Frequency–Inverse Document Frequency) Vectorizer is applied. This method converts text into feature vectors by assigning importance to words based on their frequency in an email and their rarity across the dataset. Common English stop words are removed to reduce noise and improve model accuracy.The dataset is then divided into training and testing sets using a stratified train-test split, ensuring balanced representation of both spam and ham emails in each set. The model used for classification is Multinomial Naive Bayes, which is particularly effective for text-based classification tasks due to its simplicity and fast performance.
After training, the model is evaluated using metrics such as accuracy, confusion matrix, and classification report to measure its performance. The trained model is also tested on new, unseen email messages to demonstrate its practical applicability. Overall, this project provides a clear and effective introduction to machine learning-based text classification and serves as a foundation for understanding spam detection and other NLP-based applications.

OUTPUT :

<img width="882" height="538" alt="Screenshot 2026-01-22 183039" src="https://github.com/user-attachments/assets/5fd4cb17-b64e-460d-bbda-c51318f5585d" />

