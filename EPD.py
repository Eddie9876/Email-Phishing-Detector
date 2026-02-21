"""
Email Phishing Detector - ML Beginner Tutorial
================================================

This is a beginner-friendly phishing detector that teaches ML concepts.
We'll use simple, explainable features and a decision tree classifier.

Key ML Concepts You'll Learn:
- Feature extraction (turning emails into numbers)
- Training vs Testing data
- Model training and prediction
- Accuracy evaluation
"""

import re
import pandas as pd
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


class PhishingDetector:
    """
    A simple phishing detector that uses common phishing indicators
    """
    
    def __init__(self):
        self.model = DecisionTreeClassifier(max_depth=10, random_state=42)
        self.feature_names = [
            'has_suspicious_words',
            'url_count',
            'has_ip_address',
            'has_urgent_language',
            'suspicious_sender',
            'has_misspellings',
            'short_urls_count',
            'asks_for_credentials',
            'external_links_count'
        ]
    
    def extract_features(self, email_text, sender_email):
        """
        Extract numerical features from an email
        
        This is KEY in ML: we convert text into numbers that the model can understand.
        Each feature is a "signal" that helps identify phishing.
        """
        features = []
        
        # Feature 1: Suspicious words (verify, urgent, account, suspended, etc.)
        suspicious_words = ['verify', 'urgent', 'suspended', 'locked', 'confirm', 
                          'security', 'alert', 'unusual activity', 'click here']
        has_suspicious = sum(1 for word in suspicious_words if word.lower() in email_text.lower())
        features.append(min(has_suspicious, 5))  # Cap at 5 to avoid outliers
        
        # Feature 2: Number of URLs
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 
                         email_text)
        features.append(min(len(urls), 10))
        
        # Feature 3: Has IP address in URL (phishers often use IP addresses)
        has_ip = 1 if re.search(r'https?://\d+\.\d+\.\d+\.\d+', email_text) else 0
        features.append(has_ip)
        
        # Feature 4: Urgent language
        urgent_phrases = ['act now', 'immediately', 'urgent', 'expire', 'limited time']
        has_urgent = 1 if any(phrase in email_text.lower() for phrase in urgent_phrases) else 0
        features.append(has_urgent)
        
        # Feature 5: Suspicious sender (free email domains pretending to be companies)
        suspicious_sender_patterns = ['@gmail.com', '@yahoo.com', '@hotmail.com']
        company_words = ['bank', 'paypal', 'amazon', 'microsoft', 'apple']
        sender_suspicious = 0
        if any(domain in sender_email.lower() for domain in suspicious_sender_patterns):
            if any(word in email_text.lower() for word in company_words):
                sender_suspicious = 1
        features.append(sender_suspicious)
        
        # Feature 6: Common misspellings of trusted brands
        misspellings = ['paypa1', 'micros0ft', 'amaz0n', 'g00gle', 'facebok']
        has_misspell = 1 if any(word in email_text.lower() for word in misspellings) else 0
        features.append(has_misspell)
        
        # Feature 7: Short URLs (bit.ly, tinyurl, etc.)
        short_url_services = ['bit.ly', 'tinyurl', 'goo.gl', 't.co', 'ow.ly']
        short_urls = sum(1 for service in short_url_services if service in email_text.lower())
        features.append(short_urls)
        
        # Feature 8: Asks for credentials
        credential_words = ['password', 'ssn', 'social security', 'credit card', 'account number']
        asks_credentials = 1 if any(word in email_text.lower() for word in credential_words) else 0
        features.append(asks_credentials)
        
        # Feature 9: External links (not matching sender domain)
        sender_domain = sender_email.split('@')[-1] if '@' in sender_email else ''
        external_links = 0
        for url in urls:
            try:
                domain = urlparse(url).netloc
                if sender_domain and sender_domain not in domain:
                    external_links += 1
            except:
                pass
        features.append(min(external_links, 5))
        
        return features
    
    def train(self, emails_data):
        """
        Train the model on labeled email data
        
        emails_data: list of tuples (email_text, sender_email, is_phishing)
        is_phishing: 1 for phishing, 0 for legitimate
        """
        X = []  # Features (the numbers we extract)
        y = []  # Labels (phishing or not)
        
        print("Extracting features from training emails...")
        for email_text, sender_email, label in emails_data:
            features = self.extract_features(email_text, sender_email)
            X.append(features)
            y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data: 80% for training, 20% for testing
        # This is CRUCIAL in ML - we test on data the model hasn't seen!
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\nTraining on {len(X_train)} emails...")
        print(f"Testing on {len(X_test)} emails...")
        
        # Train the model (this is where the ML magic happens!)
        self.model.fit(X_train, y_train)
        
        # Evaluate on test data
        y_pred = self.model.predict(X_test)
        
        print("\n" + "="*50)
        print("MODEL PERFORMANCE")
        print("="*50)
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nDetailed Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Legitimate', 'Phishing']))
        
        # Show feature importance
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE (what the model learned)")
        print("="*50)
        importances = self.model.feature_importances_
        for name, importance in sorted(zip(self.feature_names, importances), 
                                      key=lambda x: x[1], reverse=True):
            print(f"{name:30s}: {importance:.3f}")
    
    def predict(self, email_text, sender_email, explain=True):
        """
        Predict if an email is phishing
        
        Returns: (prediction, confidence, explanation)
        """
        features = self.extract_features(email_text, sender_email)
        features_array = np.array([features])
        
        prediction = self.model.predict(features_array)[0]
        
        # Get confidence (probability)
        probabilities = self.model.predict_proba(features_array)[0]
        confidence = probabilities[prediction]
        
        result = "PHISHING" if prediction == 1 else "LEGITIMATE"
        
        if explain:
            print(f"\n{'='*60}")
            print(f"PREDICTION: {result} (Confidence: {confidence:.1%})")
            print(f"{'='*60}")
            print("\nFeature Analysis:")
            for name, value in zip(self.feature_names, features):
                status = "✓" if value > 0 else " "
                print(f"{status} {name:30s}: {value}")
            print(f"{'='*60}\n")
        
        return prediction, confidence


# ============================================================================
# DEMO: Creating Sample Training Data
# ============================================================================

def create_sample_training_data():
    """
    Create sample emails for demonstration
    In a real system, you'd load this from a dataset
    """
    
    # Phishing emails (label = 1)
    phishing_emails = [
        ("Your PayPal account has been suspended! Urgent action required. Click here to verify: http://192.168.1.1/paypal", 
         "security@gmail.com", 1),
        ("URGENT: Your bank account will be locked in 24 hours. Verify immediately: http://bit.ly/verify123", 
         "alerts@yahoo.com", 1),
        ("Congratulations! You've won $1,000,000. Click here now to claim: http://winner.com/claim?id=123", 
         "lottery@hotmail.com", 1),
        ("Your Amazon account unusual activity detected. Confirm your password here: http://amaz0n-security.com", 
         "noreply@gmail.com", 1),
        ("Microsoft security alert! Your account has been compromised. Update your password: http://micros0ft.com/secure", 
         "security@yahoo.com", 1),
        ("FINAL NOTICE: IRS tax refund pending. Click to verify SSN: http://192.168.0.1/irs", 
         "irs@gmail.com", 1),
        ("Your credit card has been suspended due to suspicious activity. Verify now: http://tinyurl.com/cc123", 
         "alerts@hotmail.com", 1),
        ("PayPal: Confirm your account number immediately or face suspension: http://paypa1-secure.com", 
         "service@gmail.com", 1),
    ]
    
    # Legitimate emails (label = 0)
    legitimate_emails = [
        ("Hi! Just wanted to check in and see how you're doing. Let me know if you want to grab coffee this week!", 
         "friend@gmail.com", 0),
        ("Your Amazon order #12345 has shipped. Track your package at amazon.com/orders", 
         "shipment-tracking@amazon.com", 0),
        ("Quarterly team meeting scheduled for next Tuesday at 2pm. See you there!", 
         "manager@company.com", 0),
        ("Here's the report you requested. Let me know if you need any changes.", 
         "colleague@company.com", 0),
        ("Your PayPal payment to John Smith was successful. Transaction ID: ABC123", 
         "service@paypal.com", 0),
        ("Newsletter: Top 10 productivity tips for remote workers", 
         "newsletter@productivityblog.com", 0),
        ("Reminder: Your dentist appointment is tomorrow at 10am", 
         "appointments@dentalcare.com", 0),
        ("Thank you for your purchase! Your receipt is attached.", 
         "receipts@store.com", 0),
    ]
    
    return phishing_emails + legitimate_emails


# ============================================================================
# MAIN DEMO
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("EMAIL PHISHING DETECTOR - ML TUTORIAL")
    print("="*60)
    
    # Step 1: Create training data
    print("\n[Step 1] Creating sample training data...")
    training_data = create_sample_training_data()
    print(f"Created {len(training_data)} sample emails")
    
    # Step 2: Create and train the detector
    print("\n[Step 2] Training the phishing detector...")
    detector = PhishingDetector()
    detector.train(training_data)
    
    # Step 3: Test on new emails
    print("\n[Step 3] Testing on new emails...\n")
    
    # Test email 1: Phishing
    test_phishing = """
    URGENT SECURITY ALERT!
    
    Your bank account will be suspended in 24 hours due to unusual activity.
    You must verify your account number and password immediately.
    
    Click here: http://192.168.1.100/bank-verify
    
    Act now or lose access to your funds!
    """
    
    print("TEST EMAIL 1:")
    print("-" * 60)
    print(test_phishing)
    detector.predict(test_phishing, "security@gmail.com", explain=True)
    
    # Test email 2: Legitimate
    test_legit = """
    Hi there,
    
    Just following up on our meeting last week. I've attached the project 
    timeline we discussed. Let me know if you have any questions!
    
    Best regards,
    Sarah
    """
    
    print("\nTEST EMAIL 2:")
    print("-" * 60)
    print(test_legit)
    detector.predict(test_legit, "sarah@company.com", explain=True)
    
    print("\n" + "="*60)
    print("NEXT STEPS FOR LEARNING:")
    print("="*60)
    print("""
    1. Try modifying the test emails and see how predictions change
    2. Add new features to extract_features() method
    3. Try different models (RandomForest, SVM) instead of DecisionTree
    4. Load a real dataset from Kaggle and train on thousands of emails
    5. Build a web interface to check emails interactively
    
    Key files to explore:
    - sklearn documentation: https://scikit-learn.org/
    - Phishing datasets: https://www.kaggle.com/datasets (search "phishing email")
    """)