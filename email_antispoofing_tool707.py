import os
import re
import logging
import joblib
import dns.resolver
import spf
import dkim
import hashlib
from datetime import datetime
from sklearn.ensemble import IsolationForest
from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Configuration Class
class Config:
    MODEL_PATH = os.getenv("MODEL_PATH", "phishing_model.pkl")
    VECTORIZER_PATH = os.getenv("VECTORIZER_PATH", "tfidf_vectorizer.pkl")
    ANOMALY_THRESHOLD = float(os.getenv("ANOMALY_THRESHOLD", "-0.5"))
    PHISHING_THRESHOLD = float(os.getenv("PHISHING_THRESHOLD", "0.8"))
    DKIM_SELECTOR = os.getenv("DKIM_SELECTOR", "default")
    QUARANTINE_DIR = os.getenv("QUARANTINE_DIR", "/var/quarantine")

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Load security models
try:
    phishing_model = joblib.load(Config.MODEL_PATH)
    vectorizer = joblib.load(Config.VECTORIZER_PATH)
    isolation_forest = IsolationForest(contamination=0.1)
    logger.info("Security models loaded successfully")
except Exception as e:
    logger.critical(f"Failed to load models: {str(e)}")
    raise

# Email Processing Utilities
class EmailProcessor:
    @staticmethod
    def preprocess_email(content: str) -> str:
        """Sanitize and normalize email content"""
        cleaned = re.sub(r'<.*?>', '', content)
        cleaned = re.sub(r'http[s]?://\S+', '[URL]', cleaned)
        cleaned = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '[EMAIL]', cleaned)
        return cleaned.lower().strip()

    @staticmethod
    def generate_email_hash(content: str) -> str:
        """Create secure hash for email identification"""
        return hashlib.sha256(content.encode()).hexdigest()

# Security Validators
class SecurityValidator:
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    @staticmethod
    def validate_spf(sender_ip: str, sender_domain: str) -> bool:
        """Validate SPF record with retry logic"""
        try:
            result = spf.check2(i=sender_ip, s=sender_domain, h="localhost")
            return result[0] == 'pass'
        except Exception as e:
            logger.warning(f"SPF validation failed: {str(e)}")
            return False

    @staticmethod
    def validate_dkim(headers: str) -> bool:
        """Verify DKIM signature"""
        try:
            return dkim.verify(headers) is not None
        except Exception as e:
            logger.warning(f"DKIM validation failed: {str(e)}")
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    @staticmethod
    def validate_dmarc(domain: str) -> bool:
        """Check DMARC policy with DNS retries"""
        try:
            answers = dns.resolver.resolve(f'_dmarc.{domain}', 'TXT')
            return any('v=DMARC1' in str(r) for r in answers)
        except dns.resolver.NXDOMAIN:
            logger.warning(f"No DMARC record found for {domain}")
            return False
        except Exception as e:
            logger.error(f"DMARC check failed: {str(e)}")
            return False

# Threat Detection
class ThreatDetector:
    @staticmethod
    def detect_phishing(content: str) -> bool:
        """Detect phishing attempts with ML model"""
        try:
            processed = EmailProcessor.preprocess_email(content)
            features = vectorizer.transform([processed])
            return phishing_model.predict_proba(features)[0][1] > Config.PHISHING_THRESHOLD
        except Exception as e:
            logger.error(f"Phishing detection failed: {str(e)}")
            return False

    @staticmethod
    def detect_anomalies(metadata: list) -> bool:
        """Identify unusual email patterns"""
        try:
            return isolation_forest.decision_function([metadata])[0] < Config.ANOMALY_THRESHOLD
        except Exception as e:
            logger.error(f"Anomaly detection failed: {str(e)}")
            return False

# Quarantine Management
class QuarantineManager:
    @staticmethod
    def quarantine_email(content: str, reason: str):
        """Securely store suspicious emails"""
        try:
            email_hash = EmailProcessor.generate_email_hash(content)
            filename = f"{datetime.now().isoformat()}_{email_hash[:16]}.eml"
            path = os.path.join(Config.QUARANTINE_DIR, filename)
            
            with open(path, 'w') as f:
                f.write(f"// Quarantine Reason: {reason}\n")
                f.write(content)
            
            logger.info(f"Quarantined email: {filename}")
            return True
        except Exception as e:
            logger.error(f"Quarantine failed: {str(e)}")
            return False

# API Endpoints
@app.route('/api/analyze', methods=['POST'])
def analyze_email():
    try:
        email_data = request.json
        required_fields = {'content', 'sender_ip', 'sender_domain', 'headers'}
        
        if missing := required_fields - set(email_data.keys()):
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        # Validate email authentication
        auth_checks = {
            "spf": SecurityValidator.validate_spf(
                email_data['sender_ip'], 
                email_data['sender_domain']
            ),
            "dkim": SecurityValidator.validate_dkim(email_data['headers']),
            "dmarc": SecurityValidator.validate_dmarc(email_data['sender_domain'])
        }

        # Threat detection
        threats = {
            "phishing": ThreatDetector.detect_phishing(email_data['content']),
            "anomaly": ThreatDetector.detect_anomalies(
                parse_metadata(email_data.get('metadata', []))
            )
        }

        # Build response
        response = {
            "email_id": EmailProcessor.generate_email_hash(email_data['content']),
            "authentication": auth_checks,
            "threats": threats,
            "actions": []
        }

        # Take mitigation actions
        if any(threats.values()) or not all(auth_checks.values()):
            if threats["phishing"]:
                QuarantineManager.quarantine_email(
                    email_data['content'], 
                    "Phishing detected"
                )
                response["actions"].append("quarantined")
            
            if not auth_checks["spf"]:
                response["actions"].append("marked_as_spam")
            
            if threats["anomaly"]:
                response["actions"].append("flagged_for_review")

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return jsonify({"error": "Email analysis failed"}), 500

@app.route('/dashboard')
def security_dashboard():
    return render_template('dashboard.html')

@app.route('/api/threats')
def get_threat_data():
    # Implement real data collection here
    return jsonify({
        "phishing_attempts": 12,
        "spoofing_attempts": 5,
        "quarantined_emails": 8
    })

# Helper functions
def parse_metadata(raw_metadata: list) -> list:
    """Convert metadata to numerical features"""
    try:
        return [float(x) for x in raw_metadata[:10]]  # Use first 10 features
    except ValueError:
        return [0] * 10

if __name__ == '__main__':
    # Ensure quarantine directory exists
    os.makedirs(Config.QUARANTINE_DIR, exist_ok=True)

    # Train anomaly detection model with synthetic metadata
    synthetic_metadata = np.random.rand(1000, 10)  

    try:
        isolation_forest.fit(synthetic_metadata)
        logger.info("Anomaly detection model trained with synthetic data.")
    except Exception as e:
        logger.error(f"Anomaly model training failed: {str(e)}")

    # Start Flask app
    app.run(host='0.0.0.0', port=8080, debug=False)
