import json
import smtplib
from email.mime.text import MIMEText

METRICS_PATH = "metrics/latest_metrics.json"

# Thresholds for alerting (set your own)
THRESHOLDS = {
    "accuracy": 0.85,
    "precision": 0.8,
    "recall": 0.7
}

def send_alert(message):
    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587
    SMTP_USERNAME = "your_email@gmail.com"
    SMTP_PASSWORD = "your_app_password"
    TO_EMAIL = "alert_receiver@gmail.com"

    msg = MIMEText(message)
    msg["Subject"] = "🚨 Model Performance Alert"
    msg["From"] = SMTP_USERNAME
    msg["To"] = TO_EMAIL

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.sendmail(SMTP_USERNAME, TO_EMAIL, msg.as_string())

def check_metrics():
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

    alerts = []
    for metric, value in metrics.items():
        if value < THRESHOLDS[metric]:
            alerts.append(f"{metric} dropped below threshold! Current: {value:.2f}, Threshold: {THRESHOLDS[metric]:.2f}")

    if alerts:
        alert_msg = "\n".join(alerts)
        print("ALERT:\n", alert_msg)
        send_alert(alert_msg)
        return False
    else:
        print("All metrics within thresholds.")
        return True

if __name__ == "__main__":
    check_metrics()
