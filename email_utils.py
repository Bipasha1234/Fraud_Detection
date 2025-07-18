# import os
# import smtplib
# from email.message import EmailMessage

# from dotenv import load_dotenv

# load_dotenv()  # Load GMAIL_USER and GMAIL_APP_PASS from .env
# print("EMAIL →", os.getenv("GMAIL_USER"))
# print("PASS  →", "✓" if os.getenv("GMAIL_APP_PASS") else "✗")
# EMAIL = os.getenv("GMAIL_USER")
# APP_PASSWORD = os.getenv("GMAIL_APP_PASS")

# def send_email(subject, body, to_email):
#     msg = EmailMessage()
#     msg.set_content(body)
#     msg['Subject'] = subject
#     msg['From'] = EMAIL
#     msg['To'] = to_email

#     try:
#         with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
#             smtp.login(EMAIL, APP_PASSWORD)
#             smtp.send_message(msg)
#         print(f"✅ Email sent to {to_email}")
#     except Exception as e:
#         print(f"❌ Failed to send email: {e}")
