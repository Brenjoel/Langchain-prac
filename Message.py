from dotenv import load_dotenv

load_dotenv()

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import os

sender_email = os.getenv("SENDER_EMAIL")
app_password = os.getenv("SENDER_EMAIL_PASSWORD")

receiver_email =os.getenv("RECIPENT_EMAIL")

# print(sender_email, app_password , receiver_email)

msg = MIMEMultipart()
msg["From"] = sender_email
msg["To"] = receiver_email
msg["Subject"] = "Test Email from Python"

body = "Hello! This email was sent using Python."
msg.attach(MIMEText(body, "plain"))

with smtplib.SMTP("smtp.gmail.com", 587) as server:
    server.starttls()
    server.login(sender_email, app_password)
    server.send_message(msg)

print("Email sent successfully!")