import json
import smtplib

# Features Configurations
with open( "./utils/config.json", "r" ) as file:
    config = json.load( file )

class Mailer:
    """
        Class to initiate the email alert function.
    """

    def __init__(self):
        self.email = config["Email_Send"]
        self.password = config["Email_Password"]
        self.port = 465
        # Establish a secure connection to Simple Mail Transfer Protocol
        self.server = smtplib.SMTP_SSL( 'smtp.gmail.com', self.port )

    def send( self, mail ):
        self.server = smtplib.SMTP_SSL( 'smtp.gmail.com', self.port )
        self.server.login( self.email, self.password )
        # Message to send
        SUBJECT = 'ALERT!'
        TEXT = f'Too many people in one area!'
        message = 'Subject: {}\n\n{}'.format( SUBJECT, TEXT )
        # Send email
        self.server.sendmail( self.email, mail, message )
        self.server.quit()