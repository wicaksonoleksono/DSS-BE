import os
import firebase_admin
from firebase_admin import credentials

class Config:
    @staticmethod
    def init_firebase():
        try:
            # Try to get the default app
            firebase_admin.get_app()
            print("Firebase app already initialized")
        except ValueError:
            # The default app is not initialized yet, initialize it
            current_dir = os.path.dirname(os.path.abspath(__file__))
            cred_path = os.path.join(current_dir, '..', '..', 'db', 'FirebaseCred.json')

            print(f"Initializing Firebase app with credentials at: {cred_path}")

            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
