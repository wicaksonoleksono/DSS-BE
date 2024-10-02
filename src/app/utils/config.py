import firebase_admin
from firebase_admin import credentials


# ini singleton pattern karena init cuma sekali
class Config:
    _initialize = False

    @staticmethod
    def init_firebase():
        if not Config._initialize:
            cred = credentials.Certificate("../BE/FirebaseCred.json")
            firebase_admin.initialize_app(cred)
            Config._initialize = True
