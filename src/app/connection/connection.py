from app.utils.config import Config
from firebase_admin import firestore


# pakai logic ini factory method / abstrck fact method
class Connection:
    @staticmethod
    def get_collection(collection_name: str):
        Config.init_firebase()
        db = firestore.client()
        return db.collection(collection_name)
