from helpers.config import get_settings, Settings
import os
import random
import string


class BaseController:

    def __init__(self):
        self.app_settings = get_settings()

        self.base_dir = os.path.dirname(os.path.dirname(__file__))
        print('______1111________________',self.base_dir)
        self.base_dir = os.path.dirname(os.path.dirname(__file__))
        print('______22________________',self.base_dir)

        self.file_dir = os.path.join(self.base_dir, "assets", "files")

        self.database_dir = os.path.join(
            self.base_dir,
            "assets/database"
        )

    def generate_random_string(self, length: int = 12):
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

    def get_database_path(self, db_name: str):
        database_path = os.path.join(
            self.database_dir, db_name
        )

        if not os.path.exists(database_path):
            os.makedirs(database_path)

        return database_path

import os

# الحصول على المسار الأساسي للملف الحالي
base_url = os.path.dirname(os.path.abspath(__file__))

# تحديد مسار المجلد المطلوب داخل المسار الأساسي
file_dir = os.path.join(base_url, "AR_chroma_db2")

print(file_dir)  # طباعة المسار النهائي
