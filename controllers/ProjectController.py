from .BaseController import BaseController
from fastapi import UploadFile
from models import ResponseSignal
import os


class ProjectController(BaseController):

    def __int__(self):
        super().__init__()

    def get_project_path(self, project_id: str):
        project_dir = os.path.join(
            self.file_dir,
            project_id
        )
        print("________project_dir", project_dir)
        if not os.path.exists(project_dir):
            os.makedirs(project_dir, exist_ok=True)
        return project_dir
