import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from arbor.server.api.routes.files import router
from arbor.server.services.file_manager import FileManager

@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(router, prefix="/api/files")
    return app

@pytest.fixture
def client(app):
    return TestClient(app)

def test_upload_file(client):
    # Create a test file
    test_content = b"test file content"
    files = {"file": ("test.txt", test_content, "text/plain")}

    response = client.post("/api/files", files=files)

    assert response.status_code == 200
    assert "id" in response.json()