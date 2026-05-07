# conftest.py
from pathlib import Path
from datetime import datetime
import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--discovery",
        action="store_true",
        default=False,
        help="Run discovery mode instead of equation comparison",
    )
    parser.addoption(
        "--report",
        action="store_true",
        default=False,
        help="Save discovery report files",
    )
    parser.addoption(
        "--report-dir",
        action="store",
        default="reports",
        help="Base directory for reports",
    )

@pytest.fixture
def runtime_options(request):
    return {
        "discovery": request.config.getoption("--discovery"),
        "report": request.config.getoption("--report"),
        "report_dir": Path(request.config.getoption("--report-dir")),
    }

def pytest_configure(config):
    config.addinivalue_line("markers", "functional: functional tests")
    config.addinivalue_line("markers", "discovery: discovery tests")