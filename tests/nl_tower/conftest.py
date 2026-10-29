def pytest_configure(config):
    config.addinivalue_line("markers", "integration: slower end-to-end checks (a minute or two on CPU)")
