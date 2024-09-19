from subprocess import Popen

def load_jupyter_server_extension(nbapp):
    """serve the streamlit app"""
    Popen(
        [
            "streamlit",
            "run",
            "🏠_Home.py",
            "--browser.serverAddress=0.0.0.0",
            "--server.enableCORS=False",
            "--server.enableWebsocketCompression=False",  # Disable websocket compression
            "--server.enableXsrfProtection=False",  # Disable XSRF protection for local development
        ]
    )
