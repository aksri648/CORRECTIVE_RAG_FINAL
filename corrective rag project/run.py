import os
import sys
import signal
import subprocess
from dotenv import load_dotenv
from pyngrok import ngrok, exception as pyngrok_exc


load_dotenv()

STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN")
NGROK_DOMAIN = os.getenv("NGROK_DOMAIN")


def _kill_existing_tunnel():
    try:
        ngrok.kill()
    except Exception:
        pass


def _shutdown(streamlit_proc, tunnels):
    if streamlit_proc and streamlit_proc.poll() is None:
        streamlit_proc.terminate()
        try:
            streamlit_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            streamlit_proc.kill()

    for tunnel in tunnels or []:
        try:
            ngrok.disconnect(tunnel.public_url)
        except Exception:
            pass
    _kill_existing_tunnel()


def main():
    if not NGROK_AUTH_TOKEN:
        print("ERROR: NGROK_AUTH_TOKEN is not set. Add it to your .env file.")
        print("       Get a free token at https://dashboard.ngrok.com/get-started/your-authtoken")
        sys.exit(1)

    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    _kill_existing_tunnel()

    connect_kwargs = {"addr": f"http://localhost:{STREAMLIT_PORT}", "proto": "http"}
    if NGROK_DOMAIN:
        connect_kwargs["domain"] = NGROK_DOMAIN

    try:
        tunnel = ngrok.connect(**connect_kwargs)
    except pyngrok_exc.PyngrokError as exc:
        print(f"ERROR: failed to open ngrok tunnel: {exc}")
        sys.exit(1)

    print("=" * 64)
    print(f"  Public URL:  {tunnel.public_url}")
    print(f"  Local URL:   http://localhost:{STREAMLIT_PORT}")
    print("=" * 64)
    print("  Press Ctrl+C to stop both Streamlit and the ngrok tunnel.\n")

    streamlit_proc = None
    try:
        streamlit_proc = subprocess.Popen(
            [
                "streamlit", "run", "app.py",
                "--server.port", str(STREAMLIT_PORT),
                "--server.headless", "true",
                "--server.address", "0.0.0.0",
                "--browser.gatherUsageStats", "false",
            ],
            cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
        )
        streamlit_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        _shutdown(streamlit_proc, ngrok.get_tunnels())


def _signal_handler(sig, frame):
    raise KeyboardInterrupt


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


if __name__ == "__main__":
    main()
