"""
main.py — Entry Point
=======================
Convenience script to launch the Streamlit web app.
"""
import os
import sys

def main():
    print("Starting Automated Quiz Scanner Web Interface...")
    print("Press Ctrl+C to stop.")
    
    # Path to the streamlit app
    app_path = os.path.join(os.path.dirname(__file__), "ui", "app.py")
    
    # Run streamlit
    os.system(f'streamlit run "{app_path}"')

if __name__ == "__main__":
    main()
