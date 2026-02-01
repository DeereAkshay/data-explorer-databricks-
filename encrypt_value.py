import os
import sys
from cryptography.fernet import Fernet

def main():
    key = os.getenv("APP_FERNET_KEY", "").strip()
    if not key:
        print("ERROR: Set APP_FERNET_KEY first.")
        print("Example (PowerShell):  $env:APP_FERNET_KEY='paste-key-here'")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Usage: python encrypt_value.py \"my-secret-value\"")
        sys.exit(1)

    f = Fernet(key.encode("utf-8"))
    plaintext = sys.argv[1].encode("utf-8")
    cipher = f.encrypt(plaintext).decode("utf-8")
    print("enc:" + cipher)

if __name__ == "__main__":
    main()



