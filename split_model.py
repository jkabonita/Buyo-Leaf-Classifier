#!/usr/bin/env python3
"""
Split large model file into chunks under 50MB for GitHub
"""
import os

MODEL_FILE = "buyo_best.pth"
CHUNK_SIZE = 40 * 1024 * 1024  # 40 MB chunks (safe under GitHub's 50MB limit)

def split_file():
    if not os.path.exists(MODEL_FILE):
        print(f"Error: {MODEL_FILE} not found!")
        return
    
    file_size = os.path.getsize(MODEL_FILE)
    print(f"Splitting {MODEL_FILE} ({file_size / 1024 / 1024:.2f} MB)")
    
    with open(MODEL_FILE, 'rb') as f:
        chunk_num = 0
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            
            chunk_name = f"{MODEL_FILE}.part{chunk_num:02d}"
            with open(chunk_name, 'wb') as chunk_file:
                chunk_file.write(chunk)
            
            print(f"Created {chunk_name} ({len(chunk) / 1024 / 1024:.2f} MB)")
            chunk_num += 1
    
    print(f"\nSplit into {chunk_num} parts")
    print("\nTo reassemble on Raspberry Pi, run: python3 join_model.py")

if __name__ == "__main__":
    split_file()
