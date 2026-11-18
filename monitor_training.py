"""
monitor_training.py - Monitor the training progress
Usage: python monitor_training.py
"""
import time
import os
from pathlib import Path

LOG_FILE = "training.log"
CHECK_INTERVAL = 10  # seconds

def tail_file(filepath, n=50):
    """Read last n lines of a file"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            return ''.join(lines[-n:])
    except FileNotFoundError:
        return "Log file not found yet..."
    except Exception as e:
        return f"Error reading log: {e}"

def check_checkpoint():
    """Check if checkpoint exists and its modification time"""
    ckpt_path = Path("buyo_best.pth")
    if ckpt_path.exists():
        mtime = ckpt_path.stat().st_mtime
        mod_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
        size_mb = ckpt_path.stat().st_size / (1024 * 1024)
        return f"✓ Checkpoint exists: {size_mb:.1f} MB (modified: {mod_time})"
    else:
        return "✗ No checkpoint yet"

def main():
    print("=" * 70)
    print("  BUYO LEAF TRAINING MONITOR")
    print("=" * 70)
    print("  Press Ctrl+C to exit")
    print("=" * 70)
    
    try:
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print("=" * 70)
            print("  BUYO LEAF TRAINING MONITOR")
            print("=" * 70)
            print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  {check_checkpoint()}")
            print("=" * 70)
            print("\nLast 30 lines of training output:\n")
            
            # Try to read from process output if available
            # For now, just show a message
            print("Training is running in the background...")
            print("Check the Kiro terminal output for real-time progress.")
            print("\nTo view full training progress:")
            print("  1. Check the Kiro terminal tab")
            print("  2. Look for the training process output")
            
            print("\n" + "=" * 70)
            print(f"  Next update in {CHECK_INTERVAL} seconds...")
            print("  Press Ctrl+C to exit")
            print("=" * 70)
            
            time.sleep(CHECK_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

if __name__ == "__main__":
    main()
