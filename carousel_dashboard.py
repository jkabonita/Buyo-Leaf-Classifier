"""
carousel_dashboard.py — Buyo Leaf Sorter: Carousel & System Diagnostics GUI
==============================================================================
Raspberry Pi hardware controller for the conveyor / rotary carousel sorter.
Manages:
  • NEMA stepper motor (DRV8825 / A4988) via GPIO step-dir-enable pins
  • MG996R camera servo via PCA9685 (Adafruit ServoKit, address 0x40)
  • Robotic arm servo board (PCA9685, optional address 0x41)
  • CSI / V4L2 camera presence probe

Usage (on Raspberry Pi):
    python carousel_dashboard.py

SSH + X11 forwarding (from Windows host):
    ssh -X pi@<RPI_IP> python3 /path/to/carousel_dashboard.py

SSH + XMing / VcXsrv on Windows:
    set DISPLAY=localhost:0.0
    ssh pi@<RPI_IP> "DISPLAY=localhost:0.0 python3 carousel_dashboard.py"

Headless testing (use --mock flag):
    python carousel_dashboard.py --mock
"""

import os
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox

# ── CLI flag: --mock skips all hardware imports for dry-run testing ──────────
MOCK_MODE = "--mock" in sys.argv

# ==============================================================================
# 1. HARDWARE DETECTION & PERIPHERAL PROBING
# ==============================================================================
I2C_BUS        = 1
CAMERA_CHANNEL = 15   # MG996R Camera Servo Channel on PCA9685


def probe_i2c_address(addr, bus_num=1):
    """Returns True if a device ACKs at the given I2C address."""
    if MOCK_MODE:
        return True                     # pretend hardware is present
    try:
        import smbus2
        with smbus2.SMBus(bus_num) as bus:
            bus.write_quick(addr)
        return True
    except Exception:
        return False


def probe_camera_device():
    """Checks whether the CSI / V4L2 video device node exists."""
    if MOCK_MODE:
        return True
    return os.path.exists("/dev/video0")


# ── ServoKit initialisation ──────────────────────────────────────────────────
kit            = None
pca_40_online  = False
pca_41_online  = False

if not MOCK_MODE:
    try:
        from adafruit_servokit import ServoKit
        if probe_i2c_address(0x40):
            kit = ServoKit(channels=16, address=0x40)
            kit.servo[CAMERA_CHANNEL].set_pulse_width_range(500, 2500)
            pca_40_online = True
        pca_41_online = probe_i2c_address(0x41)
    except Exception as e:
        print(f"[WARN] ServoKit init failed: {e}")

# ==============================================================================
# 2. STEPPER MOTOR HARDWARE CONFIGURATION
# ==============================================================================
DIR_PIN  = 27   # GPIO 27 — Direction
STEP_PIN = 17   # GPIO 17 — Step Pulse
EN_PIN   = 22   # GPIO 22 — Driver Enable (Active LOW)

stepper_initialized = False

if not MOCK_MODE:
    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(DIR_PIN,  GPIO.OUT)
        GPIO.setup(STEP_PIN, GPIO.OUT)
        GPIO.setup(EN_PIN,   GPIO.OUT)
        GPIO.output(EN_PIN,  GPIO.LOW)   # Enable driver
        stepper_initialized = True
    except Exception as e:
        print(f"[ERROR] GPIO Init Failed: {e}")
else:
    # In mock mode define a minimal stub so the rest of the file runs anywhere
    import types
    GPIO = types.SimpleNamespace(
        BCM=11, OUT=0, HIGH=1, LOW=0,
        setmode=lambda m: None,
        setwarnings=lambda w: None,
        setup=lambda p, m: None,
        output=lambda p, v: None,
        cleanup=lambda: None,
    )
    stepper_initialized = True

# ==============================================================================
# 3. ROTATION CALIBRATION  (Class A = Origin)
# ==============================================================================
BIN_STATIONS = {
    "A":  0,
    "B":  9677,
    "C":  22644,
    "D":  35664,   # int(33966 * 1.05) — Class D +5% offset
    "E":  47804,
}

STEP_DELAY_MIN = 0.00012   # Cruising velocity (fast)
STEP_DELAY_MAX = 0.00065   # Ramp start / stop delay (slow)

# ── Shared state ─────────────────────────────────────────────────────────────
current_step_pos    = 0
current_station_name = "A"
cam_current_angle   = 90
is_busy             = False
stop_requested      = False

# ==============================================================================
# 4. MOTION CONTROLLERS
# ==============================================================================

def move_steps(num_steps, clockwise=True):
    """Generates step pulses with S-curve acceleration ramping."""
    global stop_requested
    if num_steps <= 0:
        return

    GPIO.output(DIR_PIN, GPIO.HIGH if clockwise else GPIO.LOW)
    time.sleep(0.002)

    ramp_steps = min(int(num_steps * 0.15), 1000)

    for i in range(num_steps):
        if stop_requested:
            break

        if i < ramp_steps:
            delay = STEP_DELAY_MAX - (STEP_DELAY_MAX - STEP_DELAY_MIN) * (i / ramp_steps)
        elif i > (num_steps - ramp_steps):
            remaining = num_steps - i
            delay = STEP_DELAY_MAX - (STEP_DELAY_MAX - STEP_DELAY_MIN) * (remaining / ramp_steps)
        else:
            delay = STEP_DELAY_MIN

        if MOCK_MODE:
            # Skip the actual GPIO toggle in mock mode; just simulate timing
            time.sleep(delay * 0.001)   # 1/1000 speed to keep tests snappy
            continue

        GPIO.output(STEP_PIN, GPIO.HIGH)
        time.sleep(delay)
        GPIO.output(STEP_PIN, GPIO.LOW)
        time.sleep(delay)


def execute_rotate(target_name, update_ui_cb=None):
    """Moves the carousel to the named bin station."""
    global current_step_pos, current_station_name, stop_requested

    target_name = target_name.upper()
    if target_name not in BIN_STATIONS:
        return

    target_steps = BIN_STATIONS[target_name]
    delta_steps  = target_steps - current_step_pos

    if update_ui_cb:
        direction = "CW" if delta_steps >= 0 else "CCW"
        update_ui_cb(f"Rotating to Class {target_name} ({abs(delta_steps)} steps {direction})...")

    if delta_steps > 0:
        move_steps(delta_steps, clockwise=True)
    elif delta_steps < 0:
        move_steps(abs(delta_steps), clockwise=False)

    if not stop_requested:
        current_step_pos     = target_steps
        current_station_name = target_name


def smooth_camera_pan(target_deg, step_delay=0.015):
    """Gradually steps the MG996R servo to prevent CSI-ribbon cable shock."""
    global cam_current_angle
    if kit is None:
        return

    target_deg = max(10, min(170, int(target_deg)))
    step = 1 if target_deg >= cam_current_angle else -1

    for deg in range(cam_current_angle, target_deg + step, step):
        kit.servo[CAMERA_CHANNEL].angle = deg
        time.sleep(step_delay)

    cam_current_angle = target_deg


def release_camera_servo():
    """Cuts the PWM signal to prevent MG996R thermal stress / coil buzz."""
    if kit is not None:
        kit.servo[CAMERA_CHANNEL].angle = None


# ==============================================================================
# 5. TKINTER GUI APPLICATION
# ==============================================================================

class CarouselApp(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title("Buyo Leaf Sorter — Carousel & System Diagnostics")
        self.geometry("660x680")
        self.resizable(False, False)
        self.configure(bg="#1E1E2E")

        # Dracula-inspired colour palette
        self._BG      = "#1E1E2E"
        self._SURFACE = "#282A36"
        self._COMMENT = "#6272A4"
        self._GREEN   = "#50FA7B"
        self._CYAN    = "#8BE9FD"
        self._YELLOW  = "#F1FA8C"
        self._PURPLE  = "#BD93F9"
        self._ORANGE  = "#FFB86C"
        self._PINK    = "#FF79C6"
        self._RED     = "#FF5555"
        self._FG      = "#F8F8F2"

        self.protocol("WM_DELETE_WINDOW", self.safe_exit_to_class_a)
        self._build_ui()
        self.refresh_peripherals()

    # ──────────────────────────────────────────────────────────────────────────
    # UI CONSTRUCTION
    # ──────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Header ────────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=self._SURFACE, pady=10)
        hdr.pack(fill=tk.X)

        mode_tag = "  [MOCK MODE]" if MOCK_MODE else ""
        tk.Label(
            hdr,
            text=f"🌿  BUYO LEAF SORTER  —  SYSTEM DASHBOARD{mode_tag}",
            font=("Arial", 13, "bold"),
            fg=self._GREEN,
            bg=self._SURFACE,
        ).pack()

        # ── Peripheral Status Panel ───────────────────────────────────────────
        periph_frame = tk.LabelFrame(
            self,
            text=" Connected Peripherals ",
            font=("Arial", 9, "bold"),
            fg=self._CYAN,
            bg=self._BG,
            padx=10,
            pady=6,
        )
        periph_frame.pack(fill=tk.X, padx=15, pady=6)

        self.status_badges = {}
        items = [
            ("camera",    "📷  CSI Camera Module",          0, 0),
            ("stepper",   "⚙️  Stepper Motor (Carousel)",   0, 1),
            ("servo_cam", "🎯  Camera Servo (MG996R)",       1, 0),
            ("arm",       "🦾  Robotic Arm (PCA9685)",       1, 1),
        ]

        for key, name, row, col in items:
            box = tk.Frame(periph_frame, bg=self._SURFACE, padx=8, pady=4,
                           relief=tk.RIDGE, bd=1)
            box.grid(row=row, column=col, padx=5, pady=4, sticky="ew")
            periph_frame.grid_columnconfigure(col, weight=1)

            tk.Label(box, text=name, font=("Arial", 9, "bold"),
                     fg=self._FG, bg=self._SURFACE).pack(anchor="w")

            badge = tk.Label(box, text="Checking…", font=("Arial", 9),
                             fg=self._YELLOW, bg=self._SURFACE)
            badge.pack(anchor="w")
            self.status_badges[key] = badge

        tk.Button(
            periph_frame,
            text="🔄  Scan Peripherals",
            font=("Arial", 9, "bold"),
            bg=self._COMMENT,
            fg=self._FG,
            activebackground=self._PINK,
            command=self.refresh_peripherals,
        ).grid(row=2, column=0, columnspan=2, pady=5, sticky="ew", padx=5)

        # ── Carousel Telemetry ────────────────────────────────────────────────
        telem_frame = tk.LabelFrame(
            self,
            text=" Carousel Position Telemetry ",
            font=("Arial", 9, "bold"),
            fg=self._CYAN,
            bg=self._BG,
            padx=12,
            pady=6,
        )
        telem_frame.pack(fill=tk.X, padx=15, pady=4)

        self.lbl_station = tk.Label(
            telem_frame,
            text="Active Station: Class A  (Origin)",
            font=("Arial", 12, "bold"),
            fg=self._GREEN,
            bg=self._BG,
        )
        self.lbl_station.pack(anchor="w")

        self.lbl_steps = tk.Label(
            telem_frame,
            text="Absolute Pulse Position: 0 steps",
            font=("Arial", 10),
            fg=self._PURPLE,
            bg=self._BG,
        )
        self.lbl_steps.pack(anchor="w", pady=1)

        self.lbl_status = tk.Label(
            telem_frame,
            text="Status: Ready at Origin (Class A)",
            font=("Arial", 10, "italic"),
            fg=self._YELLOW,
            bg=self._BG,
        )
        self.lbl_status.pack(anchor="w", pady=1)

        # ── Direct Bin Buttons ────────────────────────────────────────────────
        btn_frame = tk.LabelFrame(
            self,
            text=" Direct Indexing Controls  (Class A = Origin) ",
            font=("Arial", 9, "bold"),
            fg=self._CYAN,
            bg=self._BG,
            padx=8,
            pady=6,
        )
        btn_frame.pack(fill=tk.X, padx=15, pady=4)

        bins = [
            ("Class A  (Origin / 0)",  "A", "#2E7D32"),
            ("Class B  (Base)",         "B", self._COMMENT),
            ("Class C  (-10 %)",        "C", self._COMMENT),
            ("Class D  (+5 %)",         "D", self._COMMENT),
            ("Class E  (-5 %)",         "E", self._COMMENT),
        ]

        self.buttons = {}
        for idx, (label_text, code, color) in enumerate(bins):
            btn = tk.Button(
                btn_frame,
                text=label_text,
                font=("Arial", 10, "bold"),
                bg=color,
                fg=self._FG,
                activebackground=self._PINK,
                width=18,
                height=2,
                command=lambda c=code: self.threaded_task(self.task_rotate_to, c),
            )
            row, col = divmod(idx, 3)
            btn.grid(row=row, column=col, padx=4, pady=3)
            self.buttons[code] = btn

        # ── Automation & Diagnostics ──────────────────────────────────────────
        auto_frame = tk.LabelFrame(
            self,
            text=" Automation & Diagnostics ",
            font=("Arial", 9, "bold"),
            fg=self._CYAN,
            bg=self._BG,
            padx=8,
            pady=6,
        )
        auto_frame.pack(fill=tk.X, padx=15, pady=6)

        self.btn_sort_cycle = tk.Button(
            auto_frame,
            text="Sort Cycle  (Bin -> Return)",
            font=("Arial", 10, "bold"),
            bg=self._ORANGE,
            fg=self._SURFACE,
            height=2,
            command=self.prompt_sort_cycle,
        )
        self.btn_sort_cycle.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=3)

        self.btn_sweep = tk.Button(
            auto_frame,
            text="Full Carousel Sweep",
            font=("Arial", 10, "bold"),
            bg=self._PURPLE,
            fg=self._SURFACE,
            height=2,
            command=lambda: self.threaded_task(self.task_automated_sweep),
        )
        self.btn_sweep.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=3)

        self.btn_cam_test = tk.Button(
            auto_frame,
            text="Test Camera Servo",
            font=("Arial", 10, "bold"),
            bg=self._GREEN,
            fg=self._SURFACE,
            height=2,
            command=lambda: self.threaded_task(self.task_test_camera_servo),
        )
        self.btn_cam_test.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=3)

        # ── ESTOP / Exit ──────────────────────────────────────────────────────
        exit_frame = tk.Frame(self, bg=self._BG, pady=4)
        exit_frame.pack(fill=tk.X, padx=15, pady=4)

        self.btn_exit = tk.Button(
            exit_frame,
            text="REVERT TO CLASS A  (ORIGIN)  &  EXIT",
            font=("Arial", 11, "bold"),
            bg=self._RED,
            fg=self._FG,
            activebackground="#FF6E6E",
            height=2,
            command=self.safe_exit_to_class_a,
        )
        self.btn_exit.pack(fill=tk.X)

    # ──────────────────────────────────────────────────────────────────────────
    # PERIPHERAL DIAGNOSTICS
    # ──────────────────────────────────────────────────────────────────────────

    def refresh_peripherals(self):
        """Re-probes all I2C, GPIO, and camera nodes and updates status badges."""
        global kit

        # Camera
        if probe_camera_device():
            self.status_badges["camera"].config(
                text="CONNECTED  (/dev/video0)", fg=self._GREEN)
        else:
            self.status_badges["camera"].config(
                text="OFFLINE  (no /dev/video0)", fg=self._RED)

        # Stepper motor
        if stepper_initialized:
            self.status_badges["stepper"].config(
                text=f"READY  (GPIO {STEP_PIN}/{DIR_PIN}/{EN_PIN})", fg=self._GREEN)
        else:
            self.status_badges["stepper"].config(
                text="GPIO ERROR", fg=self._RED)

        # Camera servo via PCA9685 @ 0x40
        pca_40 = probe_i2c_address(0x40)
        if pca_40:
            if kit is None and not MOCK_MODE:
                try:
                    from adafruit_servokit import ServoKit
                    kit = ServoKit(channels=16, address=0x40)
                    kit.servo[CAMERA_CHANNEL].set_pulse_width_range(500, 2500)
                except Exception:
                    pass
            self.status_badges["servo_cam"].config(
                text=f"ONLINE  (PCA 0x40 / Ch {CAMERA_CHANNEL})", fg=self._GREEN)
        else:
            self.status_badges["servo_cam"].config(
                text="OFFLINE  (0x40 unreachable)", fg=self._RED)

        # Robotic arm (dual / single driver)
        pca_41 = probe_i2c_address(0x41)
        if pca_40 and pca_41:
            self.status_badges["arm"].config(
                text="DUAL DRIVER  (0x40 & 0x41)", fg=self._GREEN)
        elif pca_40:
            self.status_badges["arm"].config(
                text="PRIMARY DRIVER  (0x40 only)", fg=self._YELLOW)
        elif pca_41:
            self.status_badges["arm"].config(
                text="SECONDARY DRIVER  (0x41 only)", fg=self._YELLOW)
        else:
            self.status_badges["arm"].config(
                text="OFFLINE  (no I2C response)", fg=self._RED)

    # ──────────────────────────────────────────────────────────────────────────
    # UI HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    def set_controls_state(self, enabled=True):
        state = tk.NORMAL if enabled else tk.DISABLED
        for btn in self.buttons.values():
            btn.config(state=state)
        self.btn_sort_cycle.config(state=state)
        self.btn_sweep.config(state=state)
        self.btn_cam_test.config(state=state)
        self.btn_exit.config(state=state)

    def update_telemetry(self, status_msg="Ready"):
        origin_tag = "  (Origin)" if current_station_name == "A" else ""
        self.lbl_station.config(
            text=f"Active Station: Class {current_station_name}{origin_tag}")
        self.lbl_steps.config(
            text=f"Absolute Pulse Position: {current_step_pos} steps")
        self.lbl_status.config(text=f"Status: {status_msg}")

    def _set_status(self, msg):
        """Thread-safe status label update."""
        self.after(0, lambda: self.lbl_status.config(text=f"Status: {msg}"))

    # ──────────────────────────────────────────────────────────────────────────
    # ASYNC THREAD DISPATCHER
    # ──────────────────────────────────────────────────────────────────────────

    def threaded_task(self, target_fn, *args):
        global is_busy
        if is_busy:
            return
        is_busy = True
        self.set_controls_state(False)
        threading.Thread(
            target=self._task_wrapper,
            args=(target_fn, *args),
            daemon=True,
        ).start()

    def _task_wrapper(self, target_fn, *args):
        global is_busy
        try:
            target_fn(*args)
        finally:
            is_busy = False
            self.after(0, lambda: self.set_controls_state(True))
            self.after(0, self.update_telemetry)

    # ──────────────────────────────────────────────────────────────────────────
    # AUTOMATION ROUTINES
    # ──────────────────────────────────────────────────────────────────────────

    def task_rotate_to(self, target_name):
        execute_rotate(target_name,
                       update_ui_cb=lambda m: self._set_status(m))

    def task_sort_cycle(self, bin_target):
        if bin_target != "A":
            self._set_status(f"Indexing to Bin {bin_target}...")
            execute_rotate(bin_target)
        self._set_status(f"Dispensing into Bin {bin_target} (1.2 s)...")
        time.sleep(1.2)
        if bin_target != "A":
            self._set_status("Resetting to Class A (Origin)...")
            execute_rotate("A")

    def prompt_sort_cycle(self):
        dlg = tk.Toplevel(self)
        dlg.title("Select Bin")
        dlg.geometry("280x190")
        dlg.configure(bg=self._SURFACE)
        dlg.transient(self)
        dlg.grab_set()

        tk.Label(dlg, text="Select Target Bin for Sort Cycle:",
                 fg=self._FG, bg=self._SURFACE,
                 font=("Arial", 10, "bold")).pack(pady=12)

        combo = ttk.Combobox(dlg, values=list(BIN_STATIONS.keys()),
                             state="readonly", font=("Arial", 11))
        combo.set("B")
        combo.pack(pady=8)

        def _confirm():
            target = combo.get()
            dlg.destroy()
            self.threaded_task(self.task_sort_cycle, target)

        tk.Button(dlg, text="Start Sort Cycle",
                  bg=self._GREEN, fg=self._SURFACE,
                  font=("Arial", 10, "bold"),
                  command=_confirm).pack(pady=12)

    def task_automated_sweep(self):
        """Visits every bin station in sequence and returns to origin."""
        execute_rotate("A", update_ui_cb=lambda m: self._set_status(m))
        time.sleep(0.5)
        for b in BIN_STATIONS:
            execute_rotate(b, update_ui_cb=lambda m, n=b: self._set_status(f"Sweep -> Bin {n}..."))
            time.sleep(1.0)
        execute_rotate("A", update_ui_cb=lambda m: self._set_status("Returning to Class A..."))

    def task_test_camera_servo(self):
        """Diagnostic pan sequence: centre -> left -> right -> centre."""
        if kit is None and not MOCK_MODE:
            self._set_status("ERROR - PCA9685 driver offline!")
            time.sleep(1.5)
            return

        steps = [
            ("Centering to 90 deg",     90,  0.015),
            ("Panning Left  (35 deg)",  35,  0.012),
            ("Panning Right (145 deg)", 145, 0.012),
            ("Returning to 90 deg",     90,  0.012),
        ]
        for msg, deg, delay in steps:
            self._set_status(f"Camera Test: {msg}")
            smooth_camera_pan(deg, step_delay=delay)
            time.sleep(0.8 if deg == 90 else 1.0)

        release_camera_servo()
        self._set_status("Camera servo test complete (torque released).")

    # ──────────────────────────────────────────────────────────────────────────
    # SAFE SHUTDOWN
    # ──────────────────────────────────────────────────────────────────────────

    def safe_exit_to_class_a(self):
        global is_busy, stop_requested
        if is_busy:
            messagebox.showwarning(
                "Motion in Progress",
                "Carousel is currently moving. Please wait before exiting.")
            return

        is_busy = True
        self.set_controls_state(False)
        self._set_status("SHUTDOWN TRIGGERED - returning to Class A...")

        def _exit_worker():
            release_camera_servo()
            print("\n[EXIT] Returning carousel to Class A (Origin)...")
            execute_rotate("A",
                           update_ui_cb=lambda m: self._set_status(m))
            time.sleep(0.5)

            if not MOCK_MODE:
                print("[CLEANUP] Releasing GPIO...")
                GPIO.output(STEP_PIN, GPIO.LOW)
                GPIO.output(DIR_PIN,  GPIO.LOW)
                GPIO.cleanup()

            print("[DONE] GPIO released. Closing window.")
            self.after(0, self.destroy)

        threading.Thread(target=_exit_worker, daemon=True).start()


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    if MOCK_MODE:
        print("[INFO] Running in MOCK MODE - no real GPIO / I2C required.")
    app = CarouselApp()
    app.mainloop()
