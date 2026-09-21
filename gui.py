    import os
    import time
    import threading
    import tkinter as tk
    from tkinter import ttk, messagebox
    import RPi.GPIO as GPIO

    # ==============================================================================
    # 1. HARDWARE DETECTION & SERVO DRIVER
    # ==============================================================================
    I2C_BUS = 1
    CAMERA_CHANNEL = 15  # MG996R Camera Servo Channel on PCA9685 (0x40)

    def probe_i2c_address(addr, bus_num=1):
        try:
            import smbus2
            with smbus2.SMBus(bus_num) as bus:
                bus.write_quick(addr)
            return True
        except Exception:
            return False

    def probe_camera_device():
        return os.path.exists('/dev/video0')

    kit = None
    pca_40_online = False
    pca_41_online = False

    try:
        from adafruit_servokit import ServoKit
        if probe_i2c_address(0x40):
            kit = ServoKit(channels=16, address=0x40)
            # Using continuous_servo doesn't strictly need pulse_width_range, 
            # but we keep it safe for the standard mode fallback.
            kit.servo[CAMERA_CHANNEL].set_pulse_width_range(500, 2500)
            pca_40_online = True
        pca_41_online = probe_i2c_address(0x41)
    except Exception as e:
        print(f"[WARN] ServoKit init failed: {e}")

    # ==============================================================================
    # 2. STEPPER MOTOR GPIO CONFIGURATION
    # ==============================================================================
    DIR_PIN  = 27
    STEP_PIN = 17
    EN_PIN   = 22

    stepper_initialized = False
    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(DIR_PIN, GPIO.OUT)
        GPIO.setup(STEP_PIN, GPIO.OUT)
        GPIO.setup(EN_PIN, GPIO.OUT)
        GPIO.output(EN_PIN, GPIO.LOW)  # Active LOW
        stepper_initialized = True
    except Exception as e:
        print(f"[ERROR] GPIO Init Failed: {e}")

    # ==============================================================================
    # 3. MOTION & POSITION CALIBRATION
    # ==============================================================================
    BIN_STATIONS = {
        'A': 0,
        'B': 9677,
        'C': 22644,
        'D': 35664,
        'E': 47804
    }

    # ==============================================================================
    # PER-ARM PER-CLASS CALIBRATION  —  Explicit disk positions
    # ==============================================================================
    # CALIBRATED_POSITIONS[arm_num][class] = absolute disk step position.
    # Each (arm, class) entry is stored independently — modifying Arm 2 / Class C
    # never touches any other arm or class.
    # Populated at runtime by the calibration panel and persisted to calibration.json.
    # get_disk_target_steps() checks this dict first; if no entry exists it falls
    # back to the BIN_STATIONS + ARM_STEPPER_OFFSETS formula.
    CALIBRATED_POSITIONS = {}   # {arm_num: {'A': steps, 'B': steps, ...}}

    # Path to the JSON file that persists calibration across sessions
    CALIBRATION_FILE = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "calibration.json"
    )


    def load_calibration():
        """
        Loads calibration from calibration.json (if it exists) into
        CALIBRATED_POSITIONS and current_step_pos.

        current_step_pos is restored so the system knows where the disk
        physically is on startup (assuming a clean exit was performed, which
        homes the disk to Class A and then saves that position).
        """
        import json
        global CALIBRATED_POSITIONS, current_step_pos
        if not os.path.exists(CALIBRATION_FILE):
            print("[CALIB] No calibration.json found — starting from step 0.")
            return
        try:
            with open(CALIBRATION_FILE, 'r') as f:
                data = json.load(f)

            # Restore per-arm per-class positions
            raw_cp = data.get("calibrated_positions", {})
            CALIBRATED_POSITIONS = {
                int(arm): {cls: int(steps) for cls, steps in classes.items()}
                for arm, classes in raw_cp.items()
            }

            # Restore last known disk position so relative moves are correct
            if "last_step_pos" in data:
                current_step_pos = int(data["last_step_pos"])
                print(f"[CALIB] Restored disk position: {current_step_pos} steps")

            print(f"[CALIB] Loaded calibration from {CALIBRATION_FILE}")
        except Exception as e:
            print(f"[CALIB] Failed to load calibration.json: {e}")


    def save_calibration():
        """
        Persists CALIBRATED_POSITIONS and current_step_pos to calibration.json.
        current_step_pos is saved so the next launch restores the exact disk
        position without needing a home sensor.
        """
        import json
        try:
            data = {
                "calibrated_positions": {
                    str(arm): classes
                    for arm, classes in CALIBRATED_POSITIONS.items()
                },
                "last_step_pos": current_step_pos,
            }
            with open(CALIBRATION_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"[CALIB] Saved calibration to {CALIBRATION_FILE} (disk pos: {current_step_pos})")
        except Exception as e:
            print(f"[CALIB] Failed to save calibration.json: {e}")

    # ==============================================================================
    # CAROUSEL DISK GEOMETRY  —  Open-loop, no sensor positioning
    # ==============================================================================
    #
    #  The rotating disk has 5 bin sectors (A, B, C, D, E).
    #  9 robotic arms are fixed around the outside, evenly spaced.
    #
    #  Physical layout (from photo):
    #    • Disk rotates clockwise (positive step direction)
    #    • 9 arms at 40° intervals  (360° ÷ 9 = 40° per arm)
    #    • 5 sectors at 72° each   (360° ÷ 5 = 72° per sector)
    #
    #  Formula to put bin X under arm N (no sensor):
    #    target_steps = BIN_STATIONS[X] + (N − 1) × STEPS_PER_ARM
    #    (then take modulo CAROUSEL_FULL_STEPS for wrap-around)
    #
    # ==============================================================================
    # CALIBRATION STEP  —  measure CAROUSEL_FULL_STEPS:
    #   1. Home stepper to Class A, arm #1 (steps = 0)
    #   2. Run stepper CW until sector A visually returns to Arm #1
    #   3. Read the step counter from the UI telemetry
    #   4. Enter that value below as CAROUSEL_FULL_STEPS
    # ==============================================================================
    CAROUSEL_FULL_STEPS = 57804   # TODO: replace with measured full-revolution step count
                                # Estimated: BIN_STATIONS['E'] + gap_E_to_A ≈ 47804 + 10000

    STEPS_PER_ARM = CAROUSEL_FULL_STEPS // 9   # Steps between adjacent arms ≈ 6423

    # Bin sector angular size in steps (for reference / debugging)
    STEPS_PER_SECTOR = CAROUSEL_FULL_STEPS // 5  # ≈ 11561 steps per 72° sector

    STEP_DELAY_MIN = 0.00012
    STEP_DELAY_MAX = 0.00065

    # Arm Position Map: Arm #1 is the calibrated origin.
    # Arm #2 is exactly one 1.9cm step away from Arm #1.
    # Add more arms here if you expand the setup.
    SERVO_ARM1_ORIGIN_ANGLE = 0    # Calibrated angle pointing at Robotic Arm #1
    SERVO_ARM2_ANGLE        = 2    # Calibrated angle pointing at Robotic Arm #2 (1.9cm from Arm #1)
    SERVO_STEP_DEG          = 2    # Degrees per 1.9cm step on 180° mode

    # ==============================================================================
    # PCA9685 (0x40) Robotic Drop Arm Channel Map
    # Board: 0x40  |  All arms use kit.servo[channel]
    # ==============================================================================
    #  Arm #  |  PCA9685 Channel
    # --------+------------------
    #    1    |       3
    #    2    |       2
    #    3    |       1
    #    4    |       0
    #    5    |       8
    #    6    |       7
    #    7    |       6
    #    8    |       5
    #    9    |       4
    ARM_CHANNELS = {
        1: 3,
        2: 2,
        3: 1,
        4: 0,
        5: 8,
        6: 7,
        7: 6,
        8: 5,
        9: 4,
    }
    ARM_DROP_ANGLE = 60   # Angle to extend / release the drop mechanism
    ARM_HOME_ANGLE = 0    # Angle to retract the arm back to ready position

    # ==============================================================================
    # ARM STEPPER OFFSETS  —  Auto-computed from disk geometry
    # ==============================================================================
    # Each arm is STEPS_PER_ARM steps further CW than the previous arm.
    # Arm #1 = 0 offset (reference).  Arm #N = (N-1) × STEPS_PER_ARM.
    #
    # Explicit per-arm offsets — calibrate each arm physically.
    # Arm #1 = 0 (reference, already configured).
    # For arms 2-9: jog disk until correct bin aligns, read step count, enter here.
    # Default values below are ESTIMATES based on equal 40° spacing.
    # Replace after running the ARM OFFSET CALIBRATION tool in the UI.
    ARM_STEPPER_OFFSETS = {
        1: 0,                    # Arm #1 — CONFIGURED reference (do not change)
        2: 1 * STEPS_PER_ARM,   # Arm #2 — estimated ≈ 6423  | calibrate via UI
        3: 2 * STEPS_PER_ARM,   # Arm #3 — estimated ≈ 12845 | calibrate via UI
        4: 3 * STEPS_PER_ARM,   # Arm #4 — estimated ≈ 19268 | calibrate via UI
        5: 4 * STEPS_PER_ARM,   # Arm #5 — estimated ≈ 25690 | calibrate via UI
        6: 5 * STEPS_PER_ARM,   # Arm #6 — estimated ≈ 32113 | calibrate via UI
        7: 6 * STEPS_PER_ARM,   # Arm #7 — estimated ≈ 38535 | calibrate via UI
        8: 7 * STEPS_PER_ARM,   # Arm #8 — estimated ≈ 44957 | calibrate via UI
        9: 8 * STEPS_PER_ARM,   # Arm #9 — estimated ≈ 51380 | calibrate via UI
    }
    # Fine-tune individual arms after physical measurement, e.g.:
    # ARM_STEPPER_OFFSETS[3] = 13010   # measured value for Arm #3

    # ==============================================================================
    # SIMULATION CLASS DATA  —  Fake scan results for testing the sort logic
    # ==============================================================================
    # Change these to any class (A/B/C/D/E) to test the full sort cycle without
    # a live camera. Set to None for arms that have no leaf (will be skipped).
    SIMULATION_CLASSES = {
        1: 'C',    # Arm #1 — simulated scan result: Class C
        2: 'D',    # Arm #2 — simulated scan result: Class D
        3: 'A',    # Arm #3 — simulated scan result: Class A
        4: 'B',    # Arm #4 — simulated scan result: Class B
        5: 'E',    # Arm #5 — simulated scan result: Class E
        6: None,   # Arm #6 — no leaf (skipped)
        7: None,   # Arm #7 — no leaf (skipped)
        8: None,   # Arm #8 — no leaf (skipped)
        9: None,   # Arm #9 — no leaf (skipped)
    }

    current_step_pos = 0
    current_station_name = "A"
    cam_current_angle = SERVO_ARM1_ORIGIN_ANGLE
    net_continuous_steps = 0       # Tracks forward 1.9cm hops for 360° continuous return
    is_busy = False
    stop_requested = False

    # ==============================================================================
    # 4. ROBUST ACTUATOR CONTROLLERS
    # ==============================================================================
    def hard_kill_servo():
        """Forces PWM output to 0 duty cycle to prevent continuous rotation drift."""
        global kit
        if kit is not None:
            try:
                # 0 throttle is the proper neutral brake for 360 continuous servos
                kit.continuous_servo[CAMERA_CHANNEL].throttle = 0
                time.sleep(0.05)
                # Directly shut off PCA9685 PWM channel register to completely detach
                kit._pca.channels[CAMERA_CHANNEL].duty_cycle = 0
            except Exception:
                pass

    def stop_all_hardware():
        global stop_requested
        stop_requested = True
        hard_kill_servo()
        GPIO.output(STEP_PIN, GPIO.LOW)
        print("[SAFETY] All hardware output halted.")

    def move_steps(num_steps, clockwise=True):
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
                delay = STEP_DELAY_MAX - ((STEP_DELAY_MAX - STEP_DELAY_MIN) * (i / ramp_steps))
            elif i > (num_steps - ramp_steps):
                remaining = num_steps - i
                delay = STEP_DELAY_MAX - ((STEP_DELAY_MAX - STEP_DELAY_MIN) * (remaining / ramp_steps))
            else:
                delay = STEP_DELAY_MIN

            GPIO.output(STEP_PIN, GPIO.HIGH)
            time.sleep(delay)
            GPIO.output(STEP_PIN, GPIO.LOW)
            time.sleep(delay)

    def execute_rotate(target_name, update_ui_cb=None):
        """
        Rotates the carousel so that target_name class bin faces Arm #1.
        Uses the position saved in CALIBRATED_POSITIONS[1][target_name].

        Returns True on success, False if no calibrated position exists yet.
        When not calibrated, the disk does NOT move and the caller is informed.
        """
        global current_step_pos, current_station_name, stop_requested
        target_name = target_name.upper()

        target_steps = get_disk_target_steps(target_name, 1)
        if target_steps is None:
            msg = f"Arm #1 / Class {target_name} not calibrated — use calibration panel first."
            print(f"[CALIB] {msg}")
            if update_ui_cb:
                update_ui_cb(msg)
            return False

        if update_ui_cb:
            cw  = (target_steps - current_step_pos) % CAROUSEL_FULL_STEPS
            ccw = (current_step_pos - target_steps) % CAROUSEL_FULL_STEPS
            direction = "CW" if cw <= ccw else "CCW"
            steps = min(cw, ccw)
            update_ui_cb(
                f"Stepper → Class {target_name} ({steps} steps {direction}, "
                f"target={target_steps})..."
            )

        rotate_to_absolute_steps(target_steps, update_ui_cb=None)

        if not stop_requested:
            current_station_name = target_name
        return True


    def rotate_to_absolute_steps(target_steps, update_ui_cb=None):
        """
        Rotates the carousel disk to an ABSOLUTE step position.

        target_steps is the raw step count saved during calibration — it is
        NOT normalized with modulo because doing so would corrupt the value
        if CAROUSEL_FULL_STEPS is even slightly off.

        Shortest-path logic: picks CW or CCW whichever is fewer steps,
        using CAROUSEL_FULL_STEPS only for the direction comparison.
        """
        global current_step_pos, current_station_name, stop_requested

        target_steps = int(target_steps)

        # Net delta from current position to target
        delta = target_steps - current_step_pos

        # Pick the shortest physical path (CW or CCW) using the full-revolution count
        cw_steps  =  delta % CAROUSEL_FULL_STEPS   # positive remainder = CW
        ccw_steps = (-delta) % CAROUSEL_FULL_STEPS  # positive remainder = CCW

        if cw_steps <= ccw_steps:
            direction = "CW"
            steps = cw_steps
            clockwise = True
        else:
            direction = "CCW"
            steps = ccw_steps
            clockwise = False

        if update_ui_cb:
            update_ui_cb(f"Disk → {target_steps} steps ({steps} {direction})...")

        if steps > 0:
            move_steps(steps, clockwise=clockwise)

        if not stop_requested:
            current_step_pos = target_steps
            current_station_name = "?"


    def get_disk_target_steps(bin_class, arm_num):
        """
        Returns the saved step position for (arm_num, bin_class), or None if
        this combination has not been manually calibrated yet.

        There is NO formula fallback. Every position must be set explicitly
        via the calibration panel: move the disk, jog to exact alignment, Save.
        """
        bin_class = bin_class.upper()
        arm_data = CALIBRATED_POSITIONS.get(arm_num, {})
        return arm_data.get(bin_class, None)   # None = not yet calibrated


    def rotate_bin_to_arm(bin_class, arm_num, update_ui_cb=None):
        """
        Rotates the disk so bin_class sector is under arm_num.
        Uses the manually saved position in CALIBRATED_POSITIONS[arm_num][bin_class].
        Does nothing (and returns False) if the combination has not been calibrated yet.
        """
        target = get_disk_target_steps(bin_class, arm_num)
        if target is None:
            msg = f"Arm #{arm_num} / Class {bin_class} not calibrated — skipping."
            print(f"[CALIB] {msg}")
            if update_ui_cb:
                update_ui_cb(msg)
            return False
        if update_ui_cb:
            update_ui_cb(f"Bin {bin_class} → Arm #{arm_num}  [{target} steps]")
        rotate_to_absolute_steps(target, update_ui_cb)
        return True

    # ==============================================================================
    # 5. TKINTER APPLICATION
    # ==============================================================================
    class CarouselApp(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("Buyo Leaf Sorter - Master Dashboard")
            self.geometry("760x860")
            self.resizable(True, True)
            self.configure(bg="#1E1E2E")
            self.minsize(700, 600)

            self.protocol("WM_DELETE_WINDOW", self.safe_exit_to_class_a)

            # Load persisted calibration (must happen before build_ui so the
            # offset table renders the correct values on first draw)
            load_calibration()

            # DEFAULT TO 360 CONTINUOUS TO PREVENT INFINITE SPINNING
            self.servo_mode = tk.StringVar(value="360_CONTINUOUS")
            self.pulse_duration = tk.DoubleVar(value=0.026)  # 26ms for 1.9cm on 360 servos
            self.sync_servo_toggle = tk.BooleanVar(value=False)
            self.sim_mode = tk.BooleanVar(value=False)   # Simulation mode: uses SIMULATION_CLASSES
            self.cal_arm_var = tk.IntVar(value=1)          # Arm selector for offset calibration
            self.cal_class_var = tk.StringVar(value='A')   # Target class (derived from bin pair)
            self.cal_bin_pair_var = tk.StringVar(value='A') # Bin pair selection e.g. 'A', 'A→B', 'B→C'

            # ---- Scrollable canvas wrapper ----
            self._canvas = tk.Canvas(self, bg="#1E1E2E", highlightthickness=0)
            self._scrollbar = tk.Scrollbar(self, orient="vertical",
                                        command=self._canvas.yview)
            self._canvas.configure(yscrollcommand=self._scrollbar.set)

            self._scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            # Inner frame that holds all widgets
            self._inner_frame = tk.Frame(self._canvas, bg="#1E1E2E")
            self._inner_frame_id = self._canvas.create_window(
                (0, 0), window=self._inner_frame, anchor="nw"
            )

            # Resize scroll region when inner frame changes size
            self._inner_frame.bind(
                "<Configure>",
                lambda e: self._canvas.configure(
                    scrollregion=self._canvas.bbox("all")
                )
            )
            # Stretch inner frame to canvas width
            self._canvas.bind(
                "<Configure>",
                lambda e: self._canvas.itemconfig(
                    self._inner_frame_id, width=e.width
                )
            )

            # Mouse-wheel scrolling (Linux/RPi + Windows)
            self._canvas.bind_all("<MouseWheel>",
                lambda e: self._canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))
            self._canvas.bind_all("<Button-4>",
                lambda e: self._canvas.yview_scroll(-1, "units"))
            self._canvas.bind_all("<Button-5>",
                lambda e: self._canvas.yview_scroll(1, "units"))

            self.build_ui()
            self.refresh_peripherals()

        def build_ui(self):
            # All widgets are placed inside _inner_frame (inside the scrollable canvas)
            root = self._inner_frame

            # Header
            title_frame = tk.Frame(root, bg="#282A36", pady=6)
            title_frame.pack(fill=tk.X)
            tk.Label(
                title_frame, text="🌿 BUYO LEAF ROBOTIC CONTROLLER",
                font=("Arial", 14, "bold"), fg="#50FA7B", bg="#282A36"
            ).pack()

            # Telemetry Badges
            periph_frame = tk.LabelFrame(
                root, text=" Peripheral Telemetry ", font=("Arial", 10, "bold"),
                fg="#8BE9FD", bg="#1E1E2E", padx=8, pady=4
            )
            periph_frame.pack(fill=tk.X, padx=15, pady=4)

            self.status_badges = {}
            items = [
                ("camera", "📷 Camera Module", 0, 0),
                ("stepper", "⚙️ Stepper Carousel", 0, 1),
                ("servo_cam", "🎯 MG996R Camera Servo", 1, 0),
                ("arm", "🦾 Robotic Arm Bus", 1, 1),
            ]
            for key, name, row, col in items:
                box = tk.Frame(periph_frame, bg="#282A36", padx=6, pady=2, relief=tk.RIDGE, bd=1)
                box.grid(row=row, column=col, padx=4, pady=2, sticky="ew")
                periph_frame.grid_columnconfigure(col, weight=1)

                tk.Label(box, text=name, font=("Arial", 9, "bold"), fg="#F8F8F2", bg="#282A36").pack(anchor="w")
                lbl_val = tk.Label(box, text="Scanning...", font=("Arial", 8), fg="#F1FA8C", bg="#282A36")
                lbl_val.pack(anchor="w")
                self.status_badges[key] = lbl_val

            # Carousel Telemetry Banner
            status_frame = tk.LabelFrame(
                root, text=" Motion Telemetry ", font=("Arial", 10, "bold"),
                fg="#8BE9FD", bg="#1E1E2E", padx=10, pady=4
            )
            status_frame.pack(fill=tk.X, padx=15, pady=4)

            self.lbl_station = tk.Label(
                status_frame, text="Active Station: Class A (Origin)",
                font=("Arial", 11, "bold"), fg="#50FA7B", bg="#1E1E2E"
            )
            self.lbl_station.pack(anchor="w")

            self.lbl_steps = tk.Label(
                status_frame, text=f"Absolute Step Position: 0 pulses | Camera Angle: {cam_current_angle}° (Arm #1)",
                font=("Arial", 9), fg="#BD93F9", bg="#1E1E2E"
            )
            self.lbl_steps.pack(anchor="w")

            self.lbl_status = tk.Label(
                status_frame, text="Status: Ready at Arm #1 / Class A Origin", font=("Arial", 9, "italic"),
                fg="#F1FA8C", bg="#1E1E2E"
            )
            self.lbl_status.pack(anchor="w")

            # Stepper Carousel Controls
            btn_frame = tk.LabelFrame(
                root, text=" Carousel Station Controls (Class A = Origin) ",
                font=("Arial", 10, "bold"), fg="#8BE9FD", bg="#1E1E2E", padx=8, pady=4
            )
            btn_frame.pack(fill=tk.X, padx=15, pady=4)

            sync_chk = tk.Checkbutton(
                btn_frame,
                text="⚡ Simultaneous Mode: Move Carousel to Class + Rotate Servo 1.9cm (Hold 5s)",
                variable=self.sync_servo_toggle,
                font=("Arial", 9, "bold"),
                fg="#50FA7B",
                bg="#1E1E2E",
                selectcolor="#282A36",
                activebackground="#1E1E2E",
                activeforeground="#50FA7B"
            )
            sync_chk.pack(anchor="w", padx=4, pady=2)

            grid_btn_box = tk.Frame(btn_frame, bg="#1E1E2E")
            grid_btn_box.pack(fill=tk.X, pady=2)

            self.buttons = {}
            bins = [
                ("Class A (Origin)", "A", "#2E7D32"),
                ("Class B",          "B", "#44475A"),
                ("Class C (-10%)",   "C", "#44475A"),
                ("Class D (+5%)",    "D", "#44475A"),
                ("Class E (-5%)",    "E", "#44475A")
            ]
            for idx, (label_text, bin_code, col) in enumerate(bins):
                btn = tk.Button(
                    grid_btn_box, text=label_text, font=("Arial", 9, "bold"),
                    bg=col, fg="#FFFFFF", activebackground="#FF79C6", width=15,
                    command=lambda code=bin_code: self.handle_class_button_click(code)
                )
                if idx < 3:
                    btn.grid(row=0, column=idx, padx=3, pady=2)
                else:
                    btn.grid(row=1, column=idx - 3, padx=3, pady=2)
                self.buttons[bin_code] = btn

            # Servo Control Section: 1.9cm Step Inspection
            servo_frame = tk.LabelFrame(
                root, text=" MG996R Camera Inspection (1.9cm Rotation per Step / 5s Dwell) ",
                font=("Arial", 10, "bold"), fg="#8BE9FD", bg="#1E1E2E", padx=8, pady=4
            )
            servo_frame.pack(fill=tk.X, padx=15, pady=4)

            mode_box = tk.Frame(servo_frame, bg="#1E1E2E")
            mode_box.pack(fill=tk.X, pady=2)

            tk.Label(mode_box, text="Servo Mode:", font=("Arial", 9, "bold"), fg="#F8F8F2", bg="#1E1E2E").pack(side=tk.LEFT, padx=5)
            tk.Radiobutton(
                mode_box, text="180° Standard Positional", variable=self.servo_mode,
                value="180_STANDARD", bg="#1E1E2E", fg="#50FA7B", selectcolor="#282A36",
                activebackground="#1E1E2E"
            ).pack(side=tk.LEFT, padx=5)
            tk.Radiobutton(
                mode_box, text="360° Continuous (Timed 1.9cm Pulse)", variable=self.servo_mode,
                value="360_CONTINUOUS", bg="#1E1E2E", fg="#8BE9FD", selectcolor="#282A36",
                activebackground="#1E1E2E"
            ).pack(side=tk.LEFT, padx=5)

            # 9 Discrete Inspection Buttons
            grid_box = tk.Frame(servo_frame, bg="#1E1E2E")
            grid_box.pack(fill=tk.X, pady=4)

            self.arm_step_buttons = []
            for i in range(1, 10):
                target_val = round(i * 1.9, 1)
                target_deg_display = int(round(target_val))
                r = (i - 1) // 3
                c = (i - 1) % 3
                btn = tk.Button(
                    grid_box, text=f"Arm {i}: {target_val}cm ({target_deg_display}°)", font=("Arial", 9, "bold"),
                    bg="#282A36", fg="#50FA7B", activebackground="#44475A",
                    command=lambda step_num=i, val=target_val: self.threaded_task(self.task_execute_servo_step_with_5s_stop, step_num, val)
                )
                btn.grid(row=r, column=c, padx=3, pady=2, sticky="ew")
                grid_box.grid_columnconfigure(c, weight=1)
                self.arm_step_buttons.append(btn)

            # Helper Actions: Brake, Center
            act_box = tk.Frame(servo_frame, bg="#1E1E2E")
            act_box.pack(fill=tk.X, pady=3)

            self.btn_brake = tk.Button(
                act_box, text="🛑 HARD STOP SERVO", font=("Arial", 9, "bold"),
                bg="#FF5555", fg="#FFFFFF", command=self.task_instant_brake
            )
            self.btn_brake.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

            self.btn_center = tk.Button(
                act_box, text="🎯 Return Camera to Arm #1", font=("Arial", 9, "bold"),
                bg="#6272A4", fg="#FFFFFF", command=lambda: self.threaded_task(self.task_return_camera_to_arm1)
            )
            self.btn_center.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

            self.btn_sweep_arm = tk.Button(
                act_box, text="▶️ Sweep 9 Pos (1.9cm + 5s Stop Each)", font=("Arial", 9, "bold"),
                bg="#BD93F9", fg="#282A36", command=lambda: self.threaded_task(self.task_auto_sweep_9_steps)
            )
            self.btn_sweep_arm.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

            # Automation Section
            auto_frame = tk.LabelFrame(
                root, text=" Automation & Inspection Workflows ", font=("Arial", 10, "bold"),
                fg="#8BE9FD", bg="#1E1E2E", padx=8, pady=4
            )
            auto_frame.pack(fill=tk.X, padx=15, pady=4)

            # AUTO SORT — Primary sorting algorithm button
            sort_row = tk.Frame(auto_frame, bg="#1E1E2E")
            sort_row.pack(fill=tk.X, pady=3)

            self.btn_auto_sort = tk.Button(
                sort_row,
                text="🤖 AUTO SORT  —  Scan All 9 Arms ➔ Classify ➔ Rotate Bin ➔ Drop  (AUTO RETURN HOME)",
                font=("Arial", 10, "bold"),
                bg="#FF79C6", fg="#282A36", activebackground="#FF92D0", height=2,
                command=lambda: self.threaded_task(self.task_full_sort_cycle)
            )
            self.btn_auto_sort.pack(fill=tk.X, padx=2)

            # Simulation Mode toggle
            sim_row = tk.Frame(auto_frame, bg="#1E1E2E")
            sim_row.pack(fill=tk.X, pady=1)

            tk.Checkbutton(
                sim_row,
                text="🧪 SIMULATION MODE  —  Use SIMULATION_CLASSES test data instead of live camera",
                variable=self.sim_mode,
                font=("Arial", 9, "bold"),
                fg="#FFB86C", bg="#1E1E2E",
                selectcolor="#282A36",
                activebackground="#1E1E2E",
                activeforeground="#FFB86C"
            ).pack(anchor="w", padx=6, pady=1)

            # Dedicated Highlight Button: Test Scan & Drop Class A & B
            test_row = tk.Frame(auto_frame, bg="#1E1E2E")
            test_row.pack(fill=tk.X, pady=3)

            self.btn_test_ab = tk.Button(
                test_row,
                text="🧪 TEST SCAN & DROP: ARM #1 ➔ CLASS A  |  ARM #2 ➔ CLASS B  (AUTO RETURN HOME)",
                font=("Arial", 10, "bold"),
                bg="#50FA7B", fg="#282A36", activebackground="#69FF94", height=2,
                command=lambda: self.threaded_task(self.task_test_class_a_and_b)
            )
            self.btn_test_ab.pack(fill=tk.X, padx=2)

            # Row 2 of Automation Controls
            auto_row2 = tk.Frame(auto_frame, bg="#1E1E2E")
            auto_row2.pack(fill=tk.X, pady=2)

            self.btn_sync_move = tk.Button(
                auto_row2, text="⚡ SYNC MOVE + 5s STOP\n(Target Class + 1.9cm)", font=("Arial", 9, "bold"),
                bg="#8BE9FD", fg="#282A36", activebackground="#A4FFFF", height=2,
                command=self.prompt_simultaneous_move
            )
            self.btn_sync_move.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

            self.btn_full_inspect_cycle = tk.Button(
                auto_row2, text="🔁 FULL CYCLE (A➔E)\n(Independent 5s Hold)", font=("Arial", 9, "bold"),
                bg="#FFB86C", fg="#282A36", activebackground="#FFC785", height=2,
                command=lambda: self.threaded_task(self.task_full_inspection_cycle)
            )
            self.btn_full_inspect_cycle.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

            self.btn_sweep = tk.Button(
                auto_row2, text="Carousel Sweep\n(Fast Test)", font=("Arial", 9, "bold"),
                bg="#BD93F9", fg="#282A36", height=2,
                command=lambda: self.threaded_task(self.task_automated_sweep)
            )
            self.btn_sweep.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

            # ----------------------------------------------------------------
            # BIN POSITION CALIBRATION  — Simple Arm × Class
            # ----------------------------------------------------------------
            calib_frame = tk.LabelFrame(
                root, text=" 📏 Bin Position Calibration (Per Arm × Per Class) ",
                font=("Arial", 10, "bold"),
                fg="#FFB86C", bg="#1E1E2E", padx=8, pady=6
            )
            calib_frame.pack(fill=tk.X, padx=15, pady=4)

            # ── Arm selector + origin readout ────────────────────────────
            arm_row = tk.Frame(calib_frame, bg="#1E1E2E")
            arm_row.pack(fill=tk.X, pady=(4, 2))

            tk.Label(arm_row, text="Arm #:", font=("Arial", 9, "bold"),
                    fg="#F8F8F2", bg="#1E1E2E").pack(side=tk.LEFT, padx=4)
            arm_spin = tk.Spinbox(
                arm_row, from_=1, to=9, textvariable=self.cal_arm_var,
                width=3, font=("Arial", 11, "bold"),
                bg="#282A36", fg="#FFB86C", buttonbackground="#44475A"
            )
            arm_spin.pack(side=tk.LEFT, padx=4)

            self.lbl_origin = tk.Label(
                arm_row, text="Origin: Arm #1 / Class A = — steps",
                font=("Arial", 9, "italic"), fg="#6272A4", bg="#1E1E2E"
            )
            self.lbl_origin.pack(side=tk.LEFT, padx=16)

            # ── Class buttons — click to go to that saved position ───────
            # Each cell shows:  [Class X button]  →  saved step value label
            cls_btn_row = tk.Frame(calib_frame, bg="#1E1E2E")
            cls_btn_row.pack(fill=tk.X, pady=4)

            CLASS_COLORS = {
                'A': '#50FA7B', 'B': '#8BE9FD',
                'C': '#FFB86C', 'D': '#FF79C6', 'E': '#BD93F9'
            }
            self._cls_buttons    = {}   # cls -> Button widget
            self._cls_val_labels = {}   # cls -> Label showing saved steps

            for cls in ['A', 'B', 'C', 'D', 'E']:
                col = CLASS_COLORS[cls]
                cell = tk.Frame(cls_btn_row, bg="#282A36", relief=tk.RIDGE, bd=1)
                cell.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=3)

                btn = tk.Button(
                    cell, text=f"Class {cls}",
                    font=("Arial", 10, "bold"),
                    bg="#44475A", fg=col,
                    activebackground=col, activeforeground="#282A36",
                    relief=tk.RAISED, bd=2,
                    command=lambda c=cls: self.threaded_task(self.calibrate_go_to_class, c)
                )
                btn.pack(fill=tk.X, padx=2, pady=(4, 1))
                self._cls_buttons[cls] = btn

                val_lbl = tk.Label(cell, text="not set",
                                font=("Courier", 8), fg="#6272A4", bg="#282A36")
                val_lbl.pack(pady=(0, 4))
                self._cls_val_labels[cls] = val_lbl

            # ── Position / delta readout ─────────────────────────────────
            self.lbl_calib_steps = tk.Label(
                calib_frame,
                text=f"Current: {current_step_pos} steps  |  Δ from origin: —",
                font=("Arial", 9), fg="#F1FA8C", bg="#1E1E2E"
            )
            self.lbl_calib_steps.pack(anchor="w", padx=4, pady=2)

            # ── Jog row ──────────────────────────────────────────────────
            row_jog = tk.Frame(calib_frame, bg="#1E1E2E")
            row_jog.pack(fill=tk.X, pady=2)

            tk.Label(row_jog, text="◀ Reverse:", font=("Arial", 8, "bold"),
                    fg="#FF5555", bg="#1E1E2E").pack(side=tk.LEFT, padx=2)
            for delta in [-1000, -100, -10]:
                tk.Button(
                    row_jog, text=str(delta), font=("Arial", 9, "bold"),
                    bg="#FF5555", fg="#282A36", width=6,
                    command=lambda d=delta: self.threaded_task(self.calibrate_jog_steps, d)
                ).pack(side=tk.LEFT, padx=2)

            tk.Label(row_jog, text="  |  ", fg="#44475A", bg="#1E1E2E").pack(side=tk.LEFT)

            tk.Label(row_jog, text="Forward ▶:", font=("Arial", 8, "bold"),
                    fg="#50FA7B", bg="#1E1E2E").pack(side=tk.LEFT, padx=2)
            for delta in [10, 100, 1000]:
                tk.Button(
                    row_jog, text=f"+{delta}", font=("Arial", 9, "bold"),
                    bg="#50FA7B", fg="#282A36", width=6,
                    command=lambda d=delta: self.threaded_task(self.calibrate_jog_steps, d)
                ).pack(side=tk.LEFT, padx=2)

            # ── Tracked pos + reset home ─────────────────────────────────
            row_home = tk.Frame(calib_frame, bg="#1E1E2E")
            row_home.pack(fill=tk.X, pady=2)

            tk.Label(row_home, text="Tracked pos:", font=("Arial", 9, "bold"),
                    fg="#F8F8F2", bg="#1E1E2E").pack(side=tk.LEFT, padx=4)
            self.lbl_tracked_pos = tk.Label(
                row_home, text=f"{current_step_pos} steps",
                font=("Arial", 10, "bold"), fg="#FF79C6", bg="#1E1E2E"
            )
            self.lbl_tracked_pos.pack(side=tk.LEFT, padx=6)
            tk.Button(
                row_home, text="📍 Disk is at Class A — Reset tracker",
                font=("Arial", 9, "bold"), bg="#FF5555", fg="#FFFFFF",
                command=self.set_home_to_class_a
            ).pack(side=tk.LEFT, padx=10)
            tk.Label(
                row_home, text="← Use after crash / unexpected restart",
                font=("Arial", 8, "italic"), fg="#6272A4", bg="#1E1E2E"
            ).pack(side=tk.LEFT)

            # ── Save row ─────────────────────────────────────────────────
            row_save = tk.Frame(calib_frame, bg="#1E1E2E")
            row_save.pack(fill=tk.X, pady=4)

            self.btn_save_offset = tk.Button(
                row_save, text="✅ Save  Arm #1 / Class A",
                font=("Arial", 9, "bold"), bg="#50FA7B", fg="#282A36",
                command=self.save_arm_offset
            )
            self.btn_save_offset.pack(side=tk.LEFT, padx=4)

            self.lbl_offset_table = tk.Label(
                row_save, text=self._build_offset_table(),
                font=("Courier", 8), fg="#6272A4", bg="#1E1E2E", justify="left"
            )
            self.lbl_offset_table.pack(side=tk.LEFT, padx=12)

            # Refresh class-button labels whenever arm changes
            self.cal_arm_var.trace_add("write", lambda *_: self._refresh_calib_ui())
            self._refresh_calib_ui()

            # Safe Exit
            self.btn_exit = tk.Button(
                root, text="🚪 REVERT TO ORIGIN (ARM #1 & CLASS A) & EXIT APP", font=("Arial", 11, "bold"),
                bg="#E11D48", fg="#FFFFFF", height=2, command=self.safe_exit_to_class_a
            )
            self.btn_exit.pack(fill=tk.X, padx=15, pady=6)


        # ==========================================================================
        # CALIBRATION METHODS
        # ==========================================================================
        def _build_offset_table(self):
            """
            Shows the saved CALIBRATED_POSITIONS grid: Arm × Class.
            Only displays arms that have at least one saved class position.
            """
            if not CALIBRATED_POSITIONS:
                return "No positions calibrated yet."
            header = "      " + "  ".join(f"[{c}]" for c in ['A','B','C','D','E'])
            rows = [header]
            for arm in range(1, 10):
                arm_data = CALIBRATED_POSITIONS.get(arm, {})
                if arm_data:
                    row = f"  Arm #{arm}:" + "  ".join(
                        f"{'  --' if cls not in arm_data else str(arm_data[cls]).rjust(6)}"
                        for cls in ['A','B','C','D','E']
                    )
                    rows.append(row)
            return "\n".join(rows)

        def _refresh_calib_ui(self):
            """Refreshes class button step-value labels and origin label for the selected arm."""
            arm_num = self.cal_arm_var.get()
            arm_data = CALIBRATED_POSITIONS.get(arm_num, {})

            for cls in ['A', 'B', 'C', 'D', 'E']:
                pos = arm_data.get(cls)
                self._cls_val_labels[cls].config(
                    text=f"{pos} steps" if pos is not None else "not set",
                    fg="#F1FA8C" if pos is not None else "#6272A4"
                )

            origin_pos = arm_data.get('A')
            origin_txt = (
                f"Origin: Arm #{arm_num} / Class A = {origin_pos} steps"
                if origin_pos is not None
                else f"Origin: Arm #{arm_num} / Class A = not set"
            )
            self.lbl_origin.config(text=origin_txt)
            self.btn_save_offset.config(
                text=f"✅ Save  Arm #{arm_num} / Class {self.cal_class_var.get()}"
            )
            self.lbl_calib_steps.config(
                text=f"Current: {current_step_pos} steps  |  Δ from origin: "
                    + (f"{current_step_pos - origin_pos:+d} steps" if origin_pos is not None else "—")
            )

        def calibrate_go_to_class(self, cls):
            """
            Moves the disk to the saved position for the selected arm + cls.
            Shows how many steps were moved (delta from previous position).
            If that class has no saved position yet the disk stays and the
            user is told to jog to alignment and then Save.
            """
            arm_num = self.cal_arm_var.get()
            self.cal_class_var.set(cls)

            # Highlight the active class button
            CLASS_COLORS = {
                'A': '#50FA7B', 'B': '#8BE9FD',
                'C': '#FFB86C', 'D': '#FF79C6', 'E': '#BD93F9'
            }
            for c, btn in self._cls_buttons.items():
                color = CLASS_COLORS[c]
                if c == cls:
                    btn.config(bg=color, fg="#282A36", relief=tk.SUNKEN)
                else:
                    btn.config(bg="#44475A", fg=color, relief=tk.RAISED)

            target = get_disk_target_steps(cls, arm_num)
            if target is None:
                self.lbl_status.config(
                    text=(
                        f"Status: 📏 Arm #{arm_num} / Class {cls} not set. "
                        f"Jog from current position ({current_step_pos} steps) until aligned, then Save."
                    )
                )
                self.btn_save_offset.config(text=f"✅ Save  Arm #{arm_num} / Class {cls}")
                return

            prev_pos = current_step_pos

            # ── Compute shortest-path direction for user feedback ────────
            # Mirrors the logic inside rotate_to_absolute_steps so the status
            # label shows the actual direction BEFORE the motor starts turning.
            _delta    = target - prev_pos
            _cw_steps  =  _delta  % CAROUSEL_FULL_STEPS
            _ccw_steps = (-_delta) % CAROUSEL_FULL_STEPS
            if _cw_steps <= _ccw_steps:
                dir_lbl   = "CW →"
                dir_steps = _cw_steps
            else:
                dir_lbl   = "← CCW"
                dir_steps = _ccw_steps

            self.lbl_status.config(
                text=(
                    f"Status: 🔄 Arm #{arm_num}: Class {cls}  "
                    f"({dir_lbl}  {dir_steps} steps)..."
                )
            )
            rotate_to_absolute_steps(
                target,
                update_ui_cb=lambda m: self.lbl_status.config(
                    text=f"Status: 🔄 Arm #{arm_num} → Class {cls}: {m}"
                )
            )

            actual_moved = current_step_pos - prev_pos
            origin_pos = CALIBRATED_POSITIONS.get(arm_num, {}).get('A')
            delta_from_origin = (
                f"{current_step_pos - origin_pos:+d} steps from Class A"
                if origin_pos is not None else "origin not set"
            )
            self.lbl_calib_steps.config(
                text=(
                    f"Current: {current_step_pos} steps  |  "
                    f"{dir_lbl} {dir_steps} steps  |  {delta_from_origin}"
                )
            )
            self.lbl_tracked_pos.config(text=f"{current_step_pos} steps")
            self.btn_save_offset.config(text=f"✅ Save  Arm #{arm_num} / Class {cls}")
            self.lbl_status.config(
                text=(
                    f"Status: ✅ Arm #{arm_num} at Class {cls} "
                    f"({current_step_pos} steps  |  {dir_lbl} {dir_steps} steps). "
                    f"Fine-jog if needed, then Save."
                )
            )

        def calibrate_jog_steps(self, delta):
            """Jogs the disk by exactly delta steps (positive=CW, negative=CCW)."""
            global current_step_pos
            if delta > 0:
                move_steps(delta, clockwise=True)
            else:
                move_steps(abs(delta), clockwise=False)
            if not stop_requested:
                # Track position as plain cumulative sum — NO modulo wrapping.
                # Saved calibration positions are absolute counts from the last
                # home point; wrapping here would corrupt the reference frame.
                current_step_pos = current_step_pos + delta
            arm_num   = self.cal_arm_var.get()
            origin_pos = CALIBRATED_POSITIONS.get(arm_num, {}).get('A')
            delta_txt = (
                f"{current_step_pos - origin_pos:+d} steps from Class A"
                if origin_pos is not None else "origin not set"
            )
            self.lbl_calib_steps.config(
                text=f"Current: {current_step_pos} steps  |  {delta_txt}"
            )
            self.lbl_steps.config(
                text=f"Stepper: {current_step_pos} steps  |  Calibrating Arm #{arm_num}"
            )
            self.lbl_tracked_pos.config(text=f"{current_step_pos} steps")

        def save_arm_offset(self):
            """Saves current disk position for the selected Arm × Class. Persists to calibration.json."""
            arm_num   = self.cal_arm_var.get()
            ref_class = self.cal_class_var.get()

            if arm_num not in CALIBRATED_POSITIONS:
                CALIBRATED_POSITIONS[arm_num] = {}
            CALIBRATED_POSITIONS[arm_num][ref_class] = current_step_pos

            save_calibration()
            print(f"[CALIB] CALIBRATED_POSITIONS[{arm_num}]['{ref_class}'] = {current_step_pos}")

            self._refresh_calib_ui()
            self.lbl_offset_table.config(text=self._build_offset_table())
            self.lbl_tracked_pos.config(text=f"{current_step_pos} steps")
            self.lbl_status.config(
                text=(
                    f"✅ Arm #{arm_num} / Class {ref_class} = {current_step_pos} steps "
                    f"— saved to calibration.json"
                )
            )

        def set_home_to_class_a(self):
            """
            Manual re-home: tells the software the disk is physically at the
            Class A position RIGHT NOW.  Use after a crash or unexpected restart
            where the disk was NOT returned to Class A before the app closed.

            Sets current_step_pos = CALIBRATED_POSITIONS[1]['A'] (or 0 if not
            yet calibrated), then saves to calibration.json so every subsequent
            move is calculated from the correct baseline.
            """
            global current_step_pos
            class_a_pos = CALIBRATED_POSITIONS.get(1, {}).get('A', 0)
            current_step_pos = class_a_pos
            save_calibration()
            self.lbl_tracked_pos.config(text=f"{current_step_pos} steps")
            self.lbl_steps.config(
                text=f"Stepper: {current_step_pos} steps  |  Home = Class A"
            )
            self.lbl_status.config(
                text=(
                    f"📍 Re-homed: tracker set to {current_step_pos} steps (Class A). "
                    f"All bin moves now calculated from this baseline."
                )
            )
            print(f"[CALIB] Re-homed: current_step_pos = {current_step_pos} (Class A)")

        # ==========================================================================
        # LOGIC & TASK DISPATCHER
        # ==========================================================================
        def refresh_peripherals(self):
            global kit
            if probe_camera_device():
                self.status_badges["camera"].config(text="● CONNECTED", fg="#50FA7B")
            else:
                self.status_badges["camera"].config(text="● OFFLINE", fg="#FF5555")

            if stepper_initialized:
                self.status_badges["stepper"].config(text="● READY", fg="#50FA7B")
            else:
                self.status_badges["stepper"].config(text="● ERROR", fg="#FF5555")

            pca_40 = probe_i2c_address(0x40)
            if pca_40:
                if kit is None:
                    try:
                        from adafruit_servokit import ServoKit
                        kit = ServoKit(channels=16, address=0x40)
                        kit.servo[CAMERA_CHANNEL].set_pulse_width_range(500, 2500)
                    except Exception:
                        pass
                self.status_badges["servo_cam"].config(text="● ONLINE (0x40)", fg="#50FA7B")
            else:
                self.status_badges["servo_cam"].config(text="● OFFLINE", fg="#FF5555")

            pca_41 = probe_i2c_address(0x41)
            if pca_40 and pca_41:
                self.status_badges["arm"].config(text="● DUAL DRIVER (0x40/0x41)", fg="#50FA7B")
            elif pca_40:
                self.status_badges["arm"].config(text="● PRIMARY DRIVER (0x40)", fg="#F1FA8C")
            else:
                self.status_badges["arm"].config(text="● OFFLINE", fg="#FF5555")

        def set_controls_state(self, enabled=True):
            st = tk.NORMAL if enabled else tk.DISABLED
            for b in self.buttons.values():
                b.config(state=st)
            for b in self.arm_step_buttons:
                b.config(state=st)
            self.btn_auto_sort.config(state=st)
            self.btn_test_ab.config(state=st)
            self.btn_sync_move.config(state=st)
            self.btn_full_inspect_cycle.config(state=st)
            self.btn_sweep.config(state=st)
            self.btn_center.config(state=st)
            self.btn_sweep_arm.config(state=st)

        def threaded_task(self, target_fn, *args):
            global is_busy, stop_requested
            if is_busy:
                return
            is_busy = True
            stop_requested = False
            self.set_controls_state(False)

            def runner():
                try:
                    target_fn(*args)
                finally:
                    global is_busy
                    is_busy = False
                    self.after(0, lambda: self.set_controls_state(True))
                    self.after(0, lambda: self.lbl_status.config(text="Status: Ready"))

            threading.Thread(target=runner, daemon=True).start()

        # ==========================================================================
        # SERVO POSITION & REVERT LOGIC
        # ==========================================================================
        def move_camera_to_arm(self, arm_num):
            """
            Moves the MG996R camera servo to point precisely at the specified robotic arm.
            Arm #1 = calibrated origin (SERVO_ARM1_ORIGIN_ANGLE).
            Each subsequent arm is SERVO_STEP_DEG further away.
            For 360° continuous mode, pulses forward/backward by counting 1.9cm hops.
            """
            global cam_current_angle, net_continuous_steps, kit, stop_requested
            if kit is None:
                return

            # Build target angle for all 9 arms:
            # Arm #1 = origin, Arm #N = origin + (N-1) * SERVO_STEP_DEG
            target_angle = SERVO_ARM1_ORIGIN_ANGLE + (arm_num - 1) * SERVO_STEP_DEG
            target_angle = max(0, min(175, target_angle))   # Clamp for 180° mode
            target_hops  = arm_num - 1                      # Hops from origin for 360° mode

            mode = self.servo_mode.get()

            if mode == "180_STANDARD":
                # Direct positional move to calibrated arm angle
                if cam_current_angle == target_angle:
                    return
                step_dir = 1 if target_angle > cam_current_angle else -1
                for deg in range(cam_current_angle, target_angle + step_dir, step_dir):
                    if stop_requested:
                        break
                    kit.servo[CAMERA_CHANNEL].angle = deg
                    time.sleep(0.015)
                cam_current_angle = target_angle
                hard_kill_servo()

            else:
                # 360° Continuous: pulse forward or reverse to match target hop count
                if target_hops == net_continuous_steps:
                    pass  # Already at correct position
                elif target_hops > net_continuous_steps:
                    # Need to pulse forward (target_hops - current_hops) steps
                    steps_needed = target_hops - net_continuous_steps
                    try:
                        kit.continuous_servo[CAMERA_CHANNEL].throttle = 0.5
                        time.sleep(self.pulse_duration.get() * steps_needed)
                        net_continuous_steps = target_hops
                    finally:
                        hard_kill_servo()
                else:
                    # Need to pulse backward (current_hops - target_hops) steps
                    steps_back = net_continuous_steps - target_hops
                    try:
                        kit.continuous_servo[CAMERA_CHANNEL].throttle = -0.5
                        time.sleep(self.pulse_duration.get() * steps_back)
                        net_continuous_steps = target_hops
                    finally:
                        hard_kill_servo()

            self.lbl_steps.config(
                text=f"Stepper: {current_step_pos} steps  |  Camera: Arm #{arm_num} ({cam_current_angle}°)"
            )

        def trigger_arm_drop(self, arm_num):
            """
            Triggers the robotic arm drop servo for the given arm number.
            Looks up the PCA9685 channel from ARM_CHANNELS, then:
            1. Moves servo to ARM_DROP_ANGLE  (extend / release)
            2. Holds for 0.6s while leaf falls into bin
            3. Returns servo to ARM_HOME_ANGLE (retract)
            """
            global kit

            channel = ARM_CHANNELS.get(arm_num)
            if channel is None:
                print(f"[WARN] trigger_arm_drop: Arm #{arm_num} not in ARM_CHANNELS map.")
                return

            self.lbl_status.config(
                text=f"Status: 📦 Robotic Arm #{arm_num} — DROP (PCA9685 0x40, ch{channel})..."
            )

            if kit is None:
                self.lbl_status.config(
                    text=f"Status: ⚠️ Arm #{arm_num} drop SKIPPED — PCA9685 offline."
                )
                time.sleep(1.0)
                return

            try:
                # --- EXTEND: move arm servo to drop/release position ---
                kit.servo[channel].angle = ARM_DROP_ANGLE
                time.sleep(0.6)           # Hold while leaf drops into bin

                self.lbl_status.config(
                    text=f"Status: 📦 Robotic Arm #{arm_num} — retracting (ch{channel})..."
                )

                # --- RETRACT: return arm servo to home/ready position ---
                kit.servo[channel].angle = ARM_HOME_ANGLE
                time.sleep(0.4)           # Allow retract to complete

            except Exception as e:
                print(f"[ERROR] Arm #{arm_num} drop failed on 0x40 ch{channel}: {e}")
                try:
                    kit.servo[channel].angle = ARM_HOME_ANGLE  # Safety: always try to retract
                except Exception:
                    pass

            self.lbl_status.config(text=f"Status: ✅ Robotic Arm #{arm_num} — drop complete.")
            time.sleep(0.3)


        def move_servo_forward_1_9cm(self):
            """Single 1.9cm forward step (used by simultaneous-move & inspection buttons)."""
            global cam_current_angle, net_continuous_steps, kit, stop_requested
            if kit is None:
                return

            mode = self.servo_mode.get()
            if mode == "180_STANDARD":
                target_deg = max(0, min(175, int(round(cam_current_angle + SERVO_STEP_DEG))))
                step = 1 if target_deg >= cam_current_angle else -1
                for deg in range(cam_current_angle, target_deg + step, step):
                    if stop_requested:
                        break
                    kit.servo[CAMERA_CHANNEL].angle = deg
                    time.sleep(0.015)
                cam_current_angle = target_deg
            else:
                try:
                    kit.continuous_servo[CAMERA_CHANNEL].throttle = 0.5
                    time.sleep(self.pulse_duration.get())
                    net_continuous_steps += 1
                finally:
                    hard_kill_servo()

        def task_return_camera_to_arm1(self):
            """Returns the camera servo back to Arm #1 (calibrated origin) from any position."""
            self.move_camera_to_arm(1)
            self.lbl_status.config(text="Status: Camera servo at Arm #1 origin (torque released).")

        # ==========================================================================
        # FULL AUTO SORT ALGORITHM
        # ==========================================================================
        def classify_leaf_at_arm(self, arm_num):
            """
            Returns the class letter ('A'-'E') for the leaf at this arm position.
            - SIMULATION MODE ON:  reads from SIMULATION_CLASSES dict (test data)
            - SIMULATION MODE OFF: calls the live camera_classifier.py
            Returns None if no leaf detected or classification fails.
            """
            # ------------------------------------------------------------------
            # SIMULATION MODE: use hardcoded test results
            # ------------------------------------------------------------------
            if self.sim_mode.get():
                result = SIMULATION_CLASSES.get(arm_num)
                display = f"Class {result}" if result else "None (no leaf)"
                for t in range(3, 0, -1):
                    self.lbl_status.config(
                        text=f"Status: 🧪 [SIM] Arm #{arm_num} — scanning... {t}s  →  Result: {display}"
                    )
                    time.sleep(0.5)
                self.lbl_status.config(
                    text=f"Status: 🧪 [SIM] Arm #{arm_num} — identified as {display}"
                )
                return result

            # ------------------------------------------------------------------
            # LIVE MODE: call real camera classifier
            # ------------------------------------------------------------------
            self.lbl_status.config(
                text=f"Status: 🔍 Arm #{arm_num} — capturing frame & classifying..."
            )
            time.sleep(0.3)   # Allow camera to settle at new arm position

            try:
                # Option 1: Import classify_frame() from camera_classifier.py
                from camera_classifier import classify_frame
                result = classify_frame()              # e.g. returns 'D' or 'Class D'
                if result:
                    return result.strip()[-1].upper()  # Extract the class letter
                return None

            except Exception as e:
                print(f"[CLASSIFIER] Error at Arm #{arm_num}: {e}")
                return None

        def task_full_sort_cycle(self):
            """
            FULL AUTOMATIC SORT CYCLE
            =========================
            For each robotic arm (1 → 9):
            1. MG996R camera moves to arm position (1.9cm per step)
            2. Leaf is classified by the camera classifier
            3. Stepper carousel rotates the correct class bin to that arm's drop point
                (BIN_STATIONS[class] + ARM_STEPPER_OFFSETS[arm] for multi-lane setups)
            4. Robotic arm drops the leaf into the bin
            After all arms: camera and carousel both return to origin (Arm #1 / Class A).
            """
            global stop_requested, current_step_pos, current_station_name

            self.lbl_status.config(text="Status: 🤖 AUTO SORT — Homing systems before cycle...")
            execute_rotate('A')
            self.move_camera_to_arm(1)
            self.lbl_station.config(text="Active Station: Class A (Origin)")
            time.sleep(0.5)

            results_log = []   # Keeps a summary: [(arm_num, class, channel), ...]

            for arm_num in range(1, 10):
                if stop_requested:
                    break

                # ----------------------------------------------------------
                # STEP 1: Point MG996R camera at this arm
                # ----------------------------------------------------------
                self.lbl_status.config(
                    text=f"Status: 🤖 [{arm_num}/9] Camera → Arm #{arm_num}..."
                )
                self.move_camera_to_arm(arm_num)

                if stop_requested:
                    break

                # ----------------------------------------------------------
                # STEP 2: Classify the leaf
                # ----------------------------------------------------------
                leaf_class = self.classify_leaf_at_arm(arm_num)

                if leaf_class not in BIN_STATIONS:
                    self.lbl_status.config(
                        text=f"Status: 🤖 [{arm_num}/9] Arm #{arm_num} — no leaf detected, skipping."
                    )
                    time.sleep(0.4)
                    continue

                self.lbl_status.config(
                    text=f"Status: 🤖 [{arm_num}/9] Arm #{arm_num} — Leaf = Class {leaf_class} ✔"
                )
                time.sleep(0.3)

                if stop_requested:
                    break

                # ----------------------------------------------------------
                # STEP 3: Rotate disk so bin {leaf_class} is under Arm #{arm_num}
                # ----------------------------------------------------------
                # Formula (no sensor, open-loop):
                #   target = BIN_STATIONS[bin] + (arm_num - 1) × STEPS_PER_ARM
                #
                # This uses shortest-path CW or CCW rotation from current pos.
                # ----------------------------------------------------------
                target = get_disk_target_steps(leaf_class, arm_num)

                self.lbl_status.config(
                    text=(
                        f"Status: 🤖 [{arm_num}/9] Disk → Class {leaf_class} at Arm #{arm_num}  "
                        f"[{BIN_STATIONS[leaf_class]} + {(arm_num-1)*STEPS_PER_ARM} = {target} steps]"
                    )
                )

                rotate_to_absolute_steps(
                    target,
                    update_ui_cb=lambda m, a=arm_num, c=leaf_class: self.lbl_status.config(
                        text=f"Status: 🤖 [{a}/9] {m}"
                    )
                )

                self.lbl_station.config(
                    text=f"Active Station: Class {leaf_class} (Arm #{arm_num} offset: {offset})"
                )
                self.lbl_steps.config(
                    text=f"Stepper: {current_step_pos} steps  |  Camera: Arm #{arm_num}"
                )

                if stop_requested:
                    break

                # ----------------------------------------------------------
                # STEP 4: Drop the leaf
                # ----------------------------------------------------------
                self.trigger_arm_drop(arm_num)
                results_log.append((arm_num, leaf_class, ARM_CHANNELS.get(arm_num, '?')))

            # ------------------------------------------------------------------
            # DONE: Return everything to calibrated origin
            # ------------------------------------------------------------------
            if not stop_requested:
                self.lbl_status.config(
                    text="Status: 🤖 Sort cycle complete — homing carousel → Class A  |  camera → Arm #1..."
                )
                execute_rotate('A')
                self.move_camera_to_arm(1)

                summary = "  |  ".join([f"Arm#{a}→{c}" for a, c, _ in results_log]) or "(none dropped)"
                self.lbl_station.config(text="Active Station: Class A (Origin)")
                self.lbl_steps.config(
                    text=f"Stepper: Class A (0 steps)  |  Camera: Arm #1 ({SERVO_ARM1_ORIGIN_ANGLE}°)"
                )
                self.lbl_status.config(
                    text=f"✅ AUTO SORT done. {len(results_log)}/9 leaves dropped. {summary}"
                )
        def task_test_class_a_and_b(self):
            """
            TEST WORKFLOW — Arm #1 → Class A  |  Arm #2 → Class B
            =========================================================
            Flow:
            1. Reset: Stepper → Class A, Camera → Arm #1 (origin)
            2. Camera points at Arm #1 → Scan 5s → Robotic Arm #1 drops leaf into Class A
            3. Simultaneously: Stepper → Class B  +  Camera → Arm #2
            4. Camera points at Arm #2 → Scan 5s → Robotic Arm #2 drops leaf into Class B
            5. Auto-home: Stepper → Class A, Camera → Arm #1
            """
            global current_step_pos, current_station_name, cam_current_angle, stop_requested

            # ------------------------------------------------------------------
            # STEP 1: Home both systems to calibrated origin before starting
            # ------------------------------------------------------------------
            self.lbl_status.config(text="Status: [TEST A→B] 🔄 Homing — Carousel → Class A  |  Camera → Arm #1...")
            execute_rotate('A')
            self.move_camera_to_arm(1)          # Move MG996R precisely to Arm #1
            self.lbl_station.config(text="Active Station: Class A (Origin)")
            time.sleep(0.5)

            if stop_requested:
                return

            # ------------------------------------------------------------------
            # STEP 2: Scan at Arm #1 / Class A (5-second inspection window)
            # ------------------------------------------------------------------
            for remaining in range(5, 0, -1):
                if stop_requested:
                    return
                self.lbl_status.config(
                    text=f"Status: [TEST A→B] 🔍 Camera at Arm #1 — scanning Class A ({remaining}s)..."
                )
                self.lbl_steps.config(
                    text=f"Stepper: Class A (0 steps)  |  Camera: Arm #1 ({cam_current_angle}°)"
                )
                time.sleep(1.0)

            if stop_requested:
                return

            # ------------------------------------------------------------------
            # STEP 3: Robotic Arm #1 drops leaf into Class A bin
            # ------------------------------------------------------------------
            self.trigger_arm_drop(arm_num=1)

            if stop_requested:
                return

            # ------------------------------------------------------------------
            # STEP 4: Simultaneously move Stepper → Class B  AND  Camera → Arm #2
            # ------------------------------------------------------------------
            self.lbl_status.config(
                text="Status: [TEST A→B] ⚡ Stepper → Class B  |  MG996R → Arm #2 (1.9cm)..."
            )

            def stepper_to_b():
                execute_rotate('B', update_ui_cb=lambda m: self.lbl_status.config(
                    text=f"Status: [TEST A→B] ⚡ {m}"
                ))

            def camera_to_arm2():
                self.move_camera_to_arm(2)      # Move MG996R precisely to Arm #2

            t1 = threading.Thread(target=stepper_to_b,  daemon=True)
            t2 = threading.Thread(target=camera_to_arm2, daemon=True)
            t1.start(); t2.start()
            t1.join();  t2.join()

            self.lbl_station.config(text=f"Active Station: Class {current_station_name}")
            self.lbl_steps.config(
                text=f"Stepper: Class B ({current_step_pos} steps)  |  Camera: Arm #2 ({cam_current_angle}°)"
            )

            if stop_requested:
                return

            # ------------------------------------------------------------------
            # STEP 5: Scan at Arm #2 / Class B (5-second inspection window)
            # ------------------------------------------------------------------
            for remaining in range(5, 0, -1):
                if stop_requested:
                    return
                self.lbl_status.config(
                    text=f"Status: [TEST A→B] 🔍 Camera at Arm #2 — scanning Class B ({remaining}s)..."
                )
                time.sleep(1.0)

            if stop_requested:
                return

            # ------------------------------------------------------------------
            # STEP 6: Robotic Arm #2 drops leaf into Class B bin
            # ------------------------------------------------------------------
            self.trigger_arm_drop(arm_num=2)

            if stop_requested:
                return

            # ------------------------------------------------------------------
            # STEP 7: Auto-home both systems back to calibrated origin
            # ------------------------------------------------------------------
            self.lbl_status.config(
                text="Status: [TEST A→B] 🔄 Complete! Homing — Carousel → Class A  |  Camera → Arm #1..."
            )
            execute_rotate('A')
            self.move_camera_to_arm(1)          # Return MG996R precisely to Arm #1

            self.lbl_station.config(text="Active Station: Class A (Origin)")
            self.lbl_steps.config(
                text=f"Stepper: Class A (0 steps)  |  Camera: Arm #1 ({SERVO_ARM1_ORIGIN_ANGLE}°)"
            )
            self.lbl_status.config(text="✅ [TEST A→B] Done. All systems at calibrated origin.")

        # ==========================================================================
        # CAROUSEL & SERVO ACTION HANDLERS
        # ==========================================================================
        def task_simultaneous_move(self, target_class):
            global current_step_pos, current_station_name, cam_current_angle

            self.lbl_status.config(text=f"Status: Stepper moving to Class {target_class} | Servo stepping 1.9cm + 5s hold...")

            def stepper_worker():
                execute_rotate(target_class, update_ui_cb=lambda m: self.lbl_status.config(text=f"Status: {m}"))

            def servo_worker():
                self.move_servo_forward_1_9cm()
                hard_kill_servo()
                for remaining in range(5, 0, -1):
                    if stop_requested:
                        break
                    self.lbl_steps.config(text=f"Absolute Step: {current_step_pos} pulses | Servo Held: {cam_current_angle}° ({remaining}s remaining)")
                    time.sleep(1.0)

            t1 = threading.Thread(target=stepper_worker, daemon=True)
            t2 = threading.Thread(target=servo_worker, daemon=True)
            t1.start()
            t2.start()
            t1.join()
            t2.join()

            self.lbl_station.config(text=f"Active Station: Class {current_station_name}")
            self.lbl_steps.config(text=f"Absolute Step Position: {current_step_pos} pulses | Camera: {cam_current_angle}°")
            self.lbl_status.config(text=f"Status: Class {current_station_name} reached & 5s Servo hold finished.")

        def handle_class_button_click(self, class_code):
            if self.sync_servo_toggle.get():
                self.threaded_task(self.task_simultaneous_move, class_code)
            else:
                self.threaded_task(self.task_rotate_to, class_code)

        def task_rotate_to(self, target_name):
            execute_rotate(target_name, update_ui_cb=lambda m: self.lbl_status.config(text=f"Status: {m}"))
            self.lbl_station.config(text=f"Active Station: Class {current_station_name}")
            self.lbl_steps.config(text=f"Absolute Step Position: {current_step_pos} pulses | Camera: {cam_current_angle}°")

        def task_execute_servo_step_with_5s_stop(self, step_num, val_1_9cm):
            """Moves servo to step_num position (relative +1.9cm from current) then holds 5s."""
            global cam_current_angle, net_continuous_steps, kit, stop_requested
            if kit is None:
                self.lbl_status.config(text="Status: ERROR - PCA9685 Offline!")
                return

            mode = self.servo_mode.get()
            if mode == "180_STANDARD":
                # Relative increment: each button step advances +SERVO_STEP_DEG from current position
                target_deg = max(0, min(175, cam_current_angle + SERVO_STEP_DEG))
                self.lbl_status.config(text=f"Status: Advancing 180° Servo +{SERVO_STEP_DEG}° → {target_deg}° (Arm {step_num})...")

                step_dir = 1 if target_deg >= cam_current_angle else -1
                for deg in range(cam_current_angle, target_deg + step_dir, step_dir):
                    if stop_requested:
                        break
                    kit.servo[CAMERA_CHANNEL].angle = deg
                    time.sleep(0.015)

                cam_current_angle = target_deg
            else:
                self.lbl_status.config(text=f"Status: Pulsing 360° Servo 1.9cm (Arm {step_num})...")
                try:
                    kit.continuous_servo[CAMERA_CHANNEL].throttle = 0.5
                    time.sleep(self.pulse_duration.get())
                    net_continuous_steps += 1
                finally:
                    hard_kill_servo()

            hard_kill_servo()
            for remaining in range(5, 0, -1):
                if stop_requested:
                    break
                self.lbl_status.config(text=f"Status: Arm {step_num} ({val_1_9cm}cm) | Stopped & Holding for {remaining}s...")
                time.sleep(1.0)

            self.lbl_status.config(text=f"Status: Arm {step_num} 5-second hold complete.")

        def task_auto_sweep_9_steps(self):
            self.lbl_status.config(text="Status: Running full 9-arm sweep (1.9cm + 5s stop at each)...")
            for i in range(1, 10):
                if stop_requested:
                    break
                val = round(i * 1.9, 1)
                self.task_execute_servo_step_with_5s_stop(i, val)

            self.task_return_camera_to_arm1()
            self.lbl_status.config(text="Status: 9-arm sweep complete. Reverted to Arm #1.")

        def task_full_inspection_cycle(self):
            global stop_requested
            self.lbl_status.config(text="Status: Starting 5-Station Cycle (A➔E)...")
            time.sleep(0.5)

            for target_bin in ['A', 'B', 'C', 'D', 'E']:
                if stop_requested:
                    break
                self.task_simultaneous_move(target_bin)

            if not stop_requested:
                self.lbl_status.config(text="Status: Cycle Complete. Reverting to Class A & Arm #1...")
                execute_rotate('A')
                self.task_return_camera_to_arm1()
                self.lbl_station.config(text="Active Station: Class A (Origin)")
                self.lbl_steps.config(text=f"Absolute Step Position: 0 pulses | Camera: {SERVO_ARM1_ORIGIN_ANGLE}° (Arm #1)")
                self.lbl_status.config(text="Status: Reset to original state complete.")

        def prompt_simultaneous_move(self):
            dlg = tk.Toplevel(self)
            dlg.title("Simultaneous Move")
            dlg.geometry("300x180")
            dlg.configure(bg="#282A36")
            dlg.transient(self)
            dlg.grab_set()

            tk.Label(
                dlg, text="Select Target Class for\nSimultaneous Stepper + 5s Servo Stop:",
                fg="#F8F8F2", bg="#282A36", font=("Arial", 10, "bold")
            ).pack(pady=10)

            combo = ttk.Combobox(dlg, values=['A', 'B', 'C', 'D', 'E'], state="readonly", font=("Arial", 11))
            combo.set('B')
            combo.pack(pady=5)

            def confirm():
                target = combo.get()
                dlg.destroy()
                self.threaded_task(self.task_simultaneous_move, target)

            tk.Button(
                dlg, text="Start Move", bg="#50FA7B", fg="#282A36",
                font=("Arial", 10, "bold"), command=confirm
            ).pack(pady=12)

        def task_instant_brake(self):
            hard_kill_servo()
            self.lbl_status.config(text="Status: Servo output killed (PWM duty cycle = 0).")

        def task_automated_sweep(self):
            execute_rotate('A')
            time.sleep(0.5)
            for b in ['A', 'B', 'C', 'D', 'E']:
                execute_rotate(b, update_ui_cb=lambda m, name=b: self.lbl_status.config(text=f"Status: Sweeping Class {name}..."))
                time.sleep(0.8)
            execute_rotate('A')
            self.lbl_station.config(text="Active Station: Class A (Origin)")
            self.lbl_steps.config(text=f"Absolute Step Position: 0 pulses | Camera: {cam_current_angle}°")

        def safe_exit_to_class_a(self):
            """
            On close:
            1. Save all calibration data to calibration.json.
            2. Revert disk to Class A (origin / 0 steps) and servo to Arm #1.
            3. Release GPIO and destroy the window.
            """
            global is_busy
            if is_busy:
                stop_all_hardware()
                time.sleep(0.3)

            self.lbl_status.config(text="Status: Saving calibration & resetting hardware before exit...")

            def exit_worker():
                stop_all_hardware()
                print("[EXIT] Reverting Stepper Carousel to Class A...")
                execute_rotate('A')
                print("[EXIT] Reverting Camera Servo to Robotic Arm #1...")
                self.task_return_camera_to_arm1()
                time.sleep(0.3)

                # Save AFTER homing so last_step_pos reflects the Class A position.
                # Next launch will restore this and know exactly where the disk is.
                save_calibration()

                GPIO.output(STEP_PIN, GPIO.LOW)
                GPIO.output(DIR_PIN, GPIO.LOW)
                GPIO.output(EN_PIN, GPIO.HIGH)  # High impedance / release coils
                GPIO.cleanup()
                self.after(0, self.destroy)

            threading.Thread(target=exit_worker, daemon=True).start()

    # ==============================================================================
    # ENTRY POINT
    # ==============================================================================
    if __name__ == "__main__":
        app = CarouselApp()
        app.mainloop()
