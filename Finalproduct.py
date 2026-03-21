import cv2
import time
import json
import threading
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import math
import redis  # Add this import

# ───────────── CONFIG ─────────────
VIDEO_SOURCES = ["video1.mp4", "video2.mp4", "video3.mp4", "video4.mp4"]

VEHICLE_MODEL_PATH = "vehicle1.pt"
AMBULANCE_MODEL_PATH = "ambulance.pt"

TILE_W, TILE_H = 640, 360

# Redis Cloud Configuration
REDIS_CONFIG = {
    'host': 'redis-13746.crce283.ap-south-1-2.ec2.cloud.redislabs.com',
    'port': 13746,
    'username': 'default',
    'password': 'Pd37kH1plN8NMSjXtz6shgmHgxYCfTVG',
    'decode_responses': True
}

# Global Redis client for all lanes to share (optional)
global_redis_client = None
try:
    global_redis_client = redis.Redis(**REDIS_CONFIG)
    global_redis_client.ping()
    print("✅ Global Redis Cloud connection established")
except Exception as e:
    print(f"❌ Global Redis connection failed: {e}")

# Lane directions based on your sketch
LANE_DIRECTIONS = {
    1: "UP",     # Top road (vehicles moving upward)
    2: "DOWN",   # Bottom road (vehicles moving downward)
    3: "LEFT",   # Right road (vehicles moving left)
    4: "RIGHT"   # Left road (vehicles moving right)
}

# Lane names mapping (A, B, C, D)
LANE_NAMES = {
    1: "A",
    2: "B", 
    3: "C",
    4: "D"
}

# Vehicle types mapping for regular vehicles
VEHICLE_TYPES = {
    0: "two_wheelers",   # Assuming class 0 is bikes/motorcycles
    1: "cars",           # Assuming class 1 is cars
    2: "trucks",         # Assuming class 2 is trucks
    3: "heavy"           # Assuming class 3 is heavy vehicles
}

# Ambulance class ID (adjust based on your ambulance model)
AMBULANCE_CLASS_ID = 0  # Assuming class 0 in ambulance model is ambulance

# Phase mapping based on vehicle direction within lane
PHASE_MAPPING = {
    "LEFT": "left",
    "CENTER": "straight", 
    "RIGHT": "right"
}

# Lane division ratios (for splitting lane into 3 parts)
LEFT_RATIO = 0.33    # Left 33% of lane
CENTER_RATIO = 0.34  # Center 34% of lane  
RIGHT_RATIO = 0.33   # Right 33% of lane

# Signal timing configuration (you can modify this logic)
SIGNAL_STATES = {
    "A": {"left": "GREEN", "straight": "GREEN", "right": "RED"},
    "B": {"left": "GREEN", "straight": "RED", "right": "GREEN"},
    "C": {"left": "GREEN", "straight": "RED", "right": "GREEN"},
    "D": {"left": "RED", "straight": "GREEN", "right": "RED"}
}

# ───────────── LANE POSITION TRACKER ─────────────
class PositionTracker:
    def __init__(self):
        self.history = defaultdict(lambda: deque(maxlen=5))  # Store last 5 positions
        self.lane_positions = defaultdict(str)  # Store current lane position
        self.seen_vehicles = set()  # Track all vehicles ever seen to avoid double counting
        self.seen_ambulances = set()  # Track ambulances separately

    def update(self, tid, cx, cy, lane_width, lane_x_start):
        """Update position history for a tracked vehicle and determine lane position"""
        self.history[tid].append((cx, cy))
        
        # Calculate relative position within the lane (0 to 1)
        relative_x = (cx - lane_x_start) / lane_width
        
        # Determine which part of the lane the vehicle is in
        if relative_x < LEFT_RATIO:
            position = "LEFT"
        elif relative_x < (LEFT_RATIO + CENTER_RATIO):
            position = "CENTER"
        else:
            position = "RIGHT"
        
        self.lane_positions[tid] = position
        return position

    def is_new_vehicle(self, tid):
        """Check if this is a new vehicle we haven't seen before"""
        if tid not in self.seen_vehicles:
            self.seen_vehicles.add(tid)
            return True
        return False

    def is_new_ambulance(self, tid):
        """Check if this is a new ambulance we haven't seen before"""
        if tid not in self.seen_ambulances:
            self.seen_ambulances.add(tid)
            return True
        return False

    def get_position(self, tid):
        """Get current lane position for a vehicle"""
        return self.lane_positions.get(tid, "CENTER")

    def cleanup_old_tracks(self, active_ids):
        """Remove tracks that are no longer active"""
        inactive_ids = set(self.history.keys()) - active_ids
        for tid in inactive_ids:
            if tid in self.history:
                del self.history[tid]
            if tid in self.lane_positions:
                del self.lane_positions[tid]

# ───────────── LANE STATE ─────────────
class LaneState:
    def __init__(self, lane_id):
        self.lane_id = lane_id
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        """Reset only the vehicle counts, keep cumulative counts"""
        with self.lock:
            # Initialize cumulative counts
            if not hasattr(self, 'cumulative_vehicles'):
                self.cumulative_vehicles = {
                    "left": {"two_wheelers": 0, "cars": 0, "trucks": 0, "heavy": 0},
                    "straight": {"two_wheelers": 0, "cars": 0, "trucks": 0, "heavy": 0},
                    "right": {"two_wheelers": 0, "cars": 0, "trucks": 0, "heavy": 0}
                }
            
            # Initialize ambulance tracking
            if not hasattr(self, 'ambulance_detected'):
                self.ambulance_detected = False
                self.ambulance_count = 0
                self.ambulance_positions = []  # Track positions of ambulances
                self.current_ambulance_lane = None
                self.current_ambulance_direction = None
            
            # Create phases with cumulative counts
            self.phases = {
                "left": {
                    "signal_state": SIGNAL_STATES[LANE_NAMES[self.lane_id]]["left"],
                    "vehicles": self.cumulative_vehicles["left"].copy()
                },
                "straight": {
                    "signal_state": SIGNAL_STATES[LANE_NAMES[self.lane_id]]["straight"],
                    "vehicles": self.cumulative_vehicles["straight"].copy()
                },
                "right": {
                    "signal_state": SIGNAL_STATES[LANE_NAMES[self.lane_id]]["right"],
                    "vehicles": self.cumulative_vehicles["right"].copy()
                }
            }

    def increment_count(self, phase, vehicle_type):
        """Increment count for specific phase and vehicle type"""
        with self.lock:
            self.cumulative_vehicles[phase][vehicle_type] += 1
            self.phases[phase]["vehicles"][vehicle_type] = self.cumulative_vehicles[phase][vehicle_type]

    def set_ambulance_detected(self, detected=True, position=None):
        """Set ambulance detection flag and position"""
        with self.lock:
            self.ambulance_detected = detected
            if detected:
                self.ambulance_count += 1
                self.current_ambulance_lane = LANE_NAMES[self.lane_id]
                self.current_ambulance_direction = LANE_DIRECTIONS[self.lane_id]
                if position:
                    self.ambulance_positions.append({
                        'position': position,
                        'timestamp': time.time(),
                        'lane': LANE_NAMES[self.lane_id],
                        'direction': LANE_DIRECTIONS[self.lane_id]
                    })
            else:
                self.current_ambulance_lane = None
                self.current_ambulance_direction = None
            
            # Keep only last 10 ambulance detections
            if len(self.ambulance_positions) > 10:
                self.ambulance_positions = self.ambulance_positions[-10:]

    def get_ambulance_status(self):
        """Get current ambulance detection status"""
        with self.lock:
            return {
                'detected': self.ambulance_detected,
                'count': self.ambulance_count,
                'lane': self.current_ambulance_lane,
                'direction': self.current_ambulance_direction,
                'recent_positions': self.ambulance_positions[-3:]  # Last 3 positions
            }

    def snapshot(self):
        """Get current snapshot of lane state"""
        with self.lock:
            return {
                "lane_id": self.lane_id,
                "lane_name": LANE_NAMES[self.lane_id],
                "phases": self.phases.copy(),
                "ambulance": {
                    "detected": self.ambulance_detected,
                    "count": self.ambulance_count,
                    "lane": self.current_ambulance_lane,
                    "direction": self.current_ambulance_direction
                },
                "timestamp": time.time()
            }

# ───────────── DISPLAY ─────────────
def annotate(frame, state, lane_dir):
    tile = cv2.resize(frame, (TILE_W, TILE_H))
    snap = state.snapshot()
    
    h, w = tile.shape[:2]

    # Draw lane divisions (visual guides)
    left_boundary = int(w * LEFT_RATIO)
    center_boundary = int(w * (LEFT_RATIO + CENTER_RATIO))
    
    # Draw vertical lines to show lane divisions
    cv2.line(tile, (left_boundary, 0), (left_boundary, h), (255, 255, 0), 1)
    cv2.line(tile, (center_boundary, 0), (center_boundary, h), (255, 255, 0), 1)
    
    # Add labels for each section
    cv2.putText(tile, "LEFT", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(tile, "STRAIGHT", (left_boundary + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(tile, "RIGHT", (center_boundary + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Background for text (semi-transparent)
    overlay = tile.copy()
    
    # Adjust background height based on ambulance detection
    bg_height = 260 if snap['ambulance']['detected'] else 200
    cv2.rectangle(overlay, (0, h-bg_height), (280, h-10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, tile, 0.3, 0, tile)

    # Lane info
    lane_name = LANE_NAMES[snap['lane_id']]
    cv2.putText(tile, f"Lane {lane_name} - {lane_dir}", 
                (10, h-230 if snap['ambulance']['detected'] else h-180), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # Ambulance alert if detected - SHOW WHICH LANE
    if snap['ambulance']['detected']:
        y_pos = h-205
        # Flashing red background for ambulance alert
        if int(time.time() * 2) % 2 == 0:  # Flash every 0.5 seconds
            cv2.rectangle(tile, (5, y_pos-5), (275, y_pos+40), (0, 0, 255), -1)
        
        # Show lane information prominently
        amb_lane = snap['ambulance']['lane'] or lane_name
        amb_dir = snap['ambulance']['direction'] or lane_dir
        
        cv2.putText(tile, f"🚑 AMBULANCE IN LANE {amb_lane} 🚑", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
        cv2.putText(tile, f"Direction: {amb_dir}", 
                   (10, y_pos+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(tile, f"Count: {snap['ambulance']['count']}", 
                   (10, y_pos+35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset = y_pos + 50
    else:
        y_offset = h-160

    # Display phase counts
    phase_colors = {
        "left": (255, 0, 0),      # Blue
        "straight": (0, 255, 0),  # Green
        "right": (0, 0, 255)      # Red
    }
    
    for phase_name, phase_data in snap['phases'].items():
        # Phase name and signal state
        signal_color = (0, 255, 0) if phase_data['signal_state'] == "GREEN" else (0, 0, 255)
        cv2.putText(tile, f"{phase_name.upper()}: {phase_data['signal_state']}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, signal_color, 2)
        y_offset += 20
        
        # Vehicle counts for this phase
        for vtype, count in phase_data['vehicles'].items():
            if count > 0:
                cv2.putText(tile, f"  {vtype}: {count}", 
                           (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, phase_colors[phase_name], 1)
                y_offset += 15
        y_offset += 5

    # Add total count
    total = sum(sum(phase['vehicles'].values()) for phase in snap['phases'].values())
    cv2.putText(tile, f"TOTAL: {total}", (10, h-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return tile

# ───────────── DUAL DETECTOR ─────────────
class DualDetector:
    def __init__(self, vehicle_model_path, ambulance_model_path, conf_threshold=0.5):
        self.vehicle_model = YOLO(vehicle_model_path)
        self.ambulance_model = YOLO(ambulance_model_path)
        self.conf_threshold = conf_threshold
        
    def detect(self, frame):
        # Detect regular vehicles
        vehicle_results = self.vehicle_model.track(frame, 
                                                  persist=True, 
                                                  verbose=False,
                                                  conf=self.conf_threshold,
                                                  classes=[0, 1, 2, 3])
        
        # Detect ambulances
        ambulance_results = self.ambulance_model.track(frame,
                                                       persist=True,
                                                       verbose=False,
                                                       conf=self.conf_threshold)
        
        return vehicle_results[0], ambulance_results[0]

    def get_vehicle_type(self, class_id):
        """Map class ID to vehicle type"""
        return VEHICLE_TYPES.get(class_id, "cars")

# ───────────── THREAD ─────────────
class LaneProcessor(threading.Thread):
    def __init__(self, lane_id, src, state):
        super().__init__(daemon=True)
        self.lane_id = lane_id
        self.src = src
        self.state = state
        self.running = True
        self.tile = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
        self.position_tracker = PositionTracker()
        self.detector = DualDetector(VEHICLE_MODEL_PATH, AMBULANCE_MODEL_PATH)
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        self.last_redis_update = time.time()
        self.ambulance_alert_active = False
        self.ambulance_alert_start_time = 0
        
        # Redis client for storing output - Use the global connection or create new one
        self.redis_client = global_redis_client
        if self.redis_client:
            print(f"Lane {lane_id}: Using global Redis connection")
        else:
            # Try to create individual connection
            try:
                self.redis_client = redis.Redis(
                    host='redis-13746.crce283.ap-south-1-2.ec2.cloud.redislabs.com',
                    port=13746,
                    username='default',
                    password='Pd37kH1plN8NMSjXtz6shgmHgxYCfTVG',
                    decode_responses=True
                )
                self.redis_client.ping()
                print(f"Lane {lane_id}: Individual Redis Cloud connection successful")
            except Exception as e:
                print(f"Lane {lane_id}: Redis connection failed - {e}")
                self.redis_client = None

    def update_redis(self):
        """Update Redis with current lane data (throttled to max 10 updates per second)"""
        current_time = time.time()
        # Update Redis at most every 0.1 seconds (10 fps)
        if current_time - self.last_redis_update < 0.1:
            return
            
        if self.redis_client:
            try:
                snapshot = self.state.snapshot()
                lane_name = snapshot['lane_name']
                
                # Get current full data from Redis
                key = "traffic_intersection"
                current_data = self.redis_client.get(key)
                
                if current_data:
                    data = json.loads(current_data)
                else:
                    data = {"lanes": {}, "ambulance_alerts": {}}
                
                # Update this lane's data
                data["lanes"][lane_name] = {
                    "phases": snapshot['phases'],
                    "direction": LANE_DIRECTIONS[self.lane_id]
                }
                
                # Add ambulance alert with detailed lane information
                if snapshot['ambulance']['detected']:
                    if "ambulance_alerts" not in data:
                        data["ambulance_alerts"] = {}
                    
                    amb_lane = snapshot['ambulance']['lane'] or lane_name
                    amb_dir = snapshot['ambulance']['direction'] or LANE_DIRECTIONS[self.lane_id]
                    
                    data["ambulance_alerts"][lane_name] = {
                        "detected": True,
                        "count": snapshot['ambulance']['count'],
                        "timestamp": snapshot['timestamp'],
                        "lane": amb_lane,
                        "lane_direction": amb_dir,
                        "message": f"🚑 Emergency vehicle in Lane {amb_lane} heading {amb_dir}"
                    }
                    
                    # Also update signal states to prioritize ambulance lane
                    self.prioritize_ambulance_lane(data, amb_lane)
                
                # Store back to Redis (no expiry - counts persist)
                self.redis_client.set(key, json.dumps(data))
                self.last_redis_update = current_time
                
                # Print debug info every 30 seconds
                if int(current_time) % 30 == 0:
                    total_vehicles = sum(sum(phase['vehicles'].values()) for phase in snapshot['phases'].values())
                    print(f"Lane {lane_name}: Updated Redis - Total: {total_vehicles}, Ambulance: {snapshot['ambulance']['detected']}")
                
            except Exception as e:
                print(f"Lane {self.lane_id}: Redis update error - {e}")

    def prioritize_ambulance_lane(self, data, ambulance_lane):
        """Set all signals to GREEN for the ambulance lane"""
        # This is a simple prioritization - you can make it more sophisticated
        for lane in LANE_NAMES.values():
            if lane in data["lanes"]:
                for phase in data["lanes"][lane]["phases"]:
                    if lane == ambulance_lane:
                        # Ambulance lane gets all GREEN
                        data["lanes"][lane]["phases"][phase]["signal_state"] = "GREEN"
                    else:
                        # Other lanes get all RED when ambulance is detected
                        data["lanes"][lane]["phases"][phase]["signal_state"] = "RED"

    def run(self):
        cap = cv2.VideoCapture(self.src)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source {self.src}")
            return

        print(f"Lane {LANE_NAMES[self.lane_id]} processor started")
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                # Restart video if it ends
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Calculate FPS
            self.frame_count += 1
            if self.frame_count >= 30:
                current_time = time.time()
                self.fps = self.frame_count / (current_time - self.last_time)
                self.last_time = current_time
                self.frame_count = 0

            # Update state for display (but don't reset cumulative counts)
            self.state.reset()
            lane_dir = LANE_DIRECTIONS[self.lane_id]
            lane_name = LANE_NAMES[self.lane_id]
            
            # Get frame dimensions
            h, w = frame.shape[:2]

            # Detect vehicles and ambulances
            ambulance_detected_in_frame = False
            
            try:
                vehicle_results, ambulance_results = self.detector.detect(frame)
                
                # Process regular vehicles
                if vehicle_results.boxes is not None and vehicle_results.boxes.id is not None:
                    active_ids = set()

                    for box, tid, cls_id in zip(
                            vehicle_results.boxes.xyxy.cpu().numpy(),
                            vehicle_results.boxes.id.cpu().numpy().astype(int),
                            vehicle_results.boxes.cls.cpu().numpy().astype(int)):

                        active_ids.add(tid)

                        # Calculate center point
                        cx = int((box[0] + box[2]) / 2)
                        cy = int((box[1] + box[3]) / 2)
                        
                        # Lane boundaries
                        lane_x_start = 0
                        lane_width = w

                        # Determine position within lane
                        position = self.position_tracker.update(tid, cx, cy, lane_width, lane_x_start)
                        phase = PHASE_MAPPING[position]
                        
                        # Get vehicle type
                        vehicle_type = self.detector.get_vehicle_type(cls_id)

                        # Check if this is a new vehicle
                        if self.position_tracker.is_new_vehicle(tid):
                            self.state.increment_count(phase, vehicle_type)
                            print(f"Lane {lane_name}: New {vehicle_type} detected going {phase}")

                        # Draw bounding box with phase-based color
                        colors = {
                            "left": (255, 0, 0),      # Blue
                            "straight": (0, 255, 0),  # Green
                            "right": (0, 0, 255)      # Red
                        }
                        color = colors[phase]
                        
                        cv2.rectangle(frame, 
                                    (int(box[0]), int(box[1])), 
                                    (int(box[2]), int(box[3])), 
                                    color, 2)
                        
                        # Add vehicle info label
                        label = f"ID:{tid}-{vehicle_type[:3]}-{phase[:1]}"
                        cv2.putText(frame, label, 
                                   (int(box[0]), int(box[1]-5)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                    # Clean up old tracks
                    self.position_tracker.cleanup_old_tracks(active_ids)

                # Process ambulances
                if ambulance_results.boxes is not None and ambulance_results.boxes.id is not None:
                    for box, tid in zip(
                            ambulance_results.boxes.xyxy.cpu().numpy(),
                            ambulance_results.boxes.id.cpu().numpy().astype(int)):

                        # Calculate center point for ambulance
                        cx = int((box[0] + box[2]) / 2)
                        cy = int((box[1] + box[3]) / 2)
                        
                        # Determine position
                        position = self.position_tracker.update(f"amb_{tid}", cx, cy, w, 0)
                        phase = PHASE_MAPPING[position]
                        
                        # Check if this is a new ambulance
                        if self.position_tracker.is_new_ambulance(f"amb_{tid}"):
                            self.state.set_ambulance_detected(True, position)
                            ambulance_detected_in_frame = True
                            
                            # Get lane and direction information
                            lane_name = LANE_NAMES[self.lane_id]
                            lane_direction = LANE_DIRECTIONS[self.lane_id]
                            
                            # Clear and specific ambulance alert message
                            print("\n" + "=" * 70)
                            print(f"🚨🚨🚨 AMBULANCE DETECTED - EMERGENCY 🚨🚨🚨")
                            print(f"📍 LANE: {lane_name}")
                            print(f"➡️  DIRECTION: {lane_direction}")
                            print(f"🔄 TURNING: {phase.upper()}")
                            print(f"📌 POSITION IN LANE: {position}")
                            print("=" * 70 + "\n")
                            
                            # Start alert timer
                            self.ambulance_alert_active = True
                            self.ambulance_alert_start_time = time.time()

                        # Draw ambulance with special highlighting
                        # Magenta bounding box for ambulance
                        cv2.rectangle(frame, 
                                    (int(box[0]), int(box[1])), 
                                    (int(box[2]), int(box[3])), 
                                    (255, 0, 255), 3)  # Magenta
                        
                        # Add ambulance label with lane info and flashing effect
                        if int(time.time() * 2) % 2 == 0:
                            lane_name = LANE_NAMES[self.lane_id]
                            cv2.putText(frame, f"🚑 AMBULANCE Lane {lane_name} 🚑", 
                                       (int(box[0]), int(box[1]-10)),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Update ambulance detection state
                if not ambulance_detected_in_frame and self.ambulance_alert_active:
                    # Keep alert active for 5 seconds after last detection
                    if time.time() - self.ambulance_alert_start_time > 5:
                        self.ambulance_alert_active = False
                        self.state.set_ambulance_detected(False)

                # Update Redis with current counts
                self.update_redis()

            except Exception as e:
                print(f"Error in lane {self.lane_id}: {e}")

            # Add FPS and lane info to frame
            cv2.putText(frame, f"Lane {lane_name} - FPS: {self.fps:.1f}", 
                       (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # Add ambulance alert to frame if active with lane information
            if self.ambulance_alert_active:
                lane_name = LANE_NAMES[self.lane_id]
                lane_direction = LANE_DIRECTIONS[self.lane_id]
                
                # Add prominent alert at top of frame
                alert_y = 80
                if int(time.time() * 2) % 2 == 0:
                    cv2.rectangle(frame, (0, alert_y-25), (frame.shape[1], alert_y+35), (0, 0, 255), -1)
                
                # Multi-line alert with lane information
                cv2.putText(frame, f"🚑 EMERGENCY VEHICLE IN LANE {lane_name} 🚑", 
                           (50, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Direction: {lane_direction} | Please clear the way!", 
                           (50, alert_y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Create annotated tile
            self.tile = annotate(frame.copy(), self.state, lane_dir)

        cap.release()
        print(f"Lane {lane_name} processor stopped")

    def get_tile(self):
        return self.tile

    def stop(self):
        self.running = False

# ───────────── MAIN ─────────────
def main():
    print("=" * 70)
    print("TRAFFIC INTERSECTION MONITORING SYSTEM WITH AMBULANCE DETECTION")
    print("=" * 70)
    print(f"Processing {len(VIDEO_SOURCES)} video streams")
    print("Tracking vehicle types: two_wheelers, cars, trucks, heavy")
    print("Tracking directions: LEFT, STRAIGHT, RIGHT")
    print("🚑 AMBULANCE DETECTION ENABLED 🚑")
    print("Lane mapping: 1->A, 2->B, 3->C, 4->D")
    
    # Redis status
    if global_redis_client:
        print("\n✅ Redis Cloud: CONNECTED - Data will be stored")
    else:
        print("\n❌ Redis Cloud: NOT CONNECTED - Check your connection settings")
    
    print("=" * 70)
    
    # Initialize lane states and threads
    states = [LaneState(i+1) for i in range(4)]
    threads = []

    # Start processing threads
    for i in range(4):
        print(f"Starting lane {LANE_NAMES[i+1]} processor...")
        t = LaneProcessor(i+1, VIDEO_SOURCES[i], states[i])
        threads.append(t)
        t.start()
        time.sleep(0.5)  # Small delay between thread starts

    # Create window
    cv2.namedWindow("Traffic Intersection Monitor", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Traffic Intersection Monitor", 1280, 720)

    print("\n" + "=" * 70)
    print("System running. Press ESC to exit.")
    print("\nRedis Data: Check your dashboard at http://localhost:3000")
    print("=" * 70)

    # Main display loop
    while True:
        try:
            # Get tiles from all lanes
            tiles = [t.get_tile() for t in threads]
            
            # Create 2x2 grid
            top_row = np.hstack([tiles[0], tiles[1]])
            bottom_row = np.hstack([tiles[2], tiles[3]])
            grid = np.vstack([top_row, bottom_row])

            # Display
            cv2.imshow("Traffic Intersection Monitor", grid)

            # Check for exit (ESC key)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Display error: {e}")
            continue

    # Cleanup
    print("\nShutting down...")
    for t in threads:
        t.stop()

    # Wait for threads to finish
    for t in threads:
        t.join(timeout=2.0)

    cv2.destroyAllWindows()
    
    # Print final counts including ambulances
    print("\n" + "=" * 70)
    print("FINAL CUMULATIVE COUNTS:")
    print("=" * 70)
    total_ambulances = 0
    for i, state in enumerate(states):
        lane_name = LANE_NAMES[i+1]
        snap = state.snapshot()
        print(f"\nLane {lane_name}:")
        ambulance_status = state.get_ambulance_status()
        if ambulance_status['count'] > 0:
            print(f"  🚑 AMBULANCES DETECTED: {ambulance_status['count']} 🚑")
            total_ambulances += ambulance_status['count']
        for phase_name, phase_data in snap['phases'].items():
            total = sum(phase_data['vehicles'].values())
            print(f"  {phase_name.upper()}: {total} vehicles")
            for vtype, count in phase_data['vehicles'].items():
                if count > 0:
                    print(f"    {vtype}: {count}")
    
    print(f"\nTOTAL AMBULANCES ACROSS ALL LANES: {total_ambulances}")
    print("\nSystem stopped successfully.")

if __name__ == "__main__":
    main()