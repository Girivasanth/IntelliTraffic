import cv2
import time
import json
import threading
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import math
import redis

# ───────────── CONFIG ─────────────
VIDEO_SOURCES = ["video1.mp4", "video2.mp4", "video3.mp4", "video4.mp4"]

VEHICLE_MODEL_PATH = "vehicle1.pt"
AMBULANCE_MODEL_PATH = "ambulance.pt"

TILE_W, TILE_H = 640, 360

# Redis Cloud Configuration
REDIS_CONFIG = {
    'host':                   'redis-13746.crce283.ap-south-1-2.ec2.cloud.redislabs.com',
    'port':                   13746,
    'username':               'default',
    'password':               'Pd37kH1plN8NMSjXtz6shgmHgxYCfTVG',
    'decode_responses':       True,
    'socket_timeout':         10,
    'socket_connect_timeout': 10,
    'retry_on_timeout':       True,
    'health_check_interval':  30,   # auto-reconnect if connection drops
}

# Redis key constants (all stored as JSON strings)
REDIS_KEY_INTERSECTION = "traffic:intersection"  # Full intersection data
REDIS_KEY_AMBULANCE    = "traffic:ambulance"      # Active ambulance alert
REDIS_KEY_LANE_PREFIX  = "traffic:lane:"          # Per-lane: traffic:lane:A … D

global_redis_client = None

def init_redis():
    """Connect to Redis, fix any WRONGTYPE keys, and write an initial skeleton."""
    global global_redis_client
    try:
        client = redis.Redis(**REDIS_CONFIG)
        pong = client.ping()
        print(f"✅ Redis PING: {pong}")

        # ── Clean stale keys with wrong type ─────────────────────────────────
        keys_to_reset = (
            [REDIS_KEY_INTERSECTION, REDIS_KEY_AMBULANCE]
            + [f"{REDIS_KEY_LANE_PREFIX}{l}" for l in ["A","B","C","D"]]
            + ["traffic_intersection"]          # old key name from previous version
        )
        for key in keys_to_reset:
            ktype = client.type(key)
            if ktype not in ("string", "none"):
                client.delete(key)
                print(f"⚠️  Deleted stale key '{key}' (was type={ktype})")

        # ── Write initial empty skeleton so UI can read immediately ──────────
        skeleton = {
            "lanes": {
                lane: {
                    "direction": dir_,
                    "phases": {
                        ph: {"signal_state": "RED", "vehicles":
                             {"two_wheelers": 0, "cars": 0, "trucks": 0, "heavy": 0}}
                        for ph in ("left", "straight", "right")
                    },
                    "total_vehicles": 0
                }
                for lane, dir_ in zip(["A","B","C","D"], ["UP","DOWN","LEFT","RIGHT"])
            },
            "ambulance_alerts": {},
            "last_updated": time.time(),
        }
        client.set(REDIS_KEY_INTERSECTION, json.dumps(skeleton))
        client.set(REDIS_KEY_AMBULANCE, json.dumps({"active": False, "lane": None, "direction": None}))
        for lane in ["A","B","C","D"]:
            client.set(f"{REDIS_KEY_LANE_PREFIX}{lane}", json.dumps({
                "lane": lane, "total_vehicles": 0, "ambulance": {"detected": False}
            }))

        print("✅ Redis skeleton written. All traffic:* keys initialised.")
        print(f"   Keys: {client.keys('traffic:*')}")

        global_redis_client = client

    except redis.AuthenticationError as e:
        print(f"❌ Redis AUTH failed (wrong password?): {e}")
    except redis.ConnectionError as e:
        print(f"❌ Redis CONNECTION failed (wrong host/port?): {e}")
    except Exception as e:
        print(f"❌ Redis init error: {e}")

init_redis()

# ───────────── LANE CONFIG ─────────────
LANE_DIRECTIONS = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT"}
LANE_NAMES      = {1: "A",  2: "B",    3: "C",     4: "D"}

VEHICLE_TYPES = {
    0: "two_wheelers",
    1: "cars",
    2: "trucks",
    3: "heavy"
}

AMBULANCE_CLASS_ID = 0

PHASE_MAPPING = {"LEFT": "left", "CENTER": "straight", "RIGHT": "right"}

LEFT_RATIO   = 0.33
CENTER_RATIO = 0.34
RIGHT_RATIO  = 0.33

SIGNAL_STATES = {
    "A": {"left": "GREEN", "straight": "GREEN",  "right": "RED"},
    "B": {"left": "GREEN", "straight": "RED",    "right": "GREEN"},
    "C": {"left": "GREEN", "straight": "RED",    "right": "GREEN"},
    "D": {"left": "RED",   "straight": "GREEN",  "right": "RED"}
}

# ───────────── POSITION TRACKER ─────────────
class PositionTracker:
    def __init__(self):
        self.history         = defaultdict(lambda: deque(maxlen=5))
        self.lane_positions  = defaultdict(str)
        self.seen_vehicles   = set()
        self.seen_ambulances = set()

    def update(self, tid, cx, cy, lane_width, lane_x_start):
        self.history[tid].append((cx, cy))
        relative_x = (cx - lane_x_start) / lane_width
        if relative_x < LEFT_RATIO:
            position = "LEFT"
        elif relative_x < (LEFT_RATIO + CENTER_RATIO):
            position = "CENTER"
        else:
            position = "RIGHT"
        self.lane_positions[tid] = position
        return position

    def is_new_vehicle(self, tid):
        if tid not in self.seen_vehicles:
            self.seen_vehicles.add(tid)
            return True
        return False

    def is_new_ambulance(self, tid):
        if tid not in self.seen_ambulances:
            self.seen_ambulances.add(tid)
            return True
        return False

    def get_position(self, tid):
        return self.lane_positions.get(tid, "CENTER")

    def cleanup_old_tracks(self, active_ids):
        for tid in set(self.history.keys()) - active_ids:
            self.history.pop(tid, None)
            self.lane_positions.pop(tid, None)

# ───────────── LANE STATE ─────────────
class LaneState:
    def __init__(self, lane_id):
        self.lane_id = lane_id
        self.lock    = threading.Lock()
        self.cumulative_vehicles = {
            "left":     {"two_wheelers": 0, "cars": 0, "trucks": 0, "heavy": 0},
            "straight": {"two_wheelers": 0, "cars": 0, "trucks": 0, "heavy": 0},
            "right":    {"two_wheelers": 0, "cars": 0, "trucks": 0, "heavy": 0},
        }
        self.ambulance_detected  = False
        self.ambulance_count     = 0
        self.ambulance_positions = []
        self.current_ambulance_lane      = None
        self.current_ambulance_direction = None
        self._rebuild_phases()

    def _rebuild_phases(self):
        """Re-create phases dict from cumulative counts (call under lock)."""
        lane_name = LANE_NAMES[self.lane_id]
        self.phases = {
            phase: {
                "signal_state": SIGNAL_STATES[lane_name][phase],
                "vehicles":     self.cumulative_vehicles[phase].copy()
            }
            for phase in ("left", "straight", "right")
        }

    def reset(self):
        """Refresh phases display without wiping cumulative counts."""
        with self.lock:
            self._rebuild_phases()

    def increment_count(self, phase, vehicle_type):
        with self.lock:
            self.cumulative_vehicles[phase][vehicle_type] += 1
            self.phases[phase]["vehicles"][vehicle_type] = \
                self.cumulative_vehicles[phase][vehicle_type]

    def set_ambulance_detected(self, detected=True, position=None):
        with self.lock:
            self.ambulance_detected = detected
            if detected:
                self.ambulance_count += 1
                self.current_ambulance_lane      = LANE_NAMES[self.lane_id]
                self.current_ambulance_direction = LANE_DIRECTIONS[self.lane_id]
                if position:
                    self.ambulance_positions.append({
                        'position':  position,
                        'timestamp': time.time(),
                        'lane':      LANE_NAMES[self.lane_id],
                        'direction': LANE_DIRECTIONS[self.lane_id]
                    })
                    self.ambulance_positions = self.ambulance_positions[-10:]
            else:
                self.current_ambulance_lane      = None
                self.current_ambulance_direction = None

    def get_ambulance_status(self):
        with self.lock:
            return {
                'detected':         self.ambulance_detected,
                'count':            self.ambulance_count,
                'lane':             self.current_ambulance_lane,
                'direction':        self.current_ambulance_direction,
                'recent_positions': self.ambulance_positions[-3:]
            }

    def snapshot(self):
        with self.lock:
            return {
                "lane_id":   self.lane_id,
                "lane_name": LANE_NAMES[self.lane_id],
                "phases":    {k: dict(v) for k, v in self.phases.items()},
                "ambulance": {
                    "detected":  self.ambulance_detected,
                    "count":     self.ambulance_count,
                    "lane":      self.current_ambulance_lane,
                    "direction": self.current_ambulance_direction,
                },
                "timestamp": time.time()
            }

# ───────────── DISPLAY ─────────────
def annotate(frame, state, lane_dir):
    tile = cv2.resize(frame, (TILE_W, TILE_H))
    snap = state.snapshot()
    h, w = tile.shape[:2]

    left_boundary  = int(w * LEFT_RATIO)
    center_boundary = int(w * (LEFT_RATIO + CENTER_RATIO))
    cv2.line(tile, (left_boundary,   0), (left_boundary,   h), (255, 255, 0), 1)
    cv2.line(tile, (center_boundary, 0), (center_boundary, h), (255, 255, 0), 1)
    cv2.putText(tile, "LEFT",     (10, 30),                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
    cv2.putText(tile, "STRAIGHT", (left_boundary+10, 30),    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
    cv2.putText(tile, "RIGHT",    (center_boundary+10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

    overlay    = tile.copy()
    bg_height  = 260 if snap['ambulance']['detected'] else 200
    cv2.rectangle(overlay, (0, h-bg_height), (280, h-10), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.7, tile, 0.3, 0, tile)

    lane_name = LANE_NAMES[snap['lane_id']]
    cv2.putText(tile, f"Lane {lane_name} - {lane_dir}",
                (10, h-230 if snap['ambulance']['detected'] else h-180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    if snap['ambulance']['detected']:
        y_pos = h-205
        if int(time.time() * 2) % 2 == 0:
            cv2.rectangle(tile, (5, y_pos-5), (275, y_pos+40), (0,0,255), -1)
        amb_lane = snap['ambulance']['lane'] or lane_name
        amb_dir  = snap['ambulance']['direction'] or lane_dir
        cv2.putText(tile, f"AMBULANCE IN LANE {amb_lane}",
                   (10, y_pos),    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 2)
        cv2.putText(tile, f"Direction: {amb_dir}",
                   (10, y_pos+20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255,255,255), 1)
        cv2.putText(tile, f"Count: {snap['ambulance']['count']}",
                   (10, y_pos+35), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255,255,255), 1)
        y_offset = y_pos + 50
    else:
        y_offset = h-160

    phase_colors = {"left": (255,0,0), "straight": (0,255,0), "right": (0,0,255)}
    for phase_name, phase_data in snap['phases'].items():
        signal_color = (0,255,0) if phase_data['signal_state'] == "GREEN" else (0,0,255)
        cv2.putText(tile, f"{phase_name.upper()}: {phase_data['signal_state']}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, signal_color, 2)
        y_offset += 20
        for vtype, count in phase_data['vehicles'].items():
            if count > 0:
                cv2.putText(tile, f"  {vtype}: {count}",
                           (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, phase_colors[phase_name], 1)
                y_offset += 15
        y_offset += 5

    total = sum(sum(p['vehicles'].values()) for p in snap['phases'].values())
    cv2.putText(tile, f"TOTAL: {total}", (10, h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    return tile

# ───────────── DUAL DETECTOR ─────────────
class DualDetector:
    def __init__(self, vehicle_model_path, ambulance_model_path, conf_threshold=0.5):
        self.vehicle_model   = YOLO(vehicle_model_path)
        self.ambulance_model = YOLO(ambulance_model_path)
        self.conf_threshold  = conf_threshold

    def detect(self, frame):
        v = self.vehicle_model.track(frame,   persist=True, verbose=False,
                                     conf=self.conf_threshold, classes=[0,1,2,3])
        a = self.ambulance_model.track(frame, persist=True, verbose=False,
                                       conf=self.conf_threshold)
        return v[0], a[0]

    def get_vehicle_type(self, class_id):
        return VEHICLE_TYPES.get(class_id, "cars")

# ───────────── REDIS HELPERS ─────────────
def safe_redis_set(client, key, value_dict):
    """
    Safely store a dict as a JSON string.
    If the key exists with the wrong type, delete it first.
    Returns True on success.
    """
    try:
        payload = json.dumps(value_dict)
        client.set(key, payload)
        return True
    except redis.ResponseError as e:
        if "WRONGTYPE" in str(e):
            try:
                client.delete(key)
                client.set(key, json.dumps(value_dict))
                print(f"⚠️  Auto-fixed WRONGTYPE for key '{key}'")
                return True
            except Exception as inner:
                print(f"❌ Could not fix key '{key}': {inner}")
        else:
            print(f"❌ Redis SET error for '{key}': {e}")
        return False

def safe_redis_get(client, key):
    """Return parsed dict or None. Handles WRONGTYPE gracefully."""
    try:
        raw = client.get(key)
        return json.loads(raw) if raw else None
    except redis.ResponseError as e:
        if "WRONGTYPE" in str(e):
            client.delete(key)
            print(f"⚠️  Deleted WRONGTYPE key '{key}' on read")
        return None
    except Exception:
        return None

# ───────────── LANE THREAD ─────────────
class LaneProcessor(threading.Thread):
    def __init__(self, lane_id, src, state):
        super().__init__(daemon=True)
        self.lane_id   = lane_id
        self.src       = src
        self.state     = state
        self.running   = True
        self.tile      = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
        self.position_tracker       = PositionTracker()
        self.detector               = DualDetector(VEHICLE_MODEL_PATH, AMBULANCE_MODEL_PATH)
        self.frame_count            = 0
        self.fps                    = 0
        self.last_time              = time.time()
        self.last_redis_update      = 0.0
        self.ambulance_alert_active = False
        self.ambulance_alert_start  = 0.0

        self.redis_client = global_redis_client
        if not self.redis_client:
            try:
                self.redis_client = redis.Redis(**REDIS_CONFIG)
                self.redis_client.ping()
                print(f"Lane {lane_id}: Individual Redis connection OK")
            except Exception as e:
                print(f"Lane {lane_id}: Redis unavailable – {e}")
                self.redis_client = None

    # ── Redis update ──────────────────────────────────────────────────────────
    def _ensure_redis(self):
        """Reconnect if the client was never initialised or connection dropped."""
        if self.redis_client:
            try:
                self.redis_client.ping()
                return True
            except Exception:
                print(f"Lane {self.lane_id}: Redis ping failed – attempting reconnect…")

        try:
            self.redis_client = redis.Redis(**REDIS_CONFIG)
            self.redis_client.ping()
            print(f"Lane {self.lane_id}: ✅ Redis reconnected")
            return True
        except Exception as e:
            print(f"Lane {self.lane_id}: ❌ Redis reconnect failed – {e}")
            self.redis_client = None
            return False

    def update_redis(self):
        now = time.time()
        if now - self.last_redis_update < 0.5:   # write every 0.5 s max
            return
        self.last_redis_update = now

        if not self._ensure_redis():
            return

        try:
            snapshot  = self.state.snapshot()
            lane_name = snapshot['lane_name']
            total     = sum(sum(p['vehicles'].values()) for p in snapshot['phases'].values())

            # ── 1. Per-lane key ───────────────────────────────────────────────
            lane_payload = {
                "lane":           lane_name,
                "direction":      LANE_DIRECTIONS[self.lane_id],
                "phases":         snapshot['phases'],
                "ambulance":      snapshot['ambulance'],
                "timestamp":      now,
                "total_vehicles": total,
            }
            ok1 = safe_redis_set(self.redis_client,
                                 f"{REDIS_KEY_LANE_PREFIX}{lane_name}",
                                 lane_payload)

            # ── 2. Intersection key (read-modify-write) ───────────────────────
            data = safe_redis_get(self.redis_client, REDIS_KEY_INTERSECTION) or \
                   {"lanes": {}, "ambulance_alerts": {}, "last_updated": now}

            data["lanes"][lane_name] = {
                "direction":      LANE_DIRECTIONS[self.lane_id],
                "phases":         snapshot['phases'],
                "total_vehicles": total,
            }
            data["last_updated"] = now

            # ── 3. Ambulance ──────────────────────────────────────────────────
            if snapshot['ambulance']['detected']:
                amb_lane = snapshot['ambulance']['lane'] or lane_name
                amb_dir  = snapshot['ambulance']['direction'] or LANE_DIRECTIONS[self.lane_id]
                data.setdefault("ambulance_alerts", {})[lane_name] = {
                    "detected":       True,
                    "count":          snapshot['ambulance']['count'],
                    "timestamp":      now,
                    "lane":           amb_lane,
                    "lane_direction": amb_dir,
                    "message":        f"Emergency vehicle in Lane {amb_lane} heading {amb_dir}",
                }
                self._prioritize_ambulance_lane(data, amb_lane)
                safe_redis_set(self.redis_client, REDIS_KEY_AMBULANCE, {
                    "active":     True,
                    "lane":       amb_lane,
                    "direction":  amb_dir,
                    "timestamp":  now,
                    "all_alerts": data["ambulance_alerts"],
                })
            else:
                data.get("ambulance_alerts", {}).pop(lane_name, None)

            ok2 = safe_redis_set(self.redis_client, REDIS_KEY_INTERSECTION, data)

            # ── Confirm write every 5 s ───────────────────────────────────────
            if int(now) % 5 == 0:
                status = "✅" if (ok1 and ok2) else "⚠️"
                print(f"Lane {lane_name}: {status} Redis updated | total={total} | ambu={snapshot['ambulance']['detected']}")

        except Exception as e:
            import traceback
            print(f"Lane {self.lane_id}: ❌ Redis update_redis() exception:")
            traceback.print_exc()

    def _prioritize_ambulance_lane(self, data, ambulance_lane):
        for lane, lane_data in data.get("lanes", {}).items():
            for phase in lane_data.get("phases", {}):
                lane_data["phases"][phase]["signal_state"] = \
                    "GREEN" if lane == ambulance_lane else "RED"

    # ── Main processing loop ──────────────────────────────────────────────────
    def run(self):
        cap = cv2.VideoCapture(self.src)
        if not cap.isOpened():
            print(f"Error: Could not open {self.src}")
            return

        lane_name = LANE_NAMES[self.lane_id]
        lane_dir  = LANE_DIRECTIONS[self.lane_id]
        print(f"Lane {lane_name} processor started")

        while self.running:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # FPS
            self.frame_count += 1
            if self.frame_count >= 30:
                now = time.time()
                self.fps = self.frame_count / (now - self.last_time)
                self.last_time  = now
                self.frame_count = 0

            h, w = frame.shape[:2]
            ambulance_in_frame = False

            # ── Detection block (YOLO errors are caught but Redis still fires) ─
            try:
                vehicle_res, ambulance_res = self.detector.detect(frame)

                # ── Regular vehicles ──────────────────────────────────────────
                if (vehicle_res.boxes is not None and
                        vehicle_res.boxes.id is not None):
                    active_ids = set()
                    for box, tid, cls_id in zip(
                            vehicle_res.boxes.xyxy.cpu().numpy(),
                            vehicle_res.boxes.id.cpu().numpy().astype(int),
                            vehicle_res.boxes.cls.cpu().numpy().astype(int)):

                        active_ids.add(tid)
                        cx = int((box[0] + box[2]) / 2)
                        cy = int((box[1] + box[3]) / 2)
                        position = self.position_tracker.update(tid, cx, cy, w, 0)
                        phase    = PHASE_MAPPING[position]
                        vtype    = self.detector.get_vehicle_type(cls_id)

                        if self.position_tracker.is_new_vehicle(tid):
                            self.state.increment_count(phase, vtype)
                            print(f"Lane {lane_name}: New {vtype} → {phase}")

                        color = {"left":(255,0,0),"straight":(0,255,0),"right":(0,0,255)}[phase]
                        cv2.rectangle(frame,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),color,2)
                        cv2.putText(frame, f"ID:{tid}-{vtype[:3]}-{phase[:1]}",
                                   (int(box[0]),int(box[1]-5)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                    self.position_tracker.cleanup_old_tracks(active_ids)

                # ── Ambulances ────────────────────────────────────────────────
                if (ambulance_res.boxes is not None and
                        ambulance_res.boxes.id is not None):
                    for box, tid in zip(
                            ambulance_res.boxes.xyxy.cpu().numpy(),
                            ambulance_res.boxes.id.cpu().numpy().astype(int)):

                        cx  = int((box[0]+box[2])/2)
                        cy  = int((box[1]+box[3])/2)
                        pos = self.position_tracker.update(f"amb_{tid}", cx, cy, w, 0)
                        phase = PHASE_MAPPING[pos]

                        if self.position_tracker.is_new_ambulance(f"amb_{tid}"):
                            self.state.set_ambulance_detected(True, pos)
                            ambulance_in_frame = True
                            print("\n" + "="*70)
                            print(f"🚨 AMBULANCE DETECTED  Lane:{lane_name}  Dir:{lane_dir}  Turn:{phase.upper()}")
                            print("="*70+"\n")
                            self.ambulance_alert_active = True
                            self.ambulance_alert_start  = time.time()

                        cv2.rectangle(frame,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(255,0,255),3)
                        if int(time.time()*2)%2==0:
                            cv2.putText(frame, f"AMBULANCE Lane {lane_name}",
                                       (int(box[0]),int(box[1]-10)),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

                # Expire alert after 5 s of no detection
                if not ambulance_in_frame and self.ambulance_alert_active:
                    if time.time() - self.ambulance_alert_start > 5:
                        self.ambulance_alert_active = False
                        self.state.set_ambulance_detected(False)

            except Exception as e:
                import traceback
                print(f"❌ Detection error lane {self.lane_id}: {e}")
                traceback.print_exc()

            # ── Redis update is OUTSIDE the detection try/except ─────────────
            # Fires every 0.5 s regardless of whether YOLO succeeds or fails
            self.update_redis()

            # HUD overlay
            cv2.putText(frame, f"Lane {lane_name} - FPS: {self.fps:.1f}",
                       (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            if self.ambulance_alert_active:
                ay = 80
                if int(time.time()*2)%2==0:
                    cv2.rectangle(frame,(0,ay-25),(frame.shape[1],ay+35),(0,0,255),-1)
                cv2.putText(frame, f"EMERGENCY VEHICLE IN LANE {lane_name}",
                           (50,ay), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(frame, f"Direction: {lane_dir} | Please clear the way!",
                           (50,ay+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            self.tile = annotate(frame.copy(), self.state, lane_dir)

        cap.release()
        print(f"Lane {lane_name} processor stopped")

    def get_tile(self):
        return self.tile

    def stop(self):
        self.running = False

# ───────────── MAIN ─────────────
def main():
    print("="*70)
    print("TRAFFIC INTERSECTION MONITORING SYSTEM WITH AMBULANCE DETECTION")
    print("="*70)
    print(f"Redis keys used:")
    print(f"  {REDIS_KEY_INTERSECTION}   → full intersection JSON")
    print(f"  {REDIS_KEY_LANE_PREFIX}<A|B|C|D>  → per-lane JSON")
    print(f"  {REDIS_KEY_AMBULANCE}      → active ambulance alert")
    print("="*70)

    states  = [LaneState(i+1) for i in range(4)]
    threads = []

    for i in range(4):
        t = LaneProcessor(i+1, VIDEO_SOURCES[i], states[i])
        threads.append(t)
        t.start()
        time.sleep(0.5)

    cv2.namedWindow("Traffic Intersection Monitor", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Traffic Intersection Monitor", 1280, 720)

    print("System running. Press ESC to exit.")

    while True:
        try:
            tiles  = [t.get_tile() for t in threads]
            top    = np.hstack([tiles[0], tiles[1]])
            bottom = np.hstack([tiles[2], tiles[3]])
            grid   = np.vstack([top, bottom])
            cv2.imshow("Traffic Intersection Monitor", grid)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Display error: {e}")

    print("\nShutting down...")
    for t in threads:
        t.stop()
    for t in threads:
        t.join(timeout=2.0)
    cv2.destroyAllWindows()

    print("\nFINAL COUNTS:")
    total_ambs = 0
    for i, state in enumerate(states):
        snap = state.snapshot()
        lane = LANE_NAMES[i+1]
        print(f"\nLane {lane}:")
        amb = state.get_ambulance_status()
        if amb['count'] > 0:
            print(f"  🚑 Ambulances: {amb['count']}")
            total_ambs += amb['count']
        for ph, pd in snap['phases'].items():
            t = sum(pd['vehicles'].values())
            if t:
                print(f"  {ph}: {t} vehicles → {pd['vehicles']}")
    print(f"\nTotal ambulances: {total_ambs}")

if __name__ == "__main__":
    main()
