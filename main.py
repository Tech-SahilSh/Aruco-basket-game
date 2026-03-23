import cv2
import numpy as np
import time, random, sys
global last_spawn
# =================== BUTTON HANDLING ===================
button_clicked = None  # "start", "restart", ya None

def mouse_callback(event, x, y, flags, param):
    global button_clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        bx, by, bw, bh, btype = param  # button rectangle
        if bx <= x <= bx+bw and by <= y <= by+bh:
            button_clicked = btype

def draw_button(frame, text, x, y, w, h):
    cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 200), -1)
    cv2.putText(frame, text, (x+15, y+int(h*0.65)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    return (x, y, w, h)
# =======================================================

# =================== USER SETTINGS ===================
MARKER_DICT_NAME = "DICT_ARUCO_ORIGINAL"
TARGET_MARKER_ID = 1
MARKER_LENGTH_M  = 0.05
COIN_SPAWN_INTERVAL = 0.8
COIN_SPEED_RANGE    = (220, 320)
COIN_RADIUS         = 12
LIVES_START         = 3
# =====================================================

# --- OpenCV ArUco dictionary & detector (new + old API support) ---
aruco = cv2.aruco
DICT = getattr(aruco, MARKER_DICT_NAME)
dictionary = aruco.getPredefinedDictionary(DICT)

USE_NEW_DETECTOR = True
try:
    parameters = aruco.DetectorParameters()
    detector   = aruco.ArucoDetector(dictionary, parameters)
except AttributeError:
    USE_NEW_DETECTOR = False
    parameters = aruco.DetectorParameters_create()

HAS_EPSM = hasattr(aruco, "estimatePoseSingleMarkers")

# --- Video capture ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not found.")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

ok, frame = cap.read()
if not ok:
    print("Error: Can't read from camera.")
    sys.exit(1)
H, W = frame.shape[:2]

# --- Camera intrinsics guess ---
f = 0.9 * W
camera_matrix = np.array([[f, 0, W/2],
                          [0, f, H/2],
                          [0, 0,   1 ]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)

# --- 3D basket (box) ---
s = MARKER_LENGTH_M
basket_h = s * 0.6
box_3d = np.float32([
    [-s/2, -s/2,  0],
    [ s/2, -s/2,  0],
    [ s/2,  s/2,  0],
    [-s/2,  s/2,  0],
    [-s/2, -s/2, -basket_h],
    [ s/2, -s/2, -basket_h],
    [ s/2,  s/2, -basket_h],
    [-s/2,  s/2, -basket_h],
])

def project(rvec, tvec, pts3d):
    pts2d, _ = cv2.projectPoints(pts3d, rvec, tvec, camera_matrix, dist_coeffs)
    return pts2d.reshape(-1, 2)

def draw_box(img, pts):
    pts = pts.astype(int)
    cv2.polylines(img, [pts[0:4]], True, (0,255,0), 2)
    cv2.polylines(img, [pts[4:8]], True, (0,255,0), 2)
    for i in range(4):
        cv2.line(img, pts[i], pts[i+4], (0,255,0), 2)

# --- Game variables ---
coins = []
score = 0
lives = LIVES_START
last_spawn = time.time()
prev_t = time.time()

def spawn_coin():
    x = random.randint(int(0.08*W), int(0.92*W))
    coins.append({
        "x": x,
        "y": -20,
        "r": COIN_RADIUS,
        "vy": random.uniform(*COIN_SPEED_RANGE),
        "caught": False
    })

objp_marker_corners = np.float32([
    [-s/2,  s/2, 0],
    [ s/2,  s/2, 0],
    [ s/2, -s/2, 0],
    [-s/2, -s/2, 0],
])

# --- State Machine ---
game_state = "WAITING"
cv2.namedWindow("AR Basket Game")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    now = time.time()
    dt = max(1e-3, now - prev_t)
    prev_t = now

    # Detect marker
    if USE_NEW_DETECTOR:
        corners, ids, _ = detector.detectMarkers(frame)
    else:
        corners, ids, _ = aruco.detectMarkers(frame, dictionary, parameters=parameters)

    marker_detected = ids is not None and TARGET_MARKER_ID in ids.flatten() if ids is not None else False
    rvec, tvec, mouth_poly = None, None, None

    if game_state == "WAITING":
        cv2.putText(frame, "Show ArUco Marker to start", (50,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        if marker_detected:
            bx, by, bw, bh = 200, 200, 220, 60
            rect = draw_button(frame, "START GAME", bx, by, bw, bh)
            cv2.setMouseCallback("AR Basket Game", mouse_callback, rect+("start",))
            if button_clicked == "start":
                button_clicked = None
                score, lives, coins = 0, LIVES_START, []
                game_state = "PLAYING"

    elif game_state == "PLAYING":
        # Marker pose
        if ids is not None and TARGET_MARKER_ID in ids.flatten():
            idx = list(ids.flatten()).index(TARGET_MARKER_ID)
            corner = corners[idx].reshape(4, 2)
            if HAS_EPSM:
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                    [corner], MARKER_LENGTH_M, camera_matrix, dist_coeffs)
                rvec, tvec = rvecs[0], tvecs[0]
            else:
                ok_pnp, rvec, tvec = cv2.solvePnP(
                    objp_marker_corners, corner, camera_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if not ok_pnp: rvec = tvec = None

            aruco.drawDetectedMarkers(frame, [corner.reshape(1,4,2)], ids[idx:idx+1])
            if rvec is not None:
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, MARKER_LENGTH_M*0.5)
                box2d = project(rvec, tvec, box_3d)
                draw_box(frame, box2d)
                mouth_poly = box2d[4:8].astype(np.int32)

        # Coin spawn
        
        if (now - last_spawn) >= COIN_SPAWN_INTERVAL and lives > 0:
            spawn_coin()
            last_spawn = now

        # Coin update
        to_remove = []
        for i, coin in enumerate(coins):
            coin["y"] += coin["vy"] * dt
            center = (int(coin["x"]), int(coin["y"]))
            cv2.circle(frame, center, coin["r"], (0,215,255), -1)
            if mouth_poly is not None and cv2.pointPolygonTest(mouth_poly, center, False) >= 0:
                to_remove.append(i)
        for idx in sorted(to_remove, reverse=True):
            coins.pop(idx)
            score += 1

        # Miss check
        to_remove = []
        for i, coin in enumerate(coins):
            if coin["y"] - coin["r"] > H:
                to_remove.append(i)
                lives -= 1
        for idx in sorted(to_remove, reverse=True):
            coins.pop(idx)

        # HUD
        cv2.rectangle(frame, (0,0), (W, 40), (0,0,0), -1)
        cv2.putText(frame, f"Score: {score}   Lives: {max(lives,0)}",
                    (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        if lives <= 0:
            game_state = "GAME_OVER"

    elif game_state == "GAME_OVER":
        overlay = frame.copy()
        cv2.rectangle(overlay, (int(W*0.1), int(H*0.35)),
                      (int(W*0.9), int(H*0.65)), (0,0,0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        cv2.putText(frame, f"GAME OVER | Final Score: {score}",
                    (int(W*0.12), int(H*0.5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        bx, by, bw, bh = 200, 200, 220, 60
        rect = draw_button(frame, "PLAY AGAIN", bx, by, bw, bh)
        cv2.setMouseCallback("AR Basket Game", mouse_callback, rect+("restart",))
        if button_clicked == "restart":
            button_clicked = None
            score, lives, coins = 0, LIVES_START, []
            game_state = "PLAYING"

    cv2.imshow("AR Basket Game", frame)
    if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
        break

cap.release()
cv2.destroyAllWindows()
