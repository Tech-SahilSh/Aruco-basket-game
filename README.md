# 🎮 AR Basket Game (OpenCV + ArUco)

## 📖 Description
AR Basket Game is a real-time computer vision game built using Python and OpenCV.  
It uses an ArUco marker as a virtual basket, where players catch falling coins using hand movement through a webcam.

---

## 🎥 Demo

### ▶️ Watch Full Demo
[▶️ Click to Watch Video](https://github.com/Tech-SahilSh/Aruco-basket-game/edit/main/outputvideo.mp4)

---

## ⚙️ Components / Modules

- **Camera Module** → Captures real-time video
- **ArUco Detection** → Tracks marker position as basket
- **Game Logic** → Coin spawning, movement, scoring
- **UI System** → Start/Restart buttons using mouse events
- **Rendering Engine** → Draws coins, basket (3D), HUD

---

## 🛠️ Technologies Used

- Python  
- OpenCV (`cv2`)  
- NumPy  
- ArUco Marker Detection  
- Time & Random Libraries  

---

## 🚀 How It Works

1. Start webcam → system waits for ArUco marker  
2. Marker detected → "Start Game" button appears  
3. Coins start falling from top  
4. Move marker to catch coins  
5. Catch → Score +1  
6. Miss → Life -1  
7. Lives = 0 → Game Over  
8. Restart option available  

---

## 📚 Key Learnings

- Real-time object detection using OpenCV  
- ArUco marker tracking & pose estimation  
- 2D–3D projection concepts  
- Game loop & state management  
- Interactive UI with mouse events  

---

## ▶️ How to Run

### 1. Install Dependencies
```bash
pip install opencv-python numpy
python main.py
