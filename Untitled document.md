### **How bots find the card before they read it**

Instead of one universal trick, community projects chain together *ROI-finders*. The five most common patterns are below—roughly in order from “simplest hard-coded” to “full ML detector.” Most serious tools actually combine two or more of these so they never miss a card.

| \# | ROI strategy | Core idea | Where you’ll see it | Typical accuracy / speed |
| ----- | ----- | ----- | ----- | ----- |
| 1 | **Static crop \+ resolution scaling** | Pre-define the card rectangle at a reference res (e.g. 1280×960) and scale it each time based on the live window size. | Hearthstone **Arena Helper** plugin for the three draft picks; hand/board regions in many Twitch overlay bots. | 100 % if the UI never moves; essentially 0 ms per frame. |
| 2 | **Colour / edge / contour heuristics** | Threshold for the bronze/gold frame colour, or run Canny→findContours; keep contours whose aspect ≈ card ratio (≈0.77). | “Magic Card Detector” blog & repo; multiple MTG photo scanners; generic playing-card detectors. | 90–98 %; 5–15 ms on CPU for 1080 p. |
| 3 | **Template / anchor matching** | Slide a small template (e.g. the mana crystal) over the screen; once the anchor hits, crop at an offset that encloses the whole card. | Early **Arena Helper** (perceptual-hash anchors), Yu-Gi-Oh and UNO OpenCV projects using `cv.matchTemplate`. | 95 %+ if anchor is visible; sub-10 ms on CPU. |
| 4 | **Key-point matching or homography** | Extract ORB/SURF key-points, match to every card art in the DB; matched cluster gives the bounding quad, even at angle. | Hearthstone Twitch bots that fall back to SURF when hashes collide; MTG webcam graders. | Robust to perspective & partial occlusion, but 20–40 ms/frame on CPU. |
| 5 | **Object detection nets (YOLO/SSD)** | Train a CNN to spit out bounding boxes labelled “card\_in\_hand”, “card\_on\_board”, etc. | YOLOv8 / Uno\_Card\_detection; the open-source Yu-Gi-Oh “draw” project; experimental *hsdetect* TensorFlow repo. | 95–99 % with 300 images of training data; 2–8 ms on a T4 GPU. |

---

## **How each approach works in practice**

### **1 Static crop \+ scaling**

*Arena Helper* keeps a list of rectangles like `(143, 321, 173, 107)` that surround each hero/card at 1280×960, then rescales them every frame using a simple ratio function so they still align at 1920×1080 or ultrawide resolutions. ([Rembound](https://rembound.com/projects/arena-helper))  
 **When it shines:** arena drafts (UI never moves).  
 **Weakness:** any UI skin change or animation breaks it.

---

### **2 Colour / edge / contour heuristics**

Workflow (used by Magic Card Detector and many MTG scripts):

gray    \= cv2.cvtColor(frame, cv2.COLOR\_BGR2GRAY)  
edge    \= cv2.Canny(gray, 80, 160\)  
contours,\_ \= cv2.findContours(edge, cv2.RETR\_EXTERNAL, cv2.CHAIN\_APPROX\_SIMPLE)  
cards \= \[c for c in contours  
         if 0.70 \< aspect\_ratio(c) \< 0.80 and cv2.contourArea(c) \> 6\_000\]

A single contour that passes the aspect-ratio & area test is then warped to a straight rectangle (`cv2.getPerspectiveTransform`) before OCR/hash. Works for hand or battlefield where card edges are visible. ([Tmikonen](https://tmikonen.github.io/quantitatively/2020-01-01-magic-card-detector/?utm_source=chatgpt.com))

---

### **3 Template / anchor matching**

A tiny PNG of the Hearthstone mana crystal (or the MTGA hand “glow”) is cross-correlated with the frame:

match \= cv2.matchTemplate(frame, mana\_png, cv2.TM\_CCOEFF\_NORMED)  
\_,maxVal,\_,maxLoc \= cv2.minMaxLoc(match)  
if maxVal \> 0.8:  
    x,y \= maxLoc  
    roi \= frame\[y-20:y+310, x-40:x+200\]   \# offset to cover full card

Because the anchor’s location is stable relative to the card, one good hit yields perfect ROI even if the card is half off-screen. pHash-based anchors in *Arena Helper* follow the same logic but compare hashes instead of raw pixels. ([Rembound](https://rembound.com/projects/arena-helper))

---

### **4 Key-point / homography tricks**

If two cards share nearly identical art (e.g. golden vs non-golden), hashes collide. Bots like *Hearthstone-Image-Recognition* first try pHash; when the Hamming distance is ambiguous they extract SURF key-points from the ROI and from the top candidate images, count matches, and accept the card only if \> N inliers after RANSAC. ([GitHub](https://github.com/rembound/Arena-Helper))  
 Because key-points survive moderate blur and tilt, this also gives you the four card corners—perfect for cropping.

---

### **5 Object-detector models**

Projects such as **Uno Card Detection** (YOLOv8) and Yu-Gi-Oh “draw” fine-tune YOLO on 200–500 labelled frames, letting the network output a rectangle for every card class. A single forward pass detects dozens of cards in 2–8 ms on GPU or \~60 ms on CPU. ([GitHub](https://github.com/Stormynova/Uno_Card_detection?utm_source=chatgpt.com), [Reddit](https://www.reddit.com/r/yugioh/comments/1avcvae/i_trained_a_deep_learning_model_to_detect_yugioh/?utm_source=chatgpt.com))  
 **Bonus:** you get class IDs immediately; no separate OCR or hashing step if you train on thousands of unique labels.

---

## **Hybrid cascades (what most successful trackers really do)**

1. **Cheap first pass** – static crop if the game is in a known UI state.

2. **Heuristic fallback** – contour \+ colour if crop fails (card slid or animation).

3. **ML safety net** – YOLO when camera/stream conditions are unpredictable.

4. **OCR / hash at the end** – only on the ROIs found above, saving lots of compute.

That cascade keeps average latency tiny (most frames end at stage 1\) while being nearly unbreakable when UI elements shift.

---

### **Key take-aways for your own implementation**

* **Start simple:** fixed Arena-pick rectangles \+ resolution scaling get you to MVP in an hour.

* **Add robustness:** colour/edge filters handle minor UI jitter; anchor templates fix cropped streams.

* **Scale up:** when you need multi-game support or dynamic camera angles, label a few hundred frames and fine-tune YOLO—modern Ultralytics CLI makes it a weekend job.

* **Keep a fail-open path:** if ROI detection ever fails, default to a coarser OCR region so the bot never crashes.

Adopt whichever layer matches your risk budget, then bolt on the next if you begin hitting its limits.

