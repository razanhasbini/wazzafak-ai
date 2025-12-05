import psycopg2
import mediapipe as mp
import cv2
import json
import math

# Database connection details
db_conn = psycopg2.connect(
    dbname="your_dbname",  # replace with your database name
    user="your_dbuser",    # replace with your database user
    password="your_dbpassword",  # replace with your password
    host="localhost"
)
cursor = db_conn.cursor()

# Initialize MediaPipe Face Mesh and Hand modules
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Function to insert the stress detection result into the database
def insert_stress_result(user_id, session_id, stress_level, facial_landmarks, hand_landmarks):
    insert_query = """
    INSERT INTO ai.stress_detection_results (user_id, session_id, stress_level, facial_landmarks, hand_landmarks)
    VALUES (%s, %s, %s, %s, %s)
    """
    cursor.execute(insert_query, (user_id, session_id, stress_level, facial_landmarks, hand_landmarks))
    db_conn.commit()

# Function to calculate distance between two points
def calculate_distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

# Function to analyze facial landmarks for stress detection
def analyze_face(landmarks):
    left_eye = landmarks[33]  # Left eye landmark (example)
    right_eye = landmarks[263]  # Right eye landmark
    mouth = landmarks[13]  # Mouth landmark

    # Calculate distance between eyebrow and mouth (simplified)
    distance = calculate_distance(left_eye, right_eye)

    # Logic: Raised eyebrows = stress/anxiety
    if distance > 0.04:  # Adjust this threshold as needed
        return "stressed"
    
    # Check if the mouth is smiling (example logic)
    if calculate_distance(mouth, landmarks[14]) < 0.02:  # A smile is when the corners of the mouth are close
        return "confident"
    
    return "neutral"

# Function to analyze hand landmarks for confidence or stress
def analyze_hands(hand_landmarks):
    # Example: Detecting hand positions
    wrist = hand_landmarks[0]  # Wrist landmark
    thumb = hand_landmarks[4]  # Thumb tip
    index_finger = hand_landmarks[8]  # Index finger tip

    # Check if hands are near the face (simplified example)
    if wrist.x < 0.5 and wrist.y < 0.5:
        return "stressed"

    # Check for hand wringing (hands close to each other)
    if calculate_distance(thumb, index_finger) < 0.03:
        return "anxious"

    return "confident"

# Start webcam capture
cap = cv2.VideoCapture(0)

# Session ID and user ID (you should get these dynamically, e.g., from your app)
user_id = 1  # Replace with actual user ID
session_id = "session123"  # Replace with actual session ID

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB (MediaPipe works with RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe for face landmarks and hand landmarks
    face_results = face_mesh.process(rgb_frame)
    hand_results = hands.process(rgb_frame)

    # Initialize stress level, facial landmarks, and hand landmarks variables
    stress_level = "neutral"
    facial_landmarks = None
    hand_landmarks = None

    # Analyze facial landmarks for stress detection
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            stress_level = analyze_face(face_landmarks.landmark)
            facial_landmarks = json.dumps([{"x": landmark.x, "y": landmark.y} for landmark in face_landmarks.landmark])

    # Analyze hand landmarks for confidence or stress
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            stress_level = analyze_hands(hand_landmarks.landmark)
            hand_landmarks = json.dumps([{"x": landmark.x, "y": landmark.y} for landmark in hand_landmarks.landmark])

    # Insert the stress/anxiety result into the database
    insert_stress_result(user_id, session_id, stress_level, facial_landmarks, hand_landmarks)

    # Display the stress level on the frame (without showing landmarks)
    cv2.putText(frame, f"Stress Level: {stress_level}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Show the frame with the results (no landmarks overlay)
    cv2.imshow("Webcam Feed", frame)

    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Close the database connection
cursor.close()
db_conn.close()
