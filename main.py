import cv2
import numpy as np
import face_recognition
import os
import time
from datetime import datetime

# ==========================
# SETTINGS
# ==========================

ATTENDANCE_COOLDOWN = 30
UNKNOWN_COOLDOWN = 5

# ==========================
# LOAD TRAINING IMAGES
# ==========================

path = "images"

images = []
classNames = []

for file in os.listdir(path):

    img = cv2.imread(os.path.join(path, file))

    if img is not None:

        images.append(img)

        classNames.append(
            os.path.splitext(file)[0]
        )

print("Loaded:", classNames)

# ==========================
# CREATE ENCODINGS
# ==========================

def findEncodings(images):

    encodeList = []

    for img in images:

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        encodings = face_recognition.face_encodings(img)

        if len(encodings) > 0:

            encodeList.append(
                encodings[0]
            )

    return encodeList

# ==========================
# ATTENDANCE
# ==========================

lastAttendance = {}

def markAttendance(name):

    currentTime = time.time()

    if (
        name in lastAttendance
        and currentTime - lastAttendance[name]
        < ATTENDANCE_COOLDOWN
    ):
        return

    lastAttendance[name] = currentTime

    fileName = "attendance.csv"

    if not os.path.exists(fileName):

        with open(fileName, "w") as f:

            f.write("Name,Time\n")

    now = datetime.now()

    dtString = now.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    with open(fileName, "a") as f:

        f.write(
            f"{name},{dtString}\n"
        )

    print(
        f"Attendance marked for {name}"
    )

# ==========================
# UNKNOWN FACE STORAGE
# ==========================

def saveUnknown(img):

    folder = "unknown_faces"

    if not os.path.exists(folder):

        os.makedirs(folder)

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    filename = os.path.join(
        folder,
        f"unknown_{timestamp}.jpg"
    )

    cv2.imwrite(filename, img)

    print("Unknown face saved")

# ==========================
# ADD NEW PERSON
# ==========================

def addNewPerson(frame):

    name = input("\nEnter new person's name: ").strip()

    if name == "":
        print("Invalid name")
        return

    filename = os.path.join(
        path,
        f"{name}.jpg"
    )

    cv2.imwrite(filename, frame)

    print(
        f"{name} saved successfully."
    )

    print(
        "Restart the program to recognize the new person."
    )

# ==========================
# ENCODE KNOWN FACES
# ==========================

print("\nEncoding Faces...")

encodeListKnown = findEncodings(images)

print("Encoding Complete")

# ==========================
# CAMERA
# ==========================

cap = cv2.VideoCapture(0)

lastUnknownTime = 0

recognizedPeople = set()

unknownCount = 0

while True:

    success, img = cap.read()

    if not success:
        break

    imgSmall = cv2.resize(
        img,
        (0, 0),
        None,
        0.25,
        0.25
    )

    imgSmall = cv2.cvtColor(
        imgSmall,
        cv2.COLOR_BGR2RGB
    )

    facesCurFrame = face_recognition.face_locations(
        imgSmall
    )

    encodesCurFrame = face_recognition.face_encodings(
        imgSmall,
        facesCurFrame
    )

    # ======================
    # FACE RECOGNITION
    # ======================

    for encodeFace, faceLoc in zip(
        encodesCurFrame,
        facesCurFrame
    ):

        matches = face_recognition.compare_faces(
            encodeListKnown,
            encodeFace,
            tolerance=0.55
        )

        faceDis = face_recognition.face_distance(
            encodeListKnown,
            encodeFace
        )

        matchIndex = np.argmin(faceDis)

        y1, x2, y2, x1 = faceLoc

        y1 *= 4
        x2 *= 4
        y2 *= 4
        x1 *= 4

        if (
            len(faceDis) > 0
            and matches[matchIndex]
        ):

            name = (
                "✓ "
                + classNames[
                    matchIndex
                ].upper()
            )

            color = (0, 255, 0)

            recognizedPeople.add(
                classNames[matchIndex]
            )

            markAttendance(
                classNames[matchIndex]
                .upper()
            )

        else:

            name = "⚠ UNKNOWN"

            color = (0, 0, 255)

            currentTime = time.time()

            if (
                currentTime
                - lastUnknownTime
                > UNKNOWN_COOLDOWN
            ):

                saveUnknown(img)

                lastUnknownTime = currentTime

                unknownCount += 1

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.rectangle(
            img,
            (x1, y2 - 40),
            (x2, y2),
            color,
            cv2.FILLED
        )

        cv2.putText(
            img,
            name,
            (x1 + 6, y2 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    # ======================
    # HEADER BAR
    # ======================

    cv2.rectangle(
        img,
        (0, 0),
        (img.shape[1], 90),
        (40, 40, 40),
        -1
    )

    currentClock = datetime.now().strftime(
        "%H:%M:%S"
    )

    cv2.putText(
        img,
        "FACE RECOGNITION ATTENDANCE SYSTEM",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        img,
        f"Time: {currentClock}",
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    cv2.putText(
        img,
        f"Recognized: {len(recognizedPeople)}",
        (350, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    cv2.putText(
        img,
        f"Unknown: {unknownCount}",
        (600, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

    # ======================
    # FOOTER
    # ======================

    cv2.putText(
        img,
        "[A] Add Person",
        (15, img.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        img,
        "[S] Screenshot",
        (220, img.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        img,
        "[Q] Quit",
        (450, img.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Face Recognition Attendance System",
        img
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("s"):

        screenshotName = (
            "screenshot_"
            + datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
            + ".jpg"
        )

        cv2.imwrite(
            screenshotName,
            img
        )

        print(
            f"Saved {screenshotName}"
        )

    elif key == ord("a"):

        addNewPerson(img)

cap.release()

cv2.destroyAllWindows()