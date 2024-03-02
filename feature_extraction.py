import numpy as np

def calculate_distance(point1, point2):
    point1 = (point1.x, point1.y)
    point2 = (point2.x, point2.y)
    return np.linalg.norm(np.array(point1) - np.array(point2))


def calculate_angle(point1, point2, point3):
    """Розрахунок кута між трьома точками, p2 - вершина кута."""
    point1 = (point1.x, point1.y)
    point2 = (point2.x, point2.y)
    point3 = (point3.x, point3.y)
    p1 = np.array(point1)
    p2 = np.array(point2)
    p3 = np.array(point3)
    ba = p1 - p2
    bc = p3 - p2

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)



def find_features(landmarks):
    features = []
    # Ознака 1: Відстань між куточками рота
    distance_mouth_corners = calculate_distance(landmarks.part(48), landmarks.part(54))

    # Ознака 2: Відстань між бровами
    distance_eyebrows = calculate_distance(landmarks.part(21), landmarks.part(22))

    # Ознака 3: Відстань між верхньою губою і носом
    distance_nose_to_upper_lip = calculate_distance(landmarks.part(33), landmarks.part(51))

    # Ознака 4: Кут нахилу брів (беремо три точки для брови, щоб визначити кут)
    angle_left_eyebrow = calculate_angle(landmarks.part(17), landmarks.part(19), landmarks.part(21))
    angle_right_eyebrow = calculate_angle(landmarks.part(22), landmarks.part(24), landmarks.part(26))

    # Ознака 5: Відкриття очей (відстань між верхнім і нижнім віком)
    left_eye_opening = calculate_distance(landmarks.part(37), landmarks.part(41))
    right_eye_opening = calculate_distance(landmarks.part(44), landmarks.part(46))

    # Ознака 6: Ширина носа (відстань між крилами носа)
    nose_width = calculate_distance(landmarks.part(31), landmarks.part(35))

    # Ознака 7: Відстань між бровою і верхньою частиною віка для кожного ока
    distance_eyebrow_upper_eyelid_left = calculate_distance(landmarks.part(20), landmarks.part(37))
    distance_eyebrow_upper_eyelid_right = calculate_distance(landmarks.part(23), landmarks.part(44))

    # Ознака 8: Висота рота (відстань між верхньою та нижньою губами)
    mouth_height = calculate_distance(landmarks.part(51), landmarks.part(57))

    # Ознака 9: Відстані між кінчиком носа і кожним куточком рота
    nose_tip_to_mouth_corner_left = calculate_distance(landmarks.part(33), landmarks.part(48))
    nose_tip_to_mouth_corner_right = calculate_distance(landmarks.part(33), landmarks.part(54))

    # Ознака 10: Відстань між центром очей і куточками рота
    eyes_center_to_mouth_left = calculate_distance(landmarks.part(40), landmarks.part(48))
    eyes_center_to_mouth_right = calculate_distance(landmarks.part(47), landmarks.part(54))



    return np.array([
        distance_mouth_corners, distance_eyebrows, distance_nose_to_upper_lip, angle_left_eyebrow, angle_right_eyebrow,
         left_eye_opening, right_eye_opening,
         nose_width, distance_eyebrow_upper_eyelid_left, distance_eyebrow_upper_eyelid_right, mouth_height,
         nose_tip_to_mouth_corner_left,
         nose_tip_to_mouth_corner_right, eyes_center_to_mouth_left, eyes_center_to_mouth_right])

