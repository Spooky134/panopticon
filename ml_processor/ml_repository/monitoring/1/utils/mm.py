import cv2
import mediapipe as mp


class RecognitionModel:
    def __init__(self, model_complexity=1, min_detection_confidence=0.5):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )

        # Индексы точек ног и запястий для исключения
        self.excluded_landmarks = {17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 31, 32}

    def process_frame(self, frame):
        # Конвертация BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Инференс Holistic (поза + руки + лицо)
        results = self.holistic.process(frame_rgb)

        # Отрисовка позы (только верхняя часть тела)
        if results.pose_landmarks:
            self._draw_upper_body_only(frame, results.pose_landmarks)

        # Отрисовка рук (21 точка на каждую)
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(255, 0, 0)),
                # self.mp_drawing.DrawingSpec(color=(255, 0, 0))
            )

        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(255, 0, 0)),
                # self.mp_drawing.DrawingSpec(color=(255, 0, 0))
            )


        return frame

    def _draw_upper_body_only(self, image, landmarks):
        h, w, _ = image.shape

        for idx, landmark in enumerate(landmarks.landmark):
            if idx in self.excluded_landmarks:
                continue

            cx, cy = int(landmark.x * w), int(landmark.y * h)

            cv2.circle(image, (cx, cy), 4, (255, 0, 0), -1)


        for connection in self.mp_holistic.POSE_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]

            if start_idx not in self.excluded_landmarks and end_idx not in self.excluded_landmarks:
                start_pt = landmarks.landmark[start_idx]
                end_pt = landmarks.landmark[end_idx]

                start_px = (int(start_pt.x * w), int(start_pt.y * h))
                end_px = (int(end_pt.x * w), int(end_pt.y * h))

                cv2.line(image, start_px, end_px, (255, 255, 255), 2)