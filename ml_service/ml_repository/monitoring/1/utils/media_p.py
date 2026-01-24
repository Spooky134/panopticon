import cv2
import mediapipe as mp
import numpy as np
import math


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


        self.excluded_landmarks = {17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 31, 32}


        self._data = []


        self.motion_profile = {}
        self.angle_profile = {}


    def _calculate_angle(self, a, b, c):
        """Считает угол между тремя точками (b - центральная)"""
        a = np.array(a)  # Первая точка
        b = np.array(b)  # Центральная точка
        c = np.array(c)  # Конечная точка

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle
        return angle

    def _calculate_inclination(self, a, b):
        """Считает наклон линии между двумя точками относительно горизонта"""
        radians = np.arctan2(b[1] - a[1], b[0] - a[0])
        angle = np.degrees(radians)
        return angle

    def _get_metrics(self, landmarks):
        """
        Извлекает ключевые геометрические параметры из кадра:
        - Координаты (для старой логики)
        - Углы (для новой точной логики)
        """
        metrics = {'coords': {}, 'angles': {}}

        # 1. Сбор координат (старая логика)
        for idx, landmark in enumerate(landmarks.landmark):
            if idx in self.excluded_landmarks:
                continue
            metrics['coords'][idx] = {'x': landmark.x, 'y': landmark.y}


        # 11-12: Плечи, 23-24: Бедра
        # 13-14: Локти, 15-16: Запястья, 7-8: Уши
        l_sh = landmarks.landmark[11]
        r_sh = landmarks.landmark[12]
        l_ear = landmarks.landmark[7]
        r_ear = landmarks.landmark[8]

        # Руки (левая)
        l_el = landmarks.landmark[13]
        l_wr = landmarks.landmark[15]
        # Руки (правая)
        r_el = landmarks.landmark[14]
        r_wr = landmarks.landmark[16]

        # 2. Вычисление углов (Новая логика)

        # А) Наклон корпуса (плечевая ось)
        # Если человек наклоняется влево/вправо, угол меняется
        metrics['angles']['shoulder_tilt'] = self._calculate_inclination([l_sh.x, l_sh.y], [r_sh.x, r_sh.y])

        # Б) Наклон головы
        # Угол линии ушей. Сравниваем его с углом плеч, чтобы понять поворот головы относительно тела
        head_tilt_abs = self._calculate_inclination([l_ear.x, l_ear.y], [r_ear.x, r_ear.y])
        metrics['angles']['head_relative'] = head_tilt_abs - metrics['angles']['shoulder_tilt']

        # В) Сгиб локтей (важно для упражнений рук)
        metrics['angles']['left_elbow'] = self._calculate_angle([l_sh.x, l_sh.y], [l_el.x, l_el.y], [l_wr.x, l_wr.y])
        metrics['angles']['right_elbow'] = self._calculate_angle([r_sh.x, r_sh.y], [r_el.x, r_el.y], [r_wr.x, r_wr.y])

        return metrics

    # --- 1. Сбор статистики ---
    def get_stat(self, results):
        if not results.pose_landmarks:
            return

        # Извлекаем и координаты, и углы
        current_metrics = self._get_metrics(results.pose_landmarks)

        if current_metrics['coords']:
            self._data.append(current_metrics)

    # --- 2. Создание профиля ---
    def calibration_profile(self):
        if not self._data:
            print("Нет данных для калибровки.")
            return

        # 1. Профиль координат (старая логика - границы движений)
        first_frame_coords = self._data[0]['coords']
        coord_profile = {k: {'min_x': 1.0, 'max_x': 0.0, 'min_y': 1.0, 'max_y': 0.0} for k in first_frame_coords.keys()}

        # 2. Профиль углов (новая логика - биомеханика)
        first_frame_angles = self._data[0]['angles']
        angle_profile = {k: {'min': 360.0, 'max': -360.0} for k in first_frame_angles.keys()}

        for frame in self._data:
            # Обработка координат
            for idx, coords in frame['coords'].items():
                if idx not in coord_profile: continue
                if coords['x'] < coord_profile[idx]['min_x']: coord_profile[idx]['min_x'] = coords['x']
                if coords['x'] > coord_profile[idx]['max_x']: coord_profile[idx]['max_x'] = coords['x']
                if coords['y'] < coord_profile[idx]['min_y']: coord_profile[idx]['min_y'] = coords['y']
                if coords['y'] > coord_profile[idx]['max_y']: coord_profile[idx]['max_y'] = coords['y']

            # Обработка углов
            for name, val in frame['angles'].items():
                if val < angle_profile[name]['min']: angle_profile[name]['min'] = val
                if val > angle_profile[name]['max']: angle_profile[name]['max'] = val

        self.motion_profile = coord_profile
        self.angle_profile = angle_profile
        print(f"Профиль создан. Кадров: {len(self._data)}")
        print(f"Угловой профиль: {self.angle_profile}")  # Для отладки

        self._data = []

    # --- 3. Проверка отклонений (Мониторинг) ---
    def calibrate(self, frame, results, tolerance=0.05, angle_tolerance=15):
        """
        tolerance: допуск для координат (0.05 = 5% экрана)
        angle_tolerance: допуск для углов в градусах (например, 15 градусов свободы)
        """
        if not self.motion_profile or not results.pose_landmarks:
            return frame

        is_incorrect = False
        h, w, _ = frame.shape

        # Получаем текущие метрики
        current_metrics = self._get_metrics(results.pose_landmarks)

        # --- ПРОВЕРКА 1: ГЕОМЕТРИЯ (УГЛЫ) - Основная проверка ---
        # Проверяем, не сгорбился ли человек, не наклонил ли голову неправильно и т.д.
        for angle_name, value in current_metrics['angles'].items():
            bounds = self.angle_profile.get(angle_name)
            if not bounds: continue

            # Если угол выходит за рамки Min/Max + допуск
            if (value < bounds['min'] - angle_tolerance) or \
                    (value > bounds['max'] + angle_tolerance):
                is_incorrect = True
                # Можно добавить print(f"Fail on {angle_name}: {value}") для отладки

        # --- ПРОВЕРКА 2: КООРДИНАТЫ (ПОЗИЦИЯ) - Вспомогательная ---
        # Используем старую логику, чтобы подсветить конкретную "убежавшую" точку красным
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            if idx in self.excluded_landmarks or idx not in self.motion_profile:
                continue

            bounds = self.motion_profile[idx]

            # Тут можно дать больше свободы (tolerance * 1.5), так как мы полагаемся на углы
            pos_tolerance = tolerance

            point_error = False
            if (landmark.x < bounds['min_x'] - pos_tolerance) or \
                    (landmark.x > bounds['max_x'] + pos_tolerance) or \
                    (landmark.y < bounds['min_y'] - pos_tolerance) or \
                    (landmark.y > bounds['max_y'] + pos_tolerance):
                point_error = True
                is_incorrect = True  # Если точка улетела совсем далеко, тоже ошибка

            if point_error:
                # Визуально помечаем "плохую" точку красным
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)

                # Общая визуализация ошибки
        if is_incorrect:
            cv2.putText(frame, "INCORRECT MOVEMENT", (frame.shape[1] - 650, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)

        return frame


    # --- Основные сессии (без изменений логики отрисовки) ---

    def calibration_session(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(frame_rgb)

        if results.pose_landmarks:
            self._draw_upper_body_only(frame, results.pose_landmarks)
            self.get_stat(results)

        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(255, 0, 0)))
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(255, 0, 0)))

        return frame

    def monitoring_session(self, frame, tolerance=0.1, angle_tolerance=20):
        """
        Добавлен параметр angle_tolerance (в градусах)
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(frame_rgb)

        if results.pose_landmarks:
            self._draw_upper_body_only(frame, results.pose_landmarks)
            # Передаем новый параметр допуска углов
            frame = self.calibrate(frame, results, tolerance=tolerance, angle_tolerance=angle_tolerance)

        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(255, 0, 0)))
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(255, 0, 0)))

        return frame

    def _draw_upper_body_only(self, image, landmarks):
        h, w, _ = image.shape
        for idx, landmark in enumerate(landmarks.landmark):
            if idx in self.excluded_landmarks:
                continue
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            # Рисуем синим, если точка не перекрашена в красный внутри calibrate
            # (Но cv2.circle рисует поверх, поэтому здесь рисуем базу)
            cv2.circle(image, (cx, cy), 4, (255, 0, 0), -1)

        for connection in self.mp_holistic.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx not in self.excluded_landmarks and end_idx not in self.excluded_landmarks:
                start_pt = landmarks.landmark[start_idx]
                end_pt = landmarks.landmark[end_idx]
                start_px = (int(start_pt.x * w), int(start_pt.y * h))
                end_px = (int(end_pt.x * w), int(end_pt.y * h))
                cv2.line(image, start_px, end_px, (255, 255, 255), 2)