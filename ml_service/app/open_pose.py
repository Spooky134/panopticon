import cv2 as cv
import numpy as np

class OpenPoseProcessor:
    def __init__(self, model_path="graph_opt.pb", threshold=0.2, width=368, height=368):
        self.BODY_PARTS = { 
            "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
            "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
            "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
            "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 
        }
        
        self.POSE_PAIRS = [ 
            ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
            ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
            ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
            ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
            ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] 
        ]
        
        self.threshold = threshold
        self.inWidth = width
        self.inHeight = height
        
        # Загрузка модели
        self.net = cv.dnn.readNetFromTensorflow(model_path)
        
    def process_frame(self, frame):
        """
        Обрабатывает кадр и возвращает кадр с нарисованными позами
        """
        if frame is None:
            return None
            
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        
        # Подготовка входных данных для нейросети
        blob = cv.dnn.blobFromImage(frame, 1.0, (self.inWidth, self.inHeight), 
                                  (127.5, 127.5, 127.5), swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()
        out = out[:, :19, :, :]  # Берем только первые 19 элементов
        
        assert(len(self.BODY_PARTS) == out.shape[1])
        
        # Поиск ключевых точек
        points = []
        for i in range(len(self.BODY_PARTS)):
            heatMap = out[0, i, :, :]
            _, conf, _, point = cv.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]
            points.append((int(x), int(y)) if conf > self.threshold else None)
        
        # Рисуем скелет на кадре
        for pair in self.POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            
            idFrom = self.BODY_PARTS[partFrom]
            idTo = self.BODY_PARTS[partTo]
            
            if points[idFrom] and points[idTo]:
                cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
                cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
        
        # Добавляем информацию о производительности
        t, _ = self.net.getPerfProfile()
        freq = cv.getTickFrequency() / 1000
        cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        
        return frame
    
    def get_keypoints(self, frame):
        """
        Возвращает только ключевые точки без визуализации
        """
        if frame is None:
            return None
            
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        
        blob = cv.dnn.blobFromImage(frame, 1.0, (self.inWidth, self.inHeight), 
                                  (127.5, 127.5, 127.5), swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()
        out = out[:, :19, :, :]
        
        points = {}
        for name, i in self.BODY_PARTS.items():
            heatMap = out[0, i, :, :]
            _, conf, _, point = cv.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]
            if conf > self.threshold:
                points[name] = (int(x), int(y), conf)
            else:
                points[name] = None
        
        return points