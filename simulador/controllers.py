import class_car as cp
import cv2
import numpy as np
import cv2.aruco as aruco

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (8,6)

# Parâmetros da câmera (matriz intrínseca e distorção) - use os seus reais, se possível
# Esses valores são aproximados e servem para teste
camera_matrix = np.array([[800, 0, 320],
                          [0, 800, 240],
                          [0,   0,   1]], dtype=np.float32)

dist_coeffs = np.zeros((5, 1))  # sem distorção

# Tamanho real do marcador ArUco (em metros)
marker_length = 0.05  # 5 cm

# Inicializar dicionário e parâmetros uma vez
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

def follower_control(car, pose):
    car.step()
    print(pose)
    # Seta direção
    car.setSteer(np.deg2rad(0))

    # Seta aceleração
    car.setU(0.5)


    return

def leader_control(car):
    car.step()
    # Seta direção
    car.setSteer(np.deg2rad(0))

    # Seta aceleração
    car.setU(0.5)

    return

def follower_vision(car):
    # pega imagem
    image = car.getImage()
    image = cv2.flip(image, 0)

    result = detect_aruco_pose(image)

    pose = next((item for item in result if item["id"] == 1), None)

    return pose

def detect_aruco_pose(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(gray)

    results = []

    if ids is not None:
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

        for i, marker_id in enumerate(ids.flatten()):
            result = {
                'id': int(marker_id),
                'position': tvecs[i][0].tolist(),   # (x, y, z)
                'orientation': rvecs[i][0].tolist() # vetor de rotação (Rodrigues)
            }
            results.append(result)

    return results