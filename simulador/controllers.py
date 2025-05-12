import class_car as cp
import cv2
import numpy as np
import cv2.aruco as aruco
import pid

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


def follower_control(car):
    prev_pose = None
    while car.t <= 15:
        car.step()
        # Seta direção
        car.setSteer(np.deg2rad(0))
        pose = follower_vision(car)

        if (pose):
            print(f"distância: {pose["position"][2]}")

            distance_function(car, pose, prev_pose)

            prev_pose = pose

    return

def leader_control(car):
    # Seta direção
    car.setSteer(np.deg2rad(0))

    # Seta aceleração
    car.step()
    while car.t < 4:
        car.step()
        car.setVel(1.5)
    while car.t > 4 and car.t < 7:
        car.step()
        car.setVel(0)
    while car.t > 7:
        car.step()
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

def distance_function(car, pose, prev_pose): 
    # Parâmetros do controlador
    time_gap = 0.8
    lambda_const = 7.0
    dt = 0.05
    
    if (prev_pose == None): 
        prev_pose = pose

    epsilon = -pose['position'][2]
    v, w = car.getVel()

    epsilon_dot = (pose["position"][2] - pose["position"][2]) / dt

    spacing = epsilon + time_gap*v
    u = -(1/time_gap)*(epsilon_dot + lambda_const*spacing)

    if(epsilon > -0.5) :
        car.setVel(0)
    else: 
        car.setU(u)

    print (f"velocidade: {v}")
    return u