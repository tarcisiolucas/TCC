import cv2
import cv2.aruco as aruco
import numpy as np

# Parâmetros da câmera (matriz intrínseca e distorção) - use os seus reais, se possível
# Esses valores são aproximados e servem para teste
camera_matrix = np.array([[800, 0, 320],
                          [0, 800, 240],
                          [0,   0,   1]], dtype=np.float32)

dist_coeffs = np.zeros((5, 1))  # sem distorção

# Tamanho real do marcador ArUco (em metros)
marker_length = 0.05  # 5 cm

# Inicializa a captura de vídeo
cap = cv2.VideoCapture(0)

# Usa dicionário padrão ArUco
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Converte para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta marcadores
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        # Estima a pose dos marcadores
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

        for i, marker_id in enumerate(ids.flatten()):
            # Desenha o marcador e a pose
            aruco.drawDetectedMarkers(frame, corners)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.03)

            # Extrai pose
            tvec = tvecs[i][0]  # posição (x, y, z)
            rvec = rvecs[i][0]  # orientação (vetor de rotação)

            # Estimativa de "tamanho aparente" (distância da câmera)
            distance = np.linalg.norm(tvec)

            # Exibe informações
            text = f"ID: {marker_id} | Pos: {tvec.round(2)} | Dist: {distance:.2f}m"
            cv2.putText(frame, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Mostra o resultado
    cv2.imshow("ArUco Detection", frame)

    # Sai com ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
