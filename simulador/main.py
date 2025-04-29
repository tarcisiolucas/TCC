# -*- coding: utf-8 -*-
# Disciplina: Tópicos em Engenharia de Controle e Automação IV (ENG075): 
# Fundamentos de Veículos Autônomos - 2024/1
# Professores: Armando Alves Neto e Leonardo A. Mozelli
# Cursos: Engenharia de Controle e Automação
# DELT – Escola de Engenharia
# Universidade Federal de Minas Gerais
########################################
import class_car as cp
import controllers as ct
import numpy as np
import cv2
import matplotlib.pyplot as plt
import multiprocessing  # Alterado de threading para multiprocessing
import os


plt.rcParams['figure.figsize'] = (8,6)

# Globais
car_zero = { 'car_id': 0,
             'ts': 5.0,             # tempo da simulacao
             'save': True,
             'logfile': 'logs/',
           }

car_um = { 'car_id': 1,
           'ts': 5.0,              # tempo da simulacao
           'save': True,
           'logfile': 'logs/',
         }

########################################
# Função para execução em processos
########################################
def leader_process_func(car_parameters):
    car = cp.Car(car_parameters)  # Cria o objeto Car dentro do processo
    car.startMission()

    while car.t <= 10:
        ct.leader_control(car)

def follower_process_func(car_parameters):
    car = cp.Car(car_parameters)  # Cria o objeto Car dentro do processo
    car.startMission()
    
    while car.t <= 10:
        ct.follower_control(car, ct.follower_vision(car))


def run(parameters):
    plt.figure(1)
    plt.ion()

    # Substituindo threads por processos e movendo a criação do carro para dentro da função
    car_process = multiprocessing.Process(target=leader_process_func, args=(parameters[0],))
    follower_process = multiprocessing.Process(target=follower_process_func, args=(parameters[1],))
    
    car_process.start()
    follower_process.start()
    
    car_process.join()
    follower_process.join()
    
    plt.ioff()
    print('Terminou...')

########################################
########################################
if __name__ == "__main__":
    run([car_zero, car_um])
