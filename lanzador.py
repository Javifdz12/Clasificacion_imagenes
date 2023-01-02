import helpers
from capa_convolucion import capa_convolucion
from capa_convolucion_aumentada_1 import capa_convolucion_aumentada_1
from capa_convolucion_aumentada_4 import capa_convolucion_aumentada_4
import pandas as pnd
def menu():
    print("========================")
    print(" BIENVENIDO A CLASIFICACIÓN_IMAGENES ")
    print("========================")
    print("[1] RED NEURONAL CONVOLUCIONAL CON 1 CAPA DE CONVOLUCION ")
    print("[2] RED NEURONAL CON 1 CAPA DE CONVOLUCION Y UNA CANTIDAD DE IMAGENES AUMENTADA ")
    print("[3] RED NEURONAL DE 4 CAPAS DE CONVOLUCIONES CON UNA CANTIDAD DE IMAGENES EN AUMENTO ")
    print("[4] Salir ")
    print("========================")

def lanzar():
    red1=capa_convolucion(28,28,pnd.read_csv('fashion-mnist_train.csv'),pnd.read_csv('fashion-mnist_train.csv'))
    red2=capa_convolucion_aumentada_1(28,28,pnd.read_csv('fashion-mnist_train.csv'),pnd.read_csv('fashion-mnist_train.csv'))
    red3=capa_convolucion_aumentada_4(28,28,pnd.read_csv('fashion-mnist_train.csv'),pnd.read_csv('fashion-mnist_train.csv'))

    while True:
        menu()
        opcion=int(input("> "))
        helpers.limpiar_pantalla()

        if opcion == 1:
            red1.iniciar()
            print("========================")
            print(" ¿Quieres visualizar datos de precisión, validación y error? ")
            print("========================")
            print("[1] Si ")
            print("[2] No ")
            print("========================")
            opcion2=int(input("> "))
            if opcion2==1:
                red1.visualizar()
            else:pass

        if opcion == 2:
            red2.iniciar()
            print("========================")
            print(" ¿Quieres visualizar datos de precisión, validación y error? ")
            print("========================")
            print("[1] Si ")
            print("[2] No ")
            print("========================")
            opcion3=int(input("> "))
            if opcion3 == 1:
                red2.visualizar()
            else:pass

            print("========================")
            print(" ¿Quieres guardar el modelo? ")
            print("========================")
            print("[1] Si ")
            print("[2] No ")
            print("========================")
            opcion4=int(input("> "))
            if opcion4 == 1:
                red2.guardar()
            else:pass

        if opcion == 3:
            red3.iniciar()
            print("========================")
            print(" ¿Quieres visualizar datos de precisión, validación y error? ")
            print("========================")
            print("[1] Si ")
            print("[2] No ")
            print("========================")
            opcion5=int(input("> "))
            if opcion5==1:
                red3.visualizar()
            else:pass

            print("========================")
            print(" ¿Quieres guardar el modelo? ")
            print("========================")
            print("[1] Si ")
            print("[2] No ")
            print("========================")
            opcion6=int(input("> "))
            if opcion6==1:
                red3.guardar()
            else:pass

        if opcion == 4:
            print("Saliendo...\n")
            break