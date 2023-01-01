import pandas as pnd
import numpy as np
import matplotlib.pyplot as plt
import keras
import time
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization

class capa_convolucion_aumentada_4:
    def __init__(self,largo_imagen,ancho_imagen,observaciones_entrenamiento,observaciones_test):
        self.largo_imagen = largo_imagen
        self.ancho_imagen = ancho_imagen
        self.observaciones_entrenamiento=observaciones_entrenamiento
        self.X = np.array(self.observaciones_entrenamiento.iloc[:, 1:])
        self.y = to_categorical(np.array(observaciones_entrenamiento.iloc[:, 0]))
        self.X_aprendizaje,self.X_validacion, self.y_aprendizaje, self.y_validacion = train_test_split(self.X, self.y, test_size=0.2, random_state=13)
        self.X_validacion = self.X_validacion.reshape(self.X_validacion.shape[0], self.ancho_imagen, self.largo_imagen, 1)
        self.X_validacion = self.X_validacion.astype('float32')
        self.X_validacion /= 255
        self.observaciones_test = observaciones_test
        self.X_test = np.array(self.observaciones_test.iloc[:, 1:])
        self.y_test = to_categorical(np.array(self.observaciones_test.iloc[:, 0]))
        self.X_test = self.X_test.reshape(self.X_test.shape[0], self.ancho_imagen, self.largo_imagen, 1)
        self.X_test = self.X_test.astype('float32')
        self.X_test /= 255
        self.dimensionImagen = (self.ancho_imagen, self.largo_imagen, 1)
        self.redNeuronaConvolucion = Sequential()

    def evaluar(self):
        evaluacion = self.redNeuronaConvolucion.evaluate(self.X_test, self.y_test, verbose=0)
        print('Error:', evaluacion[0])
        print('Precisión:', evaluacion[1])

    def visualizar(self):
        #Datos de precisión (accuracy)
        plt.plot(self.historico_aprendizaje.history['accuracy'])
        plt.plot(self.historico_aprendizaje.history['val_accuracy'])
        plt.title('Precisión del modelo')
        plt.ylabel('Precisión')
        plt.xlabel('Epoch')
        plt.legend(['Aprendizaje', 'Test'], loc='upper left')
        plt.show()

        #Datos de validación y error
        plt.plot(self.historico_aprendizaje.history['loss'])
        plt.plot(self.historico_aprendizaje.history['val_loss'])
        plt.title('Error')
        plt.ylabel('Error')
        plt.xlabel('Epoch')
        plt.legend(['Aprendizaje', 'Test'], loc='upper left')
        plt.show()

        #Guardado del modelo
    def guardar(self):
        # serializar modelo a JSON
        modelo_json = self.redNeuronaConvolucion.to_json()
        with open("modelo/modelo_4convoluciones.json", "w") as json_file:
            json_file.write(modelo_json)

        # serializar pesos a HDF5
        self.redNeuronaConvolucion.save_weights("modelo/modelo_4convoluciones.h5")
        print("¡Modelo guardado!")

    def iniciar(self):
        self.redNeuronaConvolucion.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.dimensionImagen))
        self.redNeuronaConvolucion.add(BatchNormalization())
        self.redNeuronaConvolucion.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        self.redNeuronaConvolucion.add(BatchNormalization())
        self.redNeuronaConvolucion.add(MaxPooling2D(pool_size=(2, 2)))
        self.redNeuronaConvolucion.add(Dropout(0.25))
        self.redNeuronaConvolucion.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.redNeuronaConvolucion.add(BatchNormalization())
        self.redNeuronaConvolucion.add(Dropout(0.25))
        self.redNeuronaConvolucion.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.redNeuronaConvolucion.add(BatchNormalization())
        self.redNeuronaConvolucion.add(MaxPooling2D(pool_size=(2, 2)))
        self.redNeuronaConvolucion.add(Dropout(0.25))
        self.redNeuronaConvolucion.add(Flatten())
        self.redNeuronaConvolucion.add(Dense(512, activation='relu'))
        self.redNeuronaConvolucion.add(BatchNormalization())
        self.redNeuronaConvolucion.add(Dropout(0.5))
        self.redNeuronaConvolucion.add(Dense(128, activation='relu'))
        self.redNeuronaConvolucion.add(BatchNormalization())
        self.redNeuronaConvolucion.add(Dropout(0.5))
        self.redNeuronaConvolucion.add(Dense(10, activation='softmax'))
        self.redNeuronaConvolucion.compile(loss=keras.losses.categorical_crossentropy,
                                        optimizer=keras.optimizers.Adam(),
                                        metrics=['accuracy'])
        generador_imagenes = ImageDataGenerator(rotation_range=8,
                                width_shift_range=0.08,
                                shear_range=0.3,
                                height_shift_range=0.08,
                                zoom_range=0.08)
        nuevas_imagenes_aprendizaje = generador_imagenes.flow(self.X_aprendizaje, self.y_aprendizaje, batch_size=256)
        nuevas_imagenes_validacion = generador_imagenes.flow(self.X_validacion, self.y_validacion, batch_size=256)
        start = time.clock();
        self.historico_aprendizaje = self.redNeuronaConvolucion.fit_generator(nuevas_imagenes_aprendizaje,
                                                        steps_per_epoch=48000//256,
                                                        epochs=50,
                                                        validation_data=nuevas_imagenes_validacion,
                                                        validation_steps=12000//256,
                                                        use_multiprocessing=False,
                                                        verbose=1 )
        stop = time.clock();
        print("Tiempo de aprendizaje = "+str(stop-start))
        self.evaluar()




