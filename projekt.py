import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten, Dropout
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.layers.convolutional import Conv2D


def wypakuj_plik(file):
    import pickle
    fo=open(file,'rb')
    dict=pickle.load(fo,encoding='latin1')
    fo.close()
    return dict


def przygotuj_dane_przykladow_treningowych():
    batch1_data = Batch1['data']
    batch2_data = Batch2['data']
    batch3_data = Batch3['data']
    batch4_data = Batch4['data']
    batch5_data = Batch5['data']
    dane_przyklady = np.concatenate((batch1_data,
                              batch2_data,
                              batch3_data,
                              batch4_data,
                              batch5_data))
    
    dane_przyklady = dane_przyklady.reshape([-1,3,32,32]).transpose([0,2,3,1]) 
    dane_przyklady = dane_przyklady[0:Liczba_przykladow_treningowych,:,:,:]
    return dane_przyklady


def przygotuj_dane_kategorie_przykladow_treningowych():
    batch1_labels=Batch1['labels']
    batch2_labels=Batch2['labels']
    batch3_labels=Batch3['labels']
    batch4_labels=Batch4['labels']
    batch5_labels=Batch5['labels']
    dane_przyklady_kategorie = np.concatenate((batch1_labels,
                                               batch2_labels,
                                               batch3_labels,
                                               batch4_labels,
                                               batch5_labels))
    dane_przyklady_kategorie = dane_przyklady_kategorie[0:Liczba_przykladow_treningowych]
    
    return dane_przyklady_kategorie


def przygotuj_dane_przykladow_testowych():
    dane_przyklady_testowe = Batch_test['data'] 
    dane_przyklady_testowe = dane_przyklady_testowe.reshape([-1,3,32,32]).transpose([0,2,3,1])
    
    dane_przyklady_testowe = dane_przyklady_testowe[0:Liczba_przykladow_testowych,:,:,:]
    
    return dane_przyklady_testowe
    
    
def przygotuj_dane_kategorii_przykladow_testowych():
    y_test = np.asarray(Batch_test['labels'])
    return y_test[0:Liczba_przykladow_testowych]


def dodaj_warstwy_do_sieci_wariant1():
    model_sieci = Sequential()
    model_sieci.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
    model_sieci.add(MaxPooling2D(pool_size = (2,2)))
    model_sieci.add(Flatten())
    model_sieci.add(Dense(1024, activation='relu'))
    model_sieci.add(Dense(10, activation='softmax'))
    return model_sieci

def dodaj_warstwy_do_sieci_wariant2():
    model_sieci = Sequential()
    model_sieci.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(32, 32, 3)))
    model_sieci.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(32, 32, 3)))
    model_sieci.add(MaxPooling2D(pool_size = (2,2)))
    model_sieci.add(Flatten())
    model_sieci.add(Dropout(0.25))
    model_sieci.add(Dense(1024, activation='relu'))
    model_sieci.add(Dense(10, activation='softmax'))
    return model_sieci

def dodaj_warstwy_do_sieci_wariant3():
    model_sieci = Sequential()
    model_sieci.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(32, 32, 3)))
    model_sieci.add(Conv2D(32, kernel_size=(1, 1), activation='relu', input_shape=(32, 32, 3)))
    model_sieci.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(32, 32, 3)))
    model_sieci.add(MaxPooling2D(pool_size = (2,2)))
    model_sieci.add(Flatten())
    model_sieci.add(Dense(1024, activation='relu'))
    model_sieci.add(Dropout(0.25))
    model_sieci.add(Dense(612, activation='relu'))
    model_sieci.add(Dense(10, activation='softmax'))
    return model_sieci

def pokaz_wykres_dla_skutecznosci():
    plt.plot(Historia_treningu.history['acc'])
    plt.plot(Historia_treningu.history['val_acc'])
    plt.title('skutecznosc modelu')
    plt.ylabel('dokladnosc')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    
def pokaz_wykres_dla_bledow():
    plt.plot(Historia_treningu.history['loss'])
    plt.plot(Historia_treningu.history['val_loss'])
    plt.title('bledy modelu')
    plt.ylabel('blad')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
 
 
Liczba_przykladow_treningowych = 5000;
Liczba_przykladow_testowych = 1000;
Rozmiar_batch = 100;
Liczba_epok = 50;
    
Batch1=wypakuj_plik('dataset/data_batch_1')
Batch2=wypakuj_plik('dataset/data_batch_2')
Batch3=wypakuj_plik('dataset/data_batch_3')
Batch4=wypakuj_plik('dataset/data_batch_4')
Batch5=wypakuj_plik('dataset/data_batch_5')
Batch_test = wypakuj_plik('dataset/test_batch')

Przyklady_treningowe = przygotuj_dane_przykladow_treningowych()
Kategorie_treningowe = przygotuj_dane_kategorie_przykladow_treningowych()
Przyklady_testowe = przygotuj_dane_przykladow_testowych()
Kategorie_testowe = przygotuj_dane_kategorii_przykladow_testowych()


Model_sieci = dodaj_warstwy_do_sieci_wariant2()

Model_sieci.compile(loss='categorical_crossentropy', optimizer='adam',
    metrics=['accuracy'])

Historia_treningu = Model_sieci.model.fit(Przyklady_treningowe / 255.0,      to_categorical(Kategorie_treningowe),
          batch_size=Rozmiar_batch,
          epochs=Liczba_epok,
          validation_data=(Przyklady_testowe / 255.0, to_categorical(Kategorie_testowe)))

pokaz_wykres_dla_skutecznosci()
pokaz_wykres_dla_bledow()



    

    