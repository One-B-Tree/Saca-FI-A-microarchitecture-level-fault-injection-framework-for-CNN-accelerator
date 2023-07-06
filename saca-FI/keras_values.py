import numpy as np
import cv2
import gc
import os
import keras
import time
from tool.writein_csv import write_csv,write_txt
#from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras import backend as K
from keras.models import Sequential,Model,load_model
from keras.layers import Dense,Dropout,Convolution2D,Flatten
from keras.datasets import cifar10,mnist
# import matplotlib.pyplot as plt
from keras.applications import MobileNet,MobileNetV2,mobilenet
from keras.applications import ResNet50,resnet50
from keras.applications import VGG16,vgg16
from keras.applications import InceptionResNetV2
from keras.applications import imagenet_utils
from keras.utils import np_utils
from numba import cuda
np.set_printoptions(threshold=np.inf,suppress=True)
os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'

###The starting layer of each model is different, as are the positions of the COV and FC layers
def lenet_layer(layer):
    layer_k = layer
    path_mod = 'mode/modelnew.h5'
    image_data = pic_mnist()
    #image_data = pic_test()
    model = load_model(path_mod)

    #aquire the weight array
    weights_1 = model.layers[layer_k].get_weights()
    Wt_in = weights_1[0]
    print('W:',Wt_in.shape)
    bias = model.layers[layer_k].get_weights()[1]

    #aquire the input array
    if layer==0:
        Iofmap = image_data[0]
    else:
        ofmap = model.layers[layer_k].input
        sim_lay = Model(inputs=model.inputs, outputs=ofmap)
        s_rs = sim_lay.predict(image_data)
        Iofmap=s_rs[0]
    print('Input:',Iofmap.shape)
    return Wt_in, Iofmap, bias

def cifar_layer(layer=0):
    layer_k = layer
    path_mod = 'mode/cifar.h5'
    image_data = pic_cifar()
    model = load_model(path_mod)

    weights_1 = model.layers[layer_k].get_weights()
    Wt_in = weights_1[0]
    print('W:',Wt_in.shape)
    bias = model.layers[layer_k].get_weights()[1]

    #aquire the input array
    if layer==0:
        Iofmap = image_data[0]
    else:
        ofmap = model.layers[layer_k].input
        sim_lay = Model(inputs=model.inputs, outputs=ofmap)
        s_rs = sim_lay.predict(image_data)
        Iofmap=s_rs[0]
    print('Input:',Iofmap.shape)

    K.clear_session()
    return Wt_in, Iofmap, bias

def mobilenetv2_layer(layer=1):
    model = MobileNetV2(weights='imagenet', include_top=True)
    model.summary()

def vgg16_layer(layer=1):
    img=pic_ilsvrc2012()
    path='mode/vgg16_weights.h5'
    model= VGG16(weights=path,include_top=True)
    img=vgg16.preprocess_input(img)

    Wt_in = model.layers[layer].get_weights()
    bias=model.layers[layer].get_weights()[1]

    if layer==1:
        Iofmap=img[0]
    else:
        per_lay=1
        ofmap=model.layers[layer-per_lay].output
        sim_layer=Model(inputs=model.inputs,outputs=ofmap)
        s_rs=sim_layer.predict(img)
        Iofmap = s_rs[0]
        del s_rs
    del img
    gc.collect()
    return Wt_in[0],Iofmap,bias

def lenet_fc_layer(layer):
    layer_k = layer
    path_mod = 'mode/modelnew.h5'
    image_data = pic_mnist()
    model = load_model(path_mod)

    weights_1 = model.layers[layer_k].get_weights()
    Wt_in = weights_1[0]
    bias = model.layers[layer_k].get_weights()[1]

    ofmap = model.layers[layer_k].input
    sim_lay = Model(inputs=model.inputs, outputs=ofmap)
    s_rs = sim_lay.predict(image_data)
    Iofmap = s_rs[0]
    #expend the dimension of the weight and ifmap from fc layer
    Wt_in=Wt_in[np.newaxis,:]
    Wt_in = Wt_in[np.newaxis, :]
    Iofmap = Iofmap[np.newaxis, :]
    Iofmap = Iofmap[np.newaxis, :]
    print('weight', Wt_in.shape)
    print('input', Iofmap.shape)
    return Wt_in, Iofmap, bias

def cifar_fc_layer(layer):
    path_mod = 'mode/cifar.h5'
    image_data = pic_cifar()
    model = load_model(path_mod)

    weights_1 = model.layers[layer].get_weights()
    Wt_in = weights_1[0]
    bias = model.layers[layer].get_weights()[1]

    ofmap = model.layers[layer].input
    sim_lay = Model(inputs=model.inputs, outputs=ofmap)
    s_rs = sim_lay.predict(image_data)
    Iofmap = s_rs[0]
    #expend the dimension of the weight and ifmap from fc layer
    Wt_in=Wt_in[np.newaxis,:]
    Wt_in = Wt_in[np.newaxis, :]
    Iofmap = Iofmap[np.newaxis, :]
    Iofmap = Iofmap[np.newaxis, :]
    print('weight', Wt_in.shape)
    print('input', Iofmap.shape)
    return Wt_in, Iofmap, bias

def vgg16_fc_layer(layer):
    img = pic_ilsvrc2012()
    path = 'mode/vgg16_weights.h5'
    model = VGG16(weights=path, include_top=True)
    img = vgg16.preprocess_input(img)

    weights_1 = model.layers[layer].get_weights()
    Wt_in = weights_1[0]
    bias = model.layers[layer].get_weights()[1]

    ofmap = model.layers[layer].input
    sim_lay = Model(inputs=model.inputs, outputs=ofmap)
    s_rs = sim_lay.predict(img)
    Iofmap = s_rs[0]
    # expend the dimension of the weight and ifmap from fc layer
    Wt_in = Wt_in[np.newaxis, :]
    Wt_in = Wt_in[np.newaxis, :]
    Iofmap = Iofmap[np.newaxis, :]
    Iofmap = Iofmap[np.newaxis, :]
    print('weight', Wt_in.shape)
    print('input', Iofmap.shape)
    return Wt_in, Iofmap, bias

def vgg16_model(mid_data,layer=1):
    #Get the final output from layer+1
    print(' Get fin predict from layer %d :' % (layer +1))
    path = 'mode/vgg16_weights.h5'
    model = VGG16(weights=path, include_top=True)
    data=mid_data
    #data = np.expand_dims(mid_data,axis=0)
    new_mod = K.function([model.layers[layer + 1].input], [model.layers[-1].output])
    if layer>=20:
        rs = new_mod([data[0][0]])[0]
    else:
        rs = new_mod([data])[0]
    Rs = vgg16.decode_predictions(np.array(rs), top=5)
    del data, rs
    gc.collect()
    print(Rs)
    K.clear_session()
    return Rs

def lenet_model(mid_data,layer=1):
    path_mod = 'mode/modelnew.h5'
    model = load_model(path_mod)
    data=mid_data
    new_mod = K.function([model.layers[layer+1].input], [model.layers[-1].output])
    if layer==5:
        rs = new_mod([data[0][0]])[0]
    else:
        rs = new_mod([data])[0]
    list_prediction = rs.tolist()
    print(list_prediction)
    K.clear_session()
    return list_prediction

def lenet_mid(mid_data,layer=0,next_layer=1):
    path_mod = 'mode/modelnew.h5'
    model = load_model(path_mod)
    if layer+1==next_layer:
        return mid_data

    data=mid_data
    ifmap=model.layers[layer + 1].input
    ofmap=model.layers[next_layer-1].output

    new_mod = K.function([ifmap], [ofmap])
    if layer == 5:
        rs = new_mod([data[0][0]])[0]
    else:
        rs = new_mod([data])[0]

    K.clear_session()
    return rs

def cifar_model(mid_data,layer=1):
    print(' Get fin predict from layer %d :' % (layer + 1))
    path_mod = 'mode/cifar.h5'
    model = load_model(path_mod)
    data=mid_data

    new_mod = K.function([model.layers[layer + 1].input], [model.layers[-1].output])
    if layer<=5:
        rs = new_mod([data])[0]
    else:
        rs = new_mod([data[0][0]])[0]
    prediction_rs=rs.tolist()
    print(prediction_rs)

    K.clear_session()
    return prediction_rs

def cifar_mid(mid_data,layer=0,next_layer=1):
    path_mod = 'mode/cifar.h5'
    model = load_model(path_mod)
    if layer + 1 == next_layer:
        return mid_data

    data = mid_data
    ifmap = model.layers[layer + 1].input
    ofmap = model.layers[next_layer - 1].output

    new_mod = K.function([ifmap], [ofmap])
    if layer > 5:
        rs = new_mod([data[0][0]])[0]
    else:
        rs = new_mod([data])[0]

    K.clear_session()
    return rs

def mobilenet_model(mid_data,layer=3):
    print(' Get fin predict from layer %d :' % (layer + 1))
    model = MobileNet(weights='imagenet', include_top=True)
    data = mid_data
    new_mod = K.function([model.layers[layer + 1].input], [model.layers[-1].output])
    rs = new_mod([data])[0]
    Rs = mobilenet.decode_predictions(np.array(rs), top=5)
    del data, rs
    gc.collect()
    print(Rs)
    K.clear_session()
    return Rs

def change_to_form(rs):
    map = rs[0].swapaxes(1, 2)
    RS = map.swapaxes(0, 1).copy()
    return RS


def pic_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    image_data = X_test[3]
    # plt.imshow(image_data, cmap=plt.get_cmap('gray'))
    # plt.show()
    image_data = (image_data.reshape(1, 28, 28, 1)).astype("float32") / 255
    return image_data

def pic_test():
    im=np.full((1,28,28,1),1)
    return im

def pic_vgg16():
    image_path = "pic/witcher24.png"
    im = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(im)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    img = imagenet_utils.preprocess_input(img)
    return img

def pic_cifar():
    t=60#Select image number 60
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    image_data = X_test[t]
    # plt.imshow(image_data, cmap=plt.get_cmap('gray'))
    # plt.show()
    image_data = (image_data.reshape(1, 32, 32, 3)).astype("float32") / 255
    return image_data

def pic_ilsvrc2012():
    pics = [4911]
    image_path='pic/ILSVRC2012_test_00000456.JPEG'
    im = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(im)
    image_data = img.reshape((1,) + img.shape)
    return image_data


def model_in(layer=2):
    ###vgg16
    # path = 'mode/vgg16_weights.h5'
    # model = VGG16(weights=path, include_top=True)
    # image_data= pic_ilsvrc2012()
    # image_data = vgg16.preprocess_input(image_data)
    ##cifar
    path_mod = 'mode/cifar.h5'
    image_data = pic_cifar()
    model = load_model(path_mod)

    #aquire the input array
    if layer==0:#
        Iofmap = image_data[0]
    else:
        ofmap = model.layers[layer].input
        sim_lay = Model(inputs=model.inputs, outputs=ofmap)
        s_rs = sim_lay.predict(image_data)
        Iofmap=s_rs[0]
    print('Input:',Iofmap.shape)

    pp = 'cifar1os.txt'
    ddc = open(pp, 'w')
    errin_img = Iofmap.swapaxes(1, 2)
    errin_img = errin_img.swapaxes(0, 1).copy()
    print(errin_img, file=ddc)

    return Iofmap

def vgg16_in(layer):
    #Get the input for the layer layer
    layer_k = layer
    image_data = pic_ilsvrc2012()
    path = 'mode/vgg16_weights.h5'
    model = VGG16(weights=path, include_top=True)

    #aquire the input array
    if layer==1:
        Iofmap = image_data[0]
    else:
        ofmap = model.layers[layer_k].input
        sim_lay = Model(inputs=model.inputs, outputs=ofmap)
        s_rs = sim_lay.predict(image_data)
        Iofmap=s_rs[0]
    print('layer %d Output:'%(layer-1),Iofmap.shape)
    return Iofmap

if __name__ == "__main__":
    model_in()


