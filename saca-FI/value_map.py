import numpy as np
import csv
import time
import math
import sys,gc
# import pandas as pd
import keras_values
import Config_info as Info
from scale_sim.trace_gen_wrapper import create_csv_info

def padding_data(input):
    iffmap = []
    input=change_to_form(input)
    for i in range(input.shape[0]):
        p = np.pad(input[i], ((1, 1), (1, 1)), 'constant', constant_values=(0, 0)) 
        iffmap.append(p)
    Input = np.array(iffmap)
    Input = change_to_form(Input, 'r')
    return Input

def stride_data():
    pass

def get_actual_data(mode='lenet',layer=0,type='cov'):
    if mode=='lenet':
        wt_in, ifmap, bias=lenet_data(layer=layer,type=type)
    elif mode=='vgg16':
        wt_in, ifmap, bias=vgg16_data(layer=layer,type=type)
    elif mode=='cifar':
        wt_in, ifmap, bias=cifar_data(layer=layer,type=type)
    elif mode=='mobilenet':
        wt_in, ifmap, bias = mobilenet_data(layer=layer, type=type)
    else:
        pass
    return wt_in, ifmap, bias

def lenet_data(layer=0,type='cov'):
    if type=='cov':
        wt_in, ifmap, bias = keras_values.lenet_layer(layer)##aviliabily num: 0,1 in lenet5
    elif type=='fc':
        wt_in, ifmap, bias = keras_values.lenet_fc_layer(layer)#fc layer : 5,7,in lenet5
    print("in keras values")
    return wt_in, ifmap, bias

def cifar_data(layer=0,type='cov'):
    if type=='cov':
        wt_in, ifmap, bias = keras_values.cifar_layer(layer)##aviliabily num: 0,1 in lenet5
    elif type=='fc':
        wt_in, ifmap, bias = keras_values.cifar_fc_layer(layer)#fc layer : 5,7,in lenet5
    print("in keras values")
    return wt_in, ifmap, bias

def vgg16_data(layer=0,type='cov'):
    if type=='cov':
        wt_in, ifmap, bias = keras_values.vgg16_layer(layer)##aviliabily num: in lenet5
    elif type=='fc':
        wt_in, ifmap, bias = keras_values.vgg16_fc_layer(layer)#fc layer : 5,7,in lenet5
    print("in keras values")
    ##########vgg16 model has a padding processing step
    if type=='cov':
        ifmap=padding_data(ifmap)
    return wt_in, ifmap, bias

def mobilenet_data(layer=0,type='cov'):
    if type=='cov':
        wt_in, ifmap, bias = keras_values.mobilenet_layer(layer)##aviliabily num: 0,1 in lenet5
    elif type=='fc':
        pass
    print("in keras values")
    return wt_in, ifmap, bias


def create_scale_csv(pe_size,layer=1,type='cov',mode='lenet'):
    wt_in, ifmap, bias = get_actual_data(mode=mode, layer=layer, type=type)
    #test_mdnet_conv1_24x24 = [114, 114, 128, 3, 3, 128, 1, 512, 512, 1
    num_filt = wt_in.shape[3]
    channel = wt_in.shape[2]
    w_h = wt_in.shape[0]
    w_w = wt_in.shape[1]
    # i_h = ifmap.shape[0]
    i_w = ifmap.shape[1]
    scale_info=[i_w,i_w,channel,w_h,w_w,num_filt,Info.stride,pe_size,pe_size,1]
    create_csv_info(config_info=scale_info,data_flow=Info.data_flow,mode=mode,layer=layer)


def readin_ws(pe_size,scale_path='',layer=0,type='cov',mode='lenet'):
    ##########read data from current keras' layer to memoery array order by scalesim so as to calculating in the following step
    ##aquire keras data
    wt_in, ifmap, bias = get_actual_data(mode=mode,layer=layer,type=type)

    ##aquire scalesim csv form
    WRR,ARR=loadscale_ws(pe_size,scale_path)

    ##aquire WRR and ARR by putting the correct data in the memory array
    V_IFMAP,V_WTIN,tip_add,tip_union,tip_divid,save_array,num_filt=loadvalue(WRR,ARR,wt_in,ifmap,pe_size)

    #return np.array(V_IFMAP), np.array(V_WTIN), bias, tip_add,tip_union,save_array
    return V_IFMAP,V_WTIN, bias, tip_add,tip_union,save_array

def readin_is(pe_size,scale_path='',layer=0,type='cov',mode='lenet'):
    ##########read data from current keras' layer to memoery array order by scalesim so as to calculating in the following step
    ##aquire keras data
    wt_in, ifmap, bias = get_actual_data(mode=mode,layer=layer,type=type)

    ##aquire scalesim csv form
    WRR,ARR=loadscale_ws_r(pe_size,scale_path)
    #test(ARR[97][0])

    ##aquire WRR and ARR by putting the correct data in the memory array
    if type=='cov':
        Add,Union=aquire_add_union_is(wt_in,ARR,pe_size)
        ARRR,union=modify_is(ARR,Add)
        #ARRR,union=ARR,22
        ori_len=len(ARRR)
        V_IFMAP,V_WTIN,tip_add,tip_union,tip_divid,save_array,num_filt=loadvalue(ARRR,WRR,wt_in,ifmap,pe_size,lod_type='is')
        now_len=len(V_IFMAP)
        if union!='':
            tip_union=union
        if ori_len!=now_len:
            ####If parallelism occurs, set the tip_union parameter to -1
            tip_union = -1
    elif type=='fc':
        V_IFMAP, V_WTIN, tip_add, tip_union, tip_divid, save_array,num_filt= loadvalue(ARR, WRR, wt_in, ifmap, pe_size,lod_type='is',layer_type='fc')
    else:
        print('Invalid input mode, please try: 1.cov  2.fc')
        return

    parallel_wrr=''
    if tip_union==-1:
        parallel_wrr=parallel_alter(WRR[0])
    return V_WTIN,V_IFMAP,bias,tip_add,tip_union,save_array,parallel_wrr

def readin_os(pe_size,scale_path='',layer=0,type='cov',mode='lenet'):
    wt_in, ifmap, bias = get_actual_data(mode=mode, layer=layer, type=type)

    WRR, ARR = loadscale_os(pe_size, scale_path)
    V_IFMAP, V_WTIN, tip_add, tip_union, tip_divid, save_array,num_filt= loadvalue(WRR, ARR, wt_in, ifmap, pe_size,'os')
    w_pix_num=wt_in.shape[0]*wt_in.shape[1]*wt_in.shape[2]

    o_s=ifmap.shape[0]-wt_in.shape[0]+1
    k_filt = math.ceil(o_s*o_s/pe_size)

    if type=='fc':
        tip_union=1
    V_WTIN[0]=mark_os(V_WTIN[0],w_num=w_pix_num,k_filt=k_filt,fin_part=num_filt,tip_union=tip_union)
    print('Os W shape:',V_WTIN[0].shape)
    return V_IFMAP,V_WTIN, bias, tip_add, tip_union, save_array,k_filt,w_pix_num,num_filt

def readin_ws_cross(pe_size,scale_path='',layer=0,type='cov',mode='lenet',Ifmap=''):
    ##aquire keras data
    wt_in, ifmap, bias = get_actual_data(mode=mode,layer=layer,type=type)

    if Ifmap!='':
        if type=='fc':
            ifmap=np.expand_dims(Ifmap, axis=0)
        else:
            ifmap=Ifmap[0]

    ##aquire scalesim csv form
    WRR,ARR=loadscale_ws(pe_size,scale_path)

    ##aquire WRR and ARR by putting the correct data in the memory array
    V_IFMAP,V_WTIN,tip_add,tip_union,tip_divid,save_array,num_filt=loadvalue(WRR,ARR,wt_in,ifmap,pe_size)

    #return np.array(V_IFMAP), np.array(V_WTIN), bias, tip_add,tip_union,save_array
    return V_IFMAP,V_WTIN, bias, tip_add,tip_union,save_array

def readin_is_cross(pe_size,scale_path='',layer=0,type='cov',mode='lenet',Ifmap=''):
    wt_in, ifmap, bias = get_actual_data(mode=mode,layer=layer,type=type)

    if Ifmap != '':
        if type=='fc':
            ifmap=np.expand_dims(Ifmap, axis=0)
        else:
            ifmap=Ifmap[0]

    WRR,ARR=loadscale_ws_r(pe_size,scale_path)

    ##aquire WRR and ARR by putting the correct data in the memory array
    if type=='cov':
        Add,Union=aquire_add_union_is(wt_in,ARR,pe_size)
        ARRR,union=modify_is(ARR,Add)
        #ARRR,union=ARR,22
        ori_len=len(ARRR)
        V_IFMAP,V_WTIN,tip_add,tip_union,tip_divid,save_array,num_filt=loadvalue(ARRR,WRR,wt_in,ifmap,pe_size,lod_type='is')
        now_len=len(V_IFMAP)
        if union!='':
            tip_union=union
        if ori_len!=now_len:
            ####If parallelism occurs, set the tip_union parameter to -1
            tip_union = -1
    elif type=='fc':
        V_IFMAP, V_WTIN, tip_add, tip_union, tip_divid, save_array,num_filt= loadvalue(ARR, WRR, wt_in, ifmap, pe_size,lod_type='is',layer_type='fc')
    else:
        print('Invalid input mode, please try: 1.cov  2.fc')
        return

    parallel_wrr=''
    if tip_union==-1:
        parallel_wrr=parallel_alter(WRR[0])
    return V_WTIN,V_IFMAP,bias,tip_add,tip_union,save_array,parallel_wrr

def readin_os_cross(pe_size,scale_path='',layer=0,type='cov',mode='lenet',Ifmap=''):
    wt_in, ifmap, bias = get_actual_data(mode=mode, layer=layer, type=type)

    if Ifmap != '':
        if type=='fc':
            ifmap=np.expand_dims(Ifmap, axis=0)
        else:
            ifmap=Ifmap[0]

    WRR, ARR = loadscale_os(pe_size, scale_path)
    V_IFMAP, V_WTIN, tip_add, tip_union, tip_divid, save_array,num_filt= loadvalue(WRR, ARR, wt_in, ifmap, pe_size,'os')

    w_pix_num=wt_in.shape[0]*wt_in.shape[1]*wt_in.shape[2]

    o_s=ifmap.shape[0]-wt_in.shape[0]+1
    k_filt = math.ceil(o_s*o_s/pe_size)

    if type=='fc':
        tip_union=1
    V_WTIN[0]=mark_os(V_WTIN[0],w_num=w_pix_num,k_filt=k_filt,fin_part=num_filt,tip_union=tip_union)
    print('Os W shape:',V_WTIN[0].shape)
    return V_IFMAP,V_WTIN, bias, tip_add, tip_union, save_array,k_filt,w_pix_num,num_filt


def aquire_add_union_is(wt_in,WRR,pe_size):
    num_filt = wt_in.shape[3]
    channel = wt_in.shape[2]
    w_h = wt_in.shape[0]
    w_w = wt_in.shape[1]
    w_pix_num = w_h * w_w * channel
    tip_add=math.ceil(w_pix_num/pe_size)
    tip_union=math.ceil(len(WRR)/tip_add)
    return tip_add,tip_union



def modify_is(MAP,tip_add):
    if len(MAP)==1:
        return MAP,''
    save=tip_add
    count=0
    for i in range(len(MAP)-save):
        MAP[i+save]=MAP[count]
        count+=1
        if count>save-1:
            count=0
    union=len(MAP)//save
    return MAP,union



def loadvalue(WRR,ARR,wt_in,ifmap,pe_size,lod_type='ws',layer_type='cov'):
    #load value in the memory based on the scalesim csv
    #because the scalesim requrie the elements order by specific label
    #the original input from keras:ifmap(28,28,1)(h,w,channel),weight(3,3,1,32)(h,w,c,num_filt)
    num_filt=wt_in.shape[3]
    channel=wt_in.shape[2]
    w_h=wt_in.shape[0]
    w_w=wt_in.shape[1]
    i_h=ifmap.shape[0]
    i_w=ifmap.shape[1]
    w_pix_num=w_h*w_w*channel
    i_v_tape,w_v_tape=[],[]
    tip_divid = []
    save_array=[]
    if lod_type=='is':
        if layer_type=='cov':
            for i in range(len(ARR)):
                tip_divid.append(math.floor(WRR[i].shape[0] / w_pix_num))
            WRR=divid_array(WRR,num=tip_divid)
            ARR=divid_array(ARR,num=tip_divid)
            o_s=(i_w-w_w)+1
            save_array=np.zeros((num_filt,o_s,o_s),dtype=np.float32)
        for i in range(len(ARR)):
            ARR[i] = np.flipud(ARR[i])
    if lod_type=='os':
        o_s=(i_w-w_w)+1
        save_array=np.zeros((num_filt,o_s,o_s),dtype=np.float32)

    for i in range(i_h):
        for j in range(i_w):
            for k in range(channel):
                i_v_tape.append(ifmap[i][j][k])

    for p in range(num_filt):
        for i in range(w_h):
            for j in range(w_w):
                for k in range(channel):
                    w_v_tape.append(wt_in[i][j][k][p])

    for k in range(len(WRR)):
        WRR[k]=WRR[k].astype(np.float32)
        for i in range(WRR[k].shape[0]):
            for j in range(WRR[k].shape[1]):
                key=WRR[k][i][j]
                if np.isnan(key):
                    continue
                try:
                    WRR[k][i][j]=w_v_tape[int(key)]
                except:
                    print(k,'Blocks:',WRR[k].shape,key)
                    print('Coordinate:',i,j)
                    print(len(w_v_tape))

    flag_arr=False
    for k in range(len(ARR)):
        if lod_type=='ws':
            flag_arr=True
        ARR[k]=np.float32(ARR[k])
        for i in range(ARR[k].shape[0]):
            for j in range(ARR[k].shape[1]):
                if flag_arr:
                    key = int(ARR[k][i][j])
                else:
                    key=ARR[k][i][j]
                if np.isnan(key):
                    continue
                ARR[k][i][j]=i_v_tape[int(key)]
    tip_add=math.ceil(w_pix_num/pe_size)
    tip_union=math.ceil(num_filt/pe_size)

    return ARR,WRR,tip_add,tip_union,tip_divid,save_array,num_filt



def loadscale_ws(pe_size,scale_path=''):
    '''
    read scalesim file and put key in the memory,in this step return two list:WRR,ARR
    in order to simplify calculation,reshape the martix to the specific form
    premise is that pe_size is a square
    '''


    WRR=readcsv(scale_path,top=pe_size+1,bottom=2*pe_size+1,key=1000000)
    ARR=readcsv(scale_path,top=1,bottom=pe_size+1)
    del WRR[-1],ARR[0]
    ###make the data as the expected form
    for i in range(len(WRR)):
        WRR[i]=np.flipud(WRR[i])
    for j in range(len(ARR)):
        ARR[j]=flip_right_90(ARR[j])

    return WRR,ARR

def loadscale_ws_r(pe_size,scale_path=''):
    #read scalesim file and put key in the memory,in this step return two list:WRR,ARR
    #in order to simplify calculation,reshape the martix to the specific form
    #premise is that pe_size is a square

    WRR=readcsv(scale_path,top=pe_size+1,bottom=2*pe_size+1)
    ARR=readcsv(scale_path,top=1,bottom=pe_size+1,key=1000000)
    del WRR[-1],ARR[0]
    for j in range(len(ARR)):
        ARR[j] = flip_right_90(ARR[j])
    return WRR,ARR

def loadscale_os(pe_size,scale_path=''):
    WRR = readcsv(scale_path, top=pe_size + 1, bottom=2 * pe_size + 1, key=1000000,type='os')
    ARR = readcsv(scale_path, top=1, bottom=pe_size + 1,type='os')
    if len(ARR)>1:
        del ARR[-1]
    if len(WRR)>1:
        del WRR[-1]
    for j in range(len(ARR)):
        ARR[j]=np.flip(ARR[j],1)
    for j in range(len(ARR)):
        ARR[j]=flip_right_90(ARR[j])
    print('WRR,ARR', len(WRR), len(ARR))
    return WRR,ARR

def readcsv(scale_path,top,bottom,key=0,type='ws/is',num=np.nan):
    #the Is and Ws csv file have the same form ,thus we can read them by same way
    WARR=[]
    warr_tap = []
    with open(scale_path, 'rt') as csvfile:
        reader = csv.reader(csvfile)
        warr = [row[top:bottom] for row in reader]

    save = False
    for i in range(len(warr)):
        if if_empty(warr[i]) and not(save):
            WARR.append(np.array(warr_tap))
            save=True
            warr_tap=[]
        elif if_empty(warr[i]) and save:
            continue
        else:
            warr_tap.append(sample_process(warr[i],key,type=type,num=num))
            save=False
    if type=='os':
        WARR.append(np.array(warr_tap))
    else:
        WARR.append(np.array(warr_tap))
    return WARR

def if_empty(a,empty=" "):
    for i in range(len(a)):
        if a[i]!=empty:return False
    return True

def sample_process(a,b=1000000,type='ws',num=np.nan):
    '''
        Take a row vector
    '''
    rs=[]
    for i in range(len(a)):
        if type=='os':
            if a[i]==' ':
                rs.append(num)
                continue
        else:
            if a[i]==' ':
                continue
        rs.append(int(a[i])-b)
    return np.array(rs)

def divid_array(array,num):
    ARR=[]
    for i in range(len(array)):
        if num[i]==0:
            ARR.append(array[i])
            continue
        tap=np.split(array[i],num[i],axis=0)
        for j in range(len(tap)):
            arr=[]
            for k in range(len(tap[j])):
                arr.append(list(tap[j][k]))
            ARR.append(np.array(arr))
    return ARR

def mark_os(wrr,w_num,k_filt,fin_part,tip_union=1):
    '''
    The W matrix corresponding to OS is marked so that the calculation process can get the result in the output cycle.
    '''
    flagarr=np.zeros((wrr.shape),dtype=np.float32)

    for i in range(flagarr.shape[0]):
        for j in range(flagarr.shape[1]):
            for k in range(int(k_filt)):
                if i-j==(k+1)*w_num-1:
                    if k == k_filt - 1:
                        fin_part-=1
                        if fin_part<0:
                            break
                    flagarr[i][j]=k+1
                if tip_union>1:
                    for pk in range(1,tip_union):
                        narrow=pk*1000
                        if i - j == (k + 1) * w_num+pk*w_num*k_filt-1:
                            if k == k_filt - 1:
                                fin_part -= 1
                                if fin_part < 0:
                                    break
                            flagarr[i][j] = (k + 1)+narrow


    wrr=np.expand_dims(wrr,axis=0)
    flagarr=np.expand_dims(flagarr,axis=0)
    WRR=np.concatenate((wrr,flagarr),axis=0)
    return WRR

def parallel_alter(MAP):
    m_h=MAP.shape[0]
    m_w=MAP[0].shape[0]
    rs_map=np.full((m_h,m_w),np.nan)
    for i in range(m_h):
        for j in range(m_w):
            try:
                k=MAP[i][j]
                rs_map[i][j]=1
            except:
                continue
    return rs_map




def flip_right_90(arr):
    rs=arr.reshape(arr.size)
    rs=rs[::-1]
    rs=rs.reshape(arr.shape)
    rs=np.transpose(rs)[::-1]
    return rs

def change_to_form(rs,method='defult'):
    if method=='defult':
        map = rs.swapaxes(1, 2)
        RS = map.swapaxes(0, 1).copy()
    else:
        map = rs.swapaxes(0, 1)
        RS = map.swapaxes(1, 2).copy()
    return RS

def ifmap_to_form(rs,method='defult'):
    if method=='defult':
        map = rs.swapaxes(0, 2)
        RS = map.swapaxes(0, 1).copy()
    else:
        map = rs.swapaxes(0, 2)
        RS = map.swapaxes(1, 2).copy()
    return RS


def write_file(data,name='Inter_RS_0.txt'):
    doc=open(name,'w')
    print(data,file=doc)
    doc.close()