import numpy as np
import numba as nb
import random
import os
import time
import math
import sys
import gc
import copy
import part_cul
import value_map
from Config_info import Config,aquire_config
from keras_values import vgg16_model,lenet_model
from tool.writein_csv import write_csv,write_header,write_txt
from Mul_cul import Mulcul

class Autoprocess:
    '''
    experiment mode customization
    '''
    def __init__(self,pe_size,path,name='',times=3000,mod_type='lenet',fin_layer=False, layer_type='cov',
                 cul_type='ws', layer=0,err_flag=True,err_set='random',exp_set='random',GoldRs='',tip=0):
        self.pe_size=pe_size
        self.path=path
        self.name=name
        self.times=times
        self.mod_type=mod_type
        self.fin_layer=fin_layer
        self.layer_type=layer_type
        self.cul_type=cul_type
        self.layer=layer
        self.err_flag=err_flag
        self.err_set=err_set
        self.exp_set=exp_set
        self.avf_all_acc=0
        self.avf_8_acc=0
        self.avf_16_acc=0
        self.avf_clas_acc=0

        self.top1_8_acc = 0
        self.top1_16_acc=0
        self.top1_clas_acc=0

        self.GoldRs=GoldRs####Standard results for comparison
        self.Cul_cyc=''
        self.tip=tip
        self.mid_root=cul_type+'/'+mod_type+'/'+str(layer)+'/'+str(pe_size)+'/'
        self.parallel_wrr=''

        self.bit=0
        self.err_type=0
        self.X=0
        self.Y=0

    ###Three test methods: position fixation, PE fixation, completely random

    def root_manage(self):
        root_temp=self.mid_root[:-1]
        if 'hard' in self.err_set:
            self.mid_root= root_temp+self.err_set+'/'
        elif 'multi' in self.err_set:
            self.mid_root=root_temp+self.err_set+'/'
        else:
            pass

    def get_GoldRs(self,path=''):
        '''
        To get the golden result, and the number of cycles per section
        '''
        file='AVF_record/'+self.mid_root+'GoldRs'+str(self.layer)+'.txt'
        doc=open(file,'w')
        start=time.clock()
        ARR, WRR, bias, tip_add, tip_union, save,k_filt, w_pix_num, num_filt = self.readin_data(path=path)
        read_time=time.clock()
        mulcul = Mulcul(pe_size=self.pe_size, mode=self.mod_type, cul_type=self.cul_type, layer=self.layer,layer_type=self.layer_type,
                        ARR=ARR, WRR=WRR, bias=bias, tip_add=tip_add, tip_union=tip_union, save_array=save,fin_layer=self.fin_layer,
                        err_flag=False, err_set=self.err_set,k_filt=k_filt,w_pix_num=w_pix_num,num_filt=num_filt)
        Rs,CYC = mulcul.complete_process()
        cul_time=time.clock()
        print(CYC)
        self.calculate_allcyc(CYC=CYC,tip_union=tip_union)
        self.Cul_cyc=CYC
        if self.fin_layer:
            self.GoldRs = Rs
            print(Rs, file=doc)
        else:
            self.GoldRs=Rs[0]
            print(Rs[0],file=doc)
        doc.close()
        return Rs[0]


    def calculate_allcyc(self,CYC,tip_union):
        sumall = 0
        if self.cul_type == 'os':
            sumall = CYC[0]
            infer_cyc = sumall + len(CYC)
        elif self.cul_type == 'is' and tip_union == -1:
            sumall = CYC[0][0]
            infer_cyc = sumall + len(CYC) * CYC[0][1]
        else:
            for i in CYC:
                sumall += sum(i)
            if self.cul_type == 'ws':
                infer_cyc = sumall + len(CYC)
            else:
                infer_cyc = sumall
        print('Total Cyc:', sumall, 'Total Cul Cyc:', infer_cyc)


    def readin_data(self,path=''):
        ###Data used only in OS mode
        k_filt, w_pix_num, num_filt='','',''
        if path=='':
            PATH=self.path
        else:
            PATH=path

        if self.cul_type=='ws':
            ARR, WRR, bias, tip_add, tip_union, save = value_map.readin_ws(pe_size=self.pe_size, scale_path=PATH,
                                                                        layer=self.layer,type=self.layer_type,mode=self.mod_type)
        elif self.cul_type=='is':
            ARR, WRR, bias, tip_add, tip_union, save ,parallel_wrr= value_map.readin_is(pe_size=self.pe_size, scale_path=PATH,
                                                                          layer=self.layer,type=self.layer_type,mode=self.mod_type)
            self.parallel_wrr=parallel_wrr
        elif self.cul_type=='os':
            ARR, WRR, bias, tip_add, tip_union, save, k_filt, w_pix_num, num_filt = value_map.readin_os(pe_size=self.pe_size,
                                                                        scale_path=PATH,layer=self.layer,type=self.layer_type,
                                                                                                        mode=self.mod_type)

        else:
            print("Invalid parameter: cul_type.Try the following input:ws,is,os")
            return
        return ARR, WRR, bias, tip_add, tip_union, save,k_filt, w_pix_num, num_filt

    def readin_data_cross(self,ifmap='',path=''):
        ###Data used only in OS mode
        k_filt, w_pix_num, num_filt='','',''
        if self.cul_type=='ws':
            ARR, WRR, bias, tip_add, tip_union, save = value_map.readin_ws_cross(pe_size=self.pe_size, scale_path=path,
                                                        layer=self.layer,type=self.layer_type,mode=self.mod_type,Ifmap=ifmap)
        elif self.cul_type=='is':
            ARR, WRR, bias, tip_add, tip_union, save ,parallel_wrr= value_map.readin_is_cross(pe_size=self.pe_size, scale_path=path,
                                                                          layer=self.layer,type=self.layer_type,mode=self.mod_type,Ifmap=ifmap)
            self.parallel_wrr=parallel_wrr
        elif self.cul_type=='os':
            ARR, WRR, bias, tip_add, tip_union, save, k_filt, w_pix_num, num_filt = value_map.readin_os_cross(pe_size=self.pe_size,
                                                                        scale_path=path,layer=self.layer,type=self.layer_type,
                                                                                                        mode=self.mod_type,Ifmap=ifmap)

        else:
            print("Invalid parameter: cul_type.Try the following input:ws,is,os")
            return
        return ARR, WRR, bias, tip_add, tip_union, save,k_filt, w_pix_num, num_filt



    def record_mid_data(self,Rs,tipp):
        compare_info='AVF_record/'+self.mid_root+'compare_info'+str(self.tip)
        try:
            if isinstance(Rs, int):  #If the injection is invalid, the calculation is skipped
                a1, b1, c1, d1 ,top1_8,top1_16,top1_clas= 0, 0, 0, 0, 0, 0, 0
            else:
                if self.mod_type=='lenet':
                    if self.fin_layer:
                        a1, b1, c1, d1, top1_8,top1_16,top1_clas= compare(self.GoldRs, Rs)
                    else:
                        a1, b1, c1, d1,top1_8,top1_16,top1_clas = compare(self.GoldRs, Rs[0])  #
                elif self.mod_type=='vgg16':
                    if self.fin_layer:
                        a1, b1, c1, d1,top1_8,top1_16,top1_clas= compare_vgg16(self.GoldRs, Rs)
                    else:
                        a1, b1, c1, d1,top1_8,top1_16,top1_clas= compare_vgg16(self.GoldRs, Rs[0])
                elif self.mod_type=='cifar':
                    if self.fin_layer:
                        a1, b1, c1, d1,top1_8,top1_16,top1_clas = compare(self.GoldRs, Rs)
                    else:
                        a1, b1, c1, d1,top1_8,top1_16,top1_clas = compare(self.GoldRs, Rs[0])
                elif self.mod_type=='mobilenet':
                    if self.fin_layer:
                        a1, b1, c1, d1 = compare_vgg16(self.GoldRs, Rs)
                    else:
                        a1, b1, c1, d1 = compare_vgg16(self.GoldRs, Rs[0])
                else:
                    a1, b1, c1, d1 ,top1_8,top1_16,top1_clas= 0, 0, 0, 0, 0, 0, 0
        except:
            print('Unexpected results')
            a1, b1, c1, d1, top1_8, top1_16, top1_clas = 1, 1, 1, 1, 1, 1, 1
        self.avf_all_acc += a1
        self.avf_8_acc += b1
        self.avf_16_acc += c1
        self.avf_clas_acc += d1
        self.top1_8_acc +=top1_8
        self.top1_16_acc +=top1_16
        self.top1_clas_acc+=top1_clas
        print("Result of this classification:",self.avf_all_acc,self.avf_16_acc,self.avf_8_acc,self.avf_clas_acc,self.top1_16_acc,self.top1_8_acc,self.top1_clas_acc)
        record=[tipp,a1,c1,b1,d1]
        write_csv(compare_info,record)

    def get_bottom_info(self,mul_cul):
        return mul_cul.get_partcul_info()

    def auto_process_pe(self):###pe loc err
        tip=1
        type_dataflow=3
        ARR, WRR, bias, tip_add, tip_union, save,k_filt, w_pix_num, num_filt = self.readin_data()
        ######Just read the CSV mapping once
        for i in range(self.pe_size):
            for j in range(self.pe_size):
                for k in range(type_dataflow):
                    start = time.clock()
                    ARRR = ARR.copy()
                    WRRR = WRR.copy()
                    mulcul = Mulcul(pe_size=self.pe_size, mode=self.mod_type, cul_type=self.cul_type, layer=self.layer,
                                    layer_type=self.layer_type,
                                    ARR=ARRR, WRR=WRRR, bias=bias, tip_add=tip_add, tip_union=tip_union,
                                    save_array=save, fin_layer=self.fin_layer,err_flag=self.err_flag,X=i,Y=j)
                    Rs = mulcul.complete_process(name='default_mod')
                    end = time.clock()
                    print('[',str(tip),']Running time of FCV_CUL: %s Seconds' % (end - start), '\n')
                    tip+=1

    def auto_process_bit(self,x=0,y=0):###32bit err
        tip=1
        type_dataflow=3
        ARR, WRR, bias, tip_add, tip_union, save,k_filt, w_pix_num, num_filt=self.readin_data()
        ######Just read the CSV mapping once
        for i in range(32):
            for k in range(type_dataflow):
                start = time.clock()
                ARRR = ARR.copy()
                WRRR = WRR.copy()
                mulcul = Mulcul(pe_size=self.pe_size, mode=self.mod_type, cul_type=self.cul_type, layer=self.layer,
                                layer_type=self.layer_type,
                                ARR=ARRR, WRR=WRRR, bias=bias, tip_add=tip_add, tip_union=tip_union,
                                save_array=save, fin_layer=self.fin_layer, err_flag=self.err_flag, X=x, Y=y,bit=i)
                Rs = mulcul.complete_process(name='default_mod')
                end = time.clock()
                print('[',str(tip),']Running time of FCV_CUL: %s Seconds' % (end - start), '\n')
                tip+=1

    def auto_process_random(self):
        '''work for times tip,and record the avf result'''
        avf = 'AVF_record/'+self.mid_root+'AVF-rate'+str(self.tip)
        err = 'AVF_record/'+self.mid_root+'Fault_info'+str(self.tip)
        rs_info='AVF_record/'+self.mid_root+'compare_info'+str(self.tip)

        write_header(avf=avf, err=err,rs_info=rs_info,type=self.mod_type,exp_type=self.err_set)

        ARR, WRR, bias, tip_add, tip_union, save,k_filt, w_pix_num, num_filt=self.readin_data()

        tipp=1
        ###Because the input and weight are essentially the same every time, we'll just go through the ScaleSim map once
        for i in range(self.times):
            start = time.clock()
            ARRR=copy.deepcopy(ARR)
            WRRR=copy.deepcopy(WRR)
            if self.layer_type=='cov' and self.cul_type=='is':
                save=np.zeros(save.shape)
                save=save.astype(np.float32)

            mulcul=Mulcul(pe_size=self.pe_size,mode=self.mod_type,cul_type=self.cul_type,layer=self.layer,layer_type=self.layer_type,
                          ARR=ARRR, WRR=WRRR, bias=bias,tip_add=tip_add,tip_union=tip_union,save_array=save,fin_layer=self.fin_layer,
                          err_flag=self.err_flag,err_set=self.err_set,Cul_cyc=self.Cul_cyc,tip=self.tip,k_filt=k_filt,w_pix_num=w_pix_num,
                          num_filt=num_filt,times=i,parallel_wrr=self.parallel_wrr,mid_root=self.mid_root)
            Rs,CYC=mulcul.complete_process()
            ###Because each failure injection has the potential to change the value, we only pass a new value before the computation
            del ARRR,WRRR,mulcul
            self.record_mid_data(Rs,i)#Compare the result with the golden Rs

            end = time.clock()
            print('[', str(tipp), ']Running time of AVF_CUL: %s Seconds' % (end - start), '\n')
            #if tip==5or tip==10or tip==30or tip==60 or tip==100 or tip==200or tip==500 or tip==1000 or tip==1500 or tip==2000 or tip==2500 or tip%100==0:
            if tipp%50==0 or tipp==20 or tipp==50:
                record=[tipp,self.avf_all_acc/tipp,self.avf_16_acc/tipp,self.avf_8_acc/tipp,self.avf_clas_acc/tipp,self.top1_16_acc/tipp,
                        self.top1_8_acc/tipp,self.top1_clas_acc/tipp]
                write_csv('AVF_record/'+self.mid_root+'AVF-rate'+str(self.tip),record)
            tipp += 1
        doc = open('AVF_record/'+self.mid_root+'AVF-rate-Final.txt', 'a')
        print('Final result:','avf_all_acc:',self.avf_all_acc/self.times,'avf_16_acc:',self.avf_16_acc/self.times,'avf_8_acc:',self.avf_8_acc/self.times,
              'avf_clas_acc:',self.avf_clas_acc/self.times,file=doc)
        doc.close()

    def auto_cross_layer(self):
        '''Calculate the error rate by specifying PE for execution'''
        lenet_layer=[0,1,5,7]
        cifar_layer=[0,1,4,5,9,11]

        avf = 'AVF_record/' + self.mid_root + 'AVF-rate' + str(self.tip)
        err = 'AVF_record/' + self.mid_root + 'Fault_info' + str(self.tip)
        rs_info = 'AVF_record/' + self.mid_root + 'compare_info' + str(self.tip)
        rs_clas = 'AVF_record/' + self.mid_root + '/classification' + str(self.tip)

        write_header(avf=avf, err=err, rs_info=rs_info, type=self.mod_type, exp_type=self.err_set)

        tipp = 1
        Error_Flag=False
        for ik in range(self.times):
            if ik!=0:
                Error_Flag=self.err_flag
                record_err = self.Err_set() 
                err_cord=[str(ik)]
                err_cord.extend(record_err)
                write_csv(err, err_cord)

            Rs_inter=''
            start = time.clock()
            if self.mod_type=='lenet':
                '''Determine the next layer, layer type in lenet5'''
                for i in range(len(lenet_layer)):
                    self.layer=lenet_layer[i]
                    if i+1==len(lenet_layer):
                        next_layer=lenet_layer[i]
                        self.fin_layer=True
                    else:
                        next_layer=lenet_layer[i+1]
                    if self.layer>=5:
                        self.layer_type='fc'
                    else:
                        self.layer_type = 'cov'

                    ARR, WRR, bias, tip_add, tip_union, save, k_filt, w_pix_num, num_filt = self.readin_data_cross(ifmap=Rs_inter,path=self.path[i])

                    if self.layer_type == 'cov' and self.cul_type == 'is':
                        save = np.zeros(save.shape)
                        save = save.astype(np.float32)

                    mulcul = Mulcul(pe_size=self.pe_size, mode=self.mod_type, cul_type=self.cul_type, layer=self.layer,
                                    layer_type=self.layer_type,
                                    ARR=ARR, WRR=WRR, bias=bias, tip_add=tip_add, tip_union=tip_union, save_array=save,
                                    fin_layer=self.fin_layer,
                                    next_layer=next_layer,
                                    err_flag=Error_Flag, err_set=self.err_set, Cul_cyc=self.Cul_cyc, tip=self.tip,
                                    k_filt=k_filt, w_pix_num=w_pix_num,
                                    num_filt=num_filt, times=0, parallel_wrr=self.parallel_wrr, mid_root=self.mid_root,
                                    bit=self.bit,err_type=self.err_type,X=self.X,Y=self.Y,
                                    sup_set_flag=True)
                    Rs_inter, CYC = mulcul.cross_layer_process()
                    ###Because each failure injection has the potential to change the value, we only pass a new value before the computation
                    del ARR, WRR, bias, tip_add, tip_union, save, k_filt, w_pix_num, num_filt,mulcul
                    print('layer',self.layer,'compeleted!')
            elif self.mod_type=='cifar':
                '''Determine the next layer, layer type in cifar10'''
                for i in range(len(cifar_layer)):
                    self.layer = cifar_layer[i]
                    if i + 1 == len(cifar_layer):
                        next_layer = cifar_layer[i]
                        self.fin_layer = True
                    else:
                        next_layer = cifar_layer[i + 1]
                    if self.layer >= 9:
                        self.layer_type = 'fc'
                    else:
                        self.layer_type= 'cov'

                    ARR, WRR, bias, tip_add, tip_union, save, k_filt, w_pix_num, num_filt = self.readin_data_cross(
                        ifmap=Rs_inter,
                        path=self.path[i])

                    if self.layer_type == 'cov' and self.cul_type == 'is':
                        save = np.zeros(save.shape)
                        save = save.astype(np.float32)

                    mulcul = Mulcul(pe_size=self.pe_size, mode=self.mod_type, cul_type=self.cul_type, layer=self.layer,
                                    layer_type=self.layer_type,
                                    ARR=ARR, WRR=WRR, bias=bias, tip_add=tip_add, tip_union=tip_union, save_array=save,
                                    fin_layer=self.fin_layer,
                                    next_layer=next_layer,
                                    err_flag=Error_Flag, err_set=self.err_set, Cul_cyc=self.Cul_cyc, tip=self.tip,
                                    k_filt=k_filt, w_pix_num=w_pix_num,
                                    num_filt=num_filt, times=0, parallel_wrr=self.parallel_wrr, mid_root=self.mid_root,
                                    bit=self.bit, err_type=self.err_type, X=self.X, Y=self.Y,
                                    sup_set_flag=True)
                    Rs_inter, CYC = mulcul.cross_layer_process()
                    ###Because each failure injection has the potential to change the value, we only pass a new value before the computation
                    del ARR, WRR, bias, tip_add, tip_union, save, k_filt, w_pix_num, num_filt,mulcul
                    print('layer', self.layer, 'compeleted!')
            else:
                print('Functionality needs to be extended')
            end = time.clock()

            record_clas=[]
            if ik==0:
                record_clas.append('gloden')
                record_clas.extend(Rs_inter)
                self.GoldRs=Rs_inter
            else:
                record_clas.append(ik)
                record_clas.extend(Rs_inter)
            write_csv(rs_clas, record_clas)  #Record the classification results
            if ik!=0:
                self.record_mid_data(Rs_inter, ik)
            print('[',ik, ']Running time of PE_FI: %s Seconds' % (end - start), '\n')
            print(Rs_inter)#Show the classification results

            self.fin_layer = False

            if tipp % 50 == 0 or tipp == 20 or tipp == 50:
                record = [tipp, self.avf_all_acc / tipp, self.avf_16_acc / tipp, self.avf_8_acc / tipp,
                          self.avf_clas_acc / tipp, self.top1_16_acc / tipp,
                          self.top1_8_acc / tipp, self.top1_clas_acc / tipp]
                write_csv('AVF_record/' + self.mid_root + 'AVF-rate' + str(self.tip), record)
            tipp += 1
        doc = open('AVF_record/' + self.mid_root + 'AVF-rate-Final.txt', 'a')
        print('Final result:', 'avf_all_acc:', self.avf_all_acc / self.times, 'avf_16_acc:',
              self.avf_16_acc / self.times, 'avf_8_acc:', self.avf_8_acc / self.times,
              'avf_clas_acc:', self.avf_clas_acc / self.times, file=doc)
        doc.close()


    def FI_start(self,csv_exist=False,golden_path=''):
        '''Interface to user profiles'''
        #Generate scalesim-csv files that need to be read
        self.root_manage()
        root_temp=self.mid_root[:-1]
        path='AVF_record/'+root_temp
        isExists = os.path.exists(path)
        if not isExists:
            os.makedirs(path)
        if not csv_exist:
            value_map.create_scale_csv(pe_size=self.pe_size,layer=self.layer,type=self.layer_type,mode=self.mod_type)

        self.get_GoldRs(path=golden_path)
        if self.exp_set == 'pe':
            self.auto_process_pe()
        if self.exp_set == 'bit':
            self.auto_process_bit(x=0,y=0)
        if self.exp_set == 'random':
            self.auto_process_random()
        if self.exp_set == 'cross':
            self.auto_cross_layer()

    def sample_start(self):
        self.root_manage()
        root_temp = self.mid_root[:-1]
        path = 'AVF_record/' + root_temp
        isExists = os.path.exists(path)
        if not isExists:
            os.makedirs(path)

        self.auto_cross_layer()

    def Err_set(self):
        self.bit = random.randint(0, 31)

        self.err_type = random.randint(0, 2)

        self.X = random.randint(0, self.pe_size - 1)  # pe_X

        self.Y = random.randint(0, self.pe_size - 1)

        if self.err_type==0:err_loc='W'
        elif self.err_type==1:err_loc='S'
        else:err_loc='I'
        record = [self.err_set, err_loc, self.X, self.Y, self.bit]

        return record



def compare(a,b):
    '''

    :param a: Correct output values, list
    :param b: The result of the current fault injection cycle
    :return: avf rate
    Mark 0 accurately, add 1 inaccurately
    '''
    a1=0#Complete data accuracy
    b1=0#The first eight digits are accurate
    c1=0#The first 16 digits are accurate
    d1=0#Accurate classification
    top1_16=0#top1, 16-bit accuracy
    top1_8=0#top1, 8-bit accuracy
    top1_clas=0#top1, accurate classification

    c_k8=1e-8
    c_k16=1e-16

    top1_judge_clas=True
    if a.index(max(a)) != b.index(max(b)):
        top1_clas+=1
        top1_16 += 1
        top1_8 += 1
        top1_judge_clas=False
    if math.fabs(max(a)-max(b))>c_k16 and top1_judge_clas:
        top1_16+=1
    if math.fabs(max(a)-max(b))>c_k8 and top1_judge_clas:
        top1_8+=1

    for i in range(len(a)):
        if a[i]!=b[i]:
            a1+=1
            break
    for i in range(len(a)):
        if math.fabs(a[i]-b[i])>c_k8:
            b1+=1
            break
    for i in range(len(a)):
        if math.fabs(a[i]-b[i])>c_k16:
            c1+=1
            break

    compare_a=a.copy()
    for i in range(5):
        point1=compare_a.index(max(compare_a))
        point2=b.index(max(b))
        if point1!=point2:
            d1+=1
            break
        del compare_a[point1],b[point2]
    return a1,b1,c1,d1,top1_8,top1_16,top1_clas

def compare_vgg16(a,b):
    '''

    :param a: Correct output values, list
    :param b: The result of the current fault injection cycle
    :return: avf rate
    '''
    a1 = 0  # Complete data accuracy
    b1 = 0  # The first eight digits are accurate
    c1 = 0  # The first 16 digits are accurate
    d1 = 0  # Accurate classification
    c_k=1e-4
    c_k16=1e-6

    top1_16 = 0
    top1_8 = 0
    top1_clas = 0

    top1_judge_clas = True
    if a[0][0] != b[0][0]:
        top1_clas += 1
        top1_16 += 1
        top1_8 += 1
        top1_judge_clas = False
    if math.fabs(a[0][2] - b[0][2]) > c_k16 and top1_judge_clas:
        top1_16 += 1
    if math.fabs(a[0][2] - b[0][2]) > c_k and top1_judge_clas:
        top1_8 +=1

    for i in range(len(a)):
        if a[i][0]!=b[i][0]:
            d1+=1
            break
    if d1==1:
        return 1,1,1,1,top1_8,top1_16,top1_clas

    for i in range(len(a)):
        if a[i][2]!=b[i][2]:
            a1+=1
            break

    for i in range(len(a)):
        if math.fabs(a[i][2]-b[i][2])>c_k:
            b1+=1
            break

    for i in range(len(a)):
        if math.fabs(a[i][2]-b[i][2])>c_k16:
            c1+=1
            break
    return a1,b1,c1,d1,top1_8,top1_16,top1_clas

def proc_newauto(tip=0,times=10000):
    expr_config = aquire_config()
    mod_type, mod_path, data, pe_size, times, layer, layer_type, data_flow, err_set, expr_set, fin_layer, \
    err_flag, scale_path ,csv_exist= expr_config.get_config()

    auto=Autoprocess(pe_size, mod_type=mod_type,path=scale_path, cul_type=data_flow, layer=layer,layer_type=layer_type,
                    fin_layer=fin_layer,err_flag=err_flag,exp_set='random',err_set=err_set,times=times,tip=tip)
    auto.FI_start(csv_exist=csv_exist)

def start_here(group=10,times=10000):
    for i in range(group):
        try:
            proc_newauto(i,times)
        except:
            print('Error Start')

def target_pe():
    lenet_layer = [0, 1, 5, 7]
    cifar_layer = [0, 1, 4, 5, 9, 11]
    mode_layer=''

    path_root='scale_sim/RS/'
    file_name='sram_read0pe256.csv'
    path=[]
    expr_config = aquire_config()
    mod_type, mod_path, data, pe_size, times, layer, layer_type, data_flow, err_set, expr_set, fin_layer, \
    err_flag, scale_path, csv_exist = expr_config.get_config()

    path_mid=data_flow+'/'+mod_type+'/'

    '''Automatically generate the corresponding CSV file name for each layer'''
    if mod_type=='lenet':
        mode_layer=lenet_layer
    elif mod_type=='cifar':
        mode_layer=cifar_layer
    else:
        print('Functionality needs to be extended')
    for i in mode_layer:
        file_tip=list(file_name)
        file_tip[9]=str(i)
        file_tip=''.join(file_tip)
        file=path_root+path_mid+file_tip
        path.append(file)

    print (path)

    tar_pe = Autoprocess(pe_size, mod_type=mod_type, path=path, cul_type=data_flow, layer=layer,
                       layer_type=layer_type,
                       fin_layer=fin_layer, err_flag=err_flag, exp_set='cross', err_set=err_set, times=times)
    tar_pe.sample_start()


if __name__=='__main__':
    start = time.clock()
    proc_newauto()
    # target_pe()##cross layer
    end = time.clock()
    print('Running time of FCV_CUL: %s Seconds' % (end - start), '\n')

