import numpy as np
import numba as nb
import random
import time
import math
import sys
import gc
import copy
import part_cul
import value_map
from Config_info import Config,aquire_config
from keras_values import vgg16_model,lenet_model,cifar_model,mobilenet_model,lenet_mid,cifar_mid
from tool.writein_csv import write_csv,write_header,write_txt

#Err_sign=False
Err_sign=True

class Err_Info:
    def __init__(self,section,fault_cyc,X,Y,bit,err_type,cyc_inall,valid_flag,alter_flag):
        self.section = section
        self.fault_cyc = fault_cyc
        self.X = X
        self.Y = Y
        self.bit = bit
        self.err_type = err_type  # 0,w; 1,sum; 2,input
        self.cyc_inall=cyc_inall
        self.valid_flag=valid_flag
        self.alter_flag=alter_flag

    def get_info(self):
        return self.section,self.fault_cyc,self.X ,self.Y ,self.bit ,self.err_type ,self.cyc_inall,self.valid_flag,self.alter_flag

    def recheck(self):
        #Checks whether an identical record exists
        pass

    def get_bylist(self):
        if self.err_type == 0:
            Err_type='W'
        elif self.err_type==1:
            Err_type='S'
        else:
            Err_type='I'
        list_info=[int(self.fault_cyc),Err_type,self.X ,self.Y ,self.bit ,self.cyc_inall]
        return list_info



class Mulcul:
    '''
    This class is used to perform all three types of computation, interaction with Scalesim, fault customization
    '''
    def __init__(self,pe_size=32,mode='lenet',cul_type='ws',layer=0,layer_type='cov',ARR='', WRR='', bias='',
                 tip_add=0, tip_union=0, save_array='',cyc_all=50,fin_layer=False,
                 err_flag=False,err_set='random',section=0,fault_cyc=0,X=0,Y=0,bit=0,err_type=0,Cul_cyc='',tip=0,
                 k_filt=0,w_pix_num=0,num_filt=0,times=0,parallel_wrr='',mid_root='',next_layer='',sup_set_flag=False):
        self.pe_size=pe_size
        self.mode=mode
        self.cul_type=cul_type
        self.layer=layer
        self.layer_type=layer_type
        self.fin_layer=fin_layer
        self.err_flag=err_flag
        self.err_set=err_set
        self.cyc_all=cyc_all
        self.ARR=ARR
        self.WRR=WRR
        self.bias=bias
        self.tip_add=tip_add
        self.tip_union=tip_union
        self.save_array=save_array
        self.section=section
        self.fault_cyc=fault_cyc
        self.X=X
        self.Y=Y
        self.bit=bit
        self.err_type=err_type#0,w; 1,sum; 2,input
        self.Cul_cyc=Cul_cyc
        self.tip=tip#####all file times
        self.k_filt=k_filt
        self.w_pix_num=w_pix_num
        self.num_filt=num_filt
        self.times=times####times in all experiments
        self.change0_1=''
        self.mid_root=mid_root
        self.err_info='Invalid cycle'
        self.parallel_wrr=parallel_wrr

        self.multi_err_List=[]####multiple error messages
        self.Err_count=0
        self.Bit_List=[]
        self.next_layer=next_layer

        self.sup_set_flag=sup_set_flag

    #Main calculation functions, input parameters: output size, number of cores, input matrix, weight, bias, full cycle, partial cycle, PE size
    #Processing batch computation

    def err_cyc_set(self):
        '''By running the standard results once, the actual computation cycle for each section is obtained,
                    which can be used to customize fault injection cycle'''
        valid_flag=False
        alter_flag=False#Used to determine whether data substitution is taking place in the current period
        try:
            record=self.ARR.shape[0]
        except:
            record=len(self.ARR)

        cyc_part=0
        cyc_alter=0

        if self.err_set=='random_ws':
            self.section = random.randint(0, record - 1)  

            cyc_part=self.Cul_cyc[int(self.section)][0]
            cyc_alter=self.Cul_cyc[int(self.section)][1]
            cyc_inall = self.calculate_allcyc(int(self.section))

            # Setting error type:0,w; 1,sum; 2,input
            self.err_type=random.randint(0,2)


            self.bit=random.randint(0, 31)
            self.X = random.randint(0, self.pe_size-1)  
            self.Y = random.randint(0, self.pe_size-1)

            self.fault_cyc = random.randint(1, cyc_part + cyc_alter + 1)

            if self.layer_type=='fc':
                pass

            if self.fault_cyc>cyc_alter and self.fault_cyc!=cyc_part+cyc_alter+1:
                valid_flag=True
            elif self.fault_cyc<=cyc_alter:
                if self.err_type==0:
                    valid_flag = True
                    alter_flag=True
                    print('Fault injection: In data replacement cycle')
                print('Invalid injection during data replacement cycle')
            else:
                #The last write back period
                pass

        elif self.err_set=='random_is':
            self.bit = random.randint(0, 31)
            self.X = random.randint(0, self.pe_size-1)  # pe_X
            self.Y = random.randint(0, self.pe_size-1)

            self.section = random.randint(0, record - 1)
            cyc_part = self.Cul_cyc[int(self.section)][0]
            cyc_alter = self.Cul_cyc[int(self.section)][1]
            cyc_inall = self.calculate_allcyc(int(self.section))

            if self.layer_type == 'cov' and self.tip_union ==-1:
                section = self.X // self.WRR[0].shape[0]
                if section < record:
                    self.section = section
                    cyc_inall = 0
                    cyc_part = self.Cul_cyc[int(self.section)][0]
                    cyc_alter = self.parallel_wrr.shape[0]
                    print('Valid')
                else:
                    valid_flag = False
                    self.err_infor = 'Unused PE'
                    return '',valid_flag,''

            # Setting error type:0,w; 1,sum; 2,input
            self.err_type = random.randint(0, 2)
            self.fault_cyc = random.randint(1, cyc_part + cyc_alter + 1)
            self.fault_cyc = random.randint(108, cyc_part + cyc_alter + 1)

            if self.fault_cyc > cyc_alter and self.fault_cyc != cyc_part + cyc_alter + 1:
                valid_flag = True

            elif self.fault_cyc <= cyc_alter :
                if self.err_type == 0:
                    valid_flag = True
                    alter_flag = True
                    print('Fault injection: In data replacement cycle')
                else:
                    valid_flag =False
                    print('Invalid injection during data replacement cycle')
                    return '', valid_flag, ''
                if self.tip_union==-1 and not self.judge_parallel(alter_cyc=self.fault_cyc, x=self.X, y=self.Y):
                    valid_flag = False
                    return '', valid_flag, ''
            else:
                pass

            if self.layer_type=='fc':
                if self.Y!=0:
                    valid_flag=False

        elif self.err_set=='random_os':
            cyc_part=self.Cul_cyc[0]
            # Setting error type:0,w; 1,sum; 2,input
            self.err_type=random.randint(0,2)

            self.bit = random.randint(0, 31)
            self.X = random.randint(0, self.pe_size-1)  # pe_X
            self.Y = random.randint(0, self.pe_size-1)
            self.fault_cyc = random.randint(1, cyc_part)
            cyc_inall=0
            valid_flag=True

            if self.layer_type == 'fc':
                if self.X != 0: 
                    valid_flag=False


        elif self.err_set=='fix':
            pass

        elif self.err_set=='test':
            self.fault_cyc=random.randint(1,9)
            self.section=0
            valid_flag=True
            cyc_inall=0
            self.err_type=0
            alter_flag = True

        else:
            pass

        cyc = self.fault_cyc
        print('cyc,cyc_inall,cyc_alter,cyc_part,:::err_type',cyc,cyc_inall,cyc_alter,cyc_part,self.err_type)

        if not (alter_flag):
            self.fault_cyc -= cyc_alter
            cyc_inall+=cyc_alter

        print('Fault injection period %d, total period %d,fault_cyc:%d'%(cyc,cyc_inall+cyc,self.fault_cyc))
        return cyc_inall,valid_flag,alter_flag

    def err_multibit_set(self):
        Err_set=self.err_set

        if self.cul_type == 'ws':
            self.err_set = 'random_ws'
        elif self.cul_type == 'is':
            self.err_set = 'random_is'
        else:
            self.err_set = 'random_os'
        Err_count=random.randint(2,6)

        cyc_inall, valid_flag, alter_flag = self.err_cyc_set()
        for i in range(Err_count):
            bit=random.randint(0, 31)
            if bit in self.Bit_List:
                i-=1
            else:
                self.Bit_List.append(bit)
        self.err_set=Err_set
        self.Err_count=Err_count
        return cyc_inall, valid_flag, alter_flag

    def err_multierr_set(self):
        Err_set=self.err_set

        if self.cul_type == 'ws':
            self.err_set='random_ws'
        elif self.cul_type == 'is':
            self.err_set = 'random_is'
        else:
            self.err_set = 'random_os'

        Err_count=random.randint(2,6)
        for i in range(Err_count):
            cyc_inall, valid_flag, alter_flag = self.err_cyc_set()
            info=Err_Info(self.section,self.fault_cyc,self.X ,self.Y ,self.bit ,self.err_type ,cyc_inall,valid_flag,alter_flag)
            self.multi_err_List.append(info)
        self.sort_err()

        self.err_set=Err_set
        self.Err_count=Err_count

    def calculate_allcyc(self,num_k):
        '''Count all cycles up to k modules'''
        if num_k == 0:
            return 0
        sum_cyc = 0
        count = 0
        for i in self.Cul_cyc:
            sum_cyc += sum(i)
            count += 1
            if count >= num_k:
                break
        return sum_cyc

    def err_hard_set(self):
        self.bit=random.randint(0, 31)

        self.err_type = random.randint(0, 2)

        self.X = random.randint(0, self.pe_size - 1)  # pe_X

        self.Y = random.randint(0, self.pe_size - 1)

        ###Record the configuration information in a file

        if self.err_type==0:err_loc='W'
        elif self.err_type==1:err_loc='S'
        else:err_loc='I'

        record = [self.err_set, err_loc, self.X, self.Y, self.bit]
        t = part_cul.Covcul(valid_flag=False, mid_root=self.mid_root, times=self.times)
        t.err_inj(INFO=record)  ###Record fault injection information
        return True

    def err_hard_target(self):
        self.bit = 30
        # Setting error type:0,w; 1,sum; 2,input
        self.err_type = 0

        self.X = 150
        self.Y = 100
        return True

    def sort_err(self):
        self.multi_err_List.sort(key=lambda x: (x.section,x.fault_cyc,x.X,x.Y), reverse=False)

    def switch_err(self):
        '''Determine the error type and select the fault injection mode'''
        if self.err_set=='random_ws'or self.err_set=='random_is' or self.err_set=='random_os':
            ERR_class='bit_err'
        elif self.err_set=='hard_fix0'or self.err_set=='hard_fix1':
            ERR_class='hard_err'
        elif self.err_set=='target_hard':
            ERR_class='hard_t'
        elif self.err_set=='multi_err':
            ERR_class='multi_err'
        else:
            ERR_class='multi_bit'
        return ERR_class

    def mul_cul(self):
        cyc_inall, valid_flag, alter_flag='','',''
        if self.err_flag:
            if self.switch_err()=='bit_err':
                cyc_inall,valid_flag,alter_flag=self.err_cyc_set()
            elif self.switch_err()=='hard_err':
                if not self.sup_set_flag:
                    valid_flag=self.err_hard_set()
                else:
                    valid_flag=True
            elif self.switch_err()=='hard_t':
                valid_flag=self.err_hard_target()
            elif self.switch_err() == 'multi_err':
                self.err_multierr_set()
            else:
                cyc_inall, valid_flag, alter_flag=self.err_multibit_set()

        if self.cul_type=='ws':
            RS,CYC=self.mul_cul_ws(cyc_inall=cyc_inall,valid_flag=valid_flag,alter_flag=alter_flag)
        elif self.cul_type=='is':
            if self.layer_type=='cov':
                RS,CYC=self.mul_cul_is(cyc_inall=cyc_inall,valid_flag=valid_flag,alter_flag=alter_flag)
            elif self.layer_type=='fc':
                RS,CYC=self.mul_cul_is_fc(cyc_inall=cyc_inall,valid_flag=valid_flag,alter_flag=alter_flag)
            else:
                print('Invalid layer calculation mode, please try 1.cov 2.fc')
                RS,CYC='',''
        elif self.cul_type=='os':
            RS,CYC=self.mul_cul_os(cyc_inall=cyc_inall,valid_flag=valid_flag,alter_flag=alter_flag)
        else:
            print('Invalid dataflow mode, please try 1.ws 2.is 3.os')
            RS,CYC='',''
        return RS,CYC

    def mul_cul_ws(self,cyc_inall=0,valid_flag=True,alter_flag=False):
        if valid_flag==False and self.multi_err_List==[]:
            t=part_cul.Covcul(valid_flag=False,mid_root=self.mid_root,times=self.times)
            t.err_inj(info=self.err_info)###Record fault injection information
            return -1,''###Skip calculation

        ERR_class=self.switch_err()
        flag=self.fin_layer
        file = "test/inter_out.txt"

        Cul_cyc=np.zeros((len(self.ARR),2))
        RS = []
        RRS=[]
        first_record=True

        ##Verify wether the fault injection is valid in advance
        if self.err_flag and ERR_class!='hard_err' and self.multi_err_List==[]:
            if self.section>3:
                t = part_cul.Covcul(array_w=self.WRR[self.section], array_i=self.ARR[self.section], file=file, bias=self.bias,
                                   cul_layer=self.layer_type,err_flag=self.err_flag, fault_cyc=self.fault_cyc, fault_inall=cyc_inall,
                                   alter_flag=alter_flag,errtype=self.err_type,test_valid=True,tip=self.tip,bitflip=self.bit,
                                   coord_x=self.X,coord_y=self.Y,times=self.times,mid_root=self.mid_root)
                T, CYC = t.acc_cul()
                if isinstance(T, int):
                    return -1, ''

        if self.multi_err_List!=[]:
            t = part_cul.Covcul(valid_flag=False, mid_root=self.mid_root, times=self.times,Multi_err_list=self.multi_err_List)
            num=1
            List_err=[]
            for i in self.multi_err_List:
                info=i.get_bylist()
                List_err.append('err'+str(num))
                List_err+=info
                num+=1
            t.err_inj_mul(info=List_err) 
        for i in range(len(self.ARR)):
            if ERR_class=='bit_err' or ERR_class=='multi_bit':
                if i==self.section:
                    Bit=self.bit
                    if self.Bit_List!=[]:
                        Bit=self.Bit_List
                    t = part_cul.Covcul(array_w=self.WRR[i], array_i=self.ARR[i], file=file, bias=self.bias, cul_layer=self.layer_type,
                                       err_flag=self.err_flag,fault_cyc=self.fault_cyc,fault_inall=cyc_inall,alter_flag=alter_flag,
                                        errtype=self.err_type,tip=self.tip,bitflip=Bit,coord_x=self.X,coord_y=self.Y,times=self.times,
                                        mid_root=self.mid_root,pe_size=self.pe_size)
                    T, CYC = t.acc_cul()
                    self.store_partcul_info(t)

                else:
                    t = part_cul.Covcul(array_w=self.WRR[i], array_i=self.ARR[i], file=file, bias=self.bias,cul_layer=self.layer_type,
                                       tip=self.tip,times=self.times,mid_root=self.mid_root,pe_size=self.pe_size)
                    T,CYC=t.acc_cul()
            elif ERR_class=='hard_err'or ERR_class=='hard_t':
                if self.err_set=='hard_fix0':
                    hard_fix=0
                else:hard_fix=1
                t = part_cul.Covcul(array_w=self.WRR[i], array_i=self.ARR[i], file=file, bias=self.bias,
                                    cul_layer=self.layer_type,err_flag=self.err_flag,pe_size=self.pe_size,
                                    errtype=self.err_type, tip=self.tip,bitflip=self.bit, coord_x=self.X,
                                    coord_y=self.Y, times=self.times,mid_root=self.mid_root,hard_fix=hard_fix)
                T, CYC = t.acc_cul()
            elif ERR_class=='multi_err' and self.multi_err_List!=[]:
                Listerr = []
                for j in self.multi_err_List:
                    if j.section==i:
                        Listerr.append(j)
                t = part_cul.Covcul(array_w=self.WRR[i], array_i=self.ARR[i], file=file, bias=self.bias,cul_layer=self.layer_type,cul_type='ws',
                                    err_flag=True,errtype=self.err_type,Multi_err_list=Listerr,
                                    tip=self.tip, times=self.times, mid_root=self.mid_root, pe_size=self.pe_size)
                T, CYC = t.acc_cul_mulerr()
            else:
                t = part_cul.Covcul(array_w=self.WRR[i], array_i=self.ARR[i], file=file, bias=self.bias,
                                    cul_layer=self.layer_type,
                                    tip=self.tip, times=self.times, mid_root=self.mid_root, pe_size=self.pe_size)
                T, CYC = t.acc_cul()
            # If the error injected into the replacement data cycle does not take effect, the computation is omitted
            if isinstance(T, int):
                return -1,''

            key_add=i%self.tip_add

            if self.tip_union>1:
                if (first_record):
                    RS = T
                    first_record = False
                elif (key_add%self.tip_add!=0 and key_add!=0):
                    RS+=T
                else:
                    RRS.append(RS)
                    del RS
                    RS = np.zeros(T.shape, dtype=np.float32)
                    RS += T
                    self.tip_union -= 1
            else:
                if (first_record):
                    RS = T
                    first_record = False
                elif (i == len(self.ARR) - 1):
                    RS += T
                    RRS.append(RS)
                else:
                    RS+=T


            Cul_cyc[i][0]=CYC[0] #Computation cycle
            Cul_cyc[i][1] = CYC[1] #Data replacement cycle

        if RRS==[]:
            RRS.append(RS)
        result=RRS[0]
        result=result.astype(np.float32)
        for j in range(len(RRS)-1):
            result=np.concatenate((result,RRS[j+1]),axis=0)
        if flag:
            result=softmax_process(alter_fc_form(result),self.bias)
        else:
            result= activation_process(result, self.bias)

        return result,Cul_cyc

    def mul_cul_is(self,cyc_inall=0,valid_flag=True,alter_flag=False):
        if valid_flag==False and self.multi_err_List==[]:
            t=part_cul.Covcul(valid_flag=False,mid_root=self.mid_root,times=self.times)
            t.err_inj(info=self.err_info)###Record fault injection information
            return -1,''###Skip calculation

        ERR_class = self.switch_err()

        #######internal result
        flag=self.fin_layer
        file = "test/inter_out.txt"

        Cul_cyc = np.zeros((len(self.ARR), 2))
        his_sum=0
        parallel_flag=False
        if self.tip_union==-1:
            parallel_flag=True

        if self.err_flag and ERR_class!='hard_err' and self.multi_err_List==[]:
            if self.section>3:
                t = part_cul.Covcul(array_w=self.WRR[self.section], array_i=self.ARR[self.section], file=file, bias=self.bias,
                                   cul_layer='cov',cul_type='is',err_flag=True, fault_cyc=self.fault_cyc, fault_inall=cyc_inall,
                                   alter_flag=alter_flag,errtype=self.err_type,test_valid=True,tip=self.tip,bitflip=self.bit,
                                   coord_x=self.X,coord_y=self.Y,times=self.times,mid_root=self.mid_root,parallel_flag=parallel_flag)
                T, CYC = t.acc_cul()
                if isinstance(T, int):
                    return -1, ''

        tip_add=self.tip_add
        tip_union=self.tip_union

        if self.multi_err_List != []:
            t = part_cul.Covcul(valid_flag=False, mid_root=self.mid_root, times=self.times,
                                Multi_err_list=self.multi_err_List)
            num = 1
            List_err = []
            for i in self.multi_err_List:
                info = i.get_bylist()
                List_err.append('err' + str(num))
                List_err += info
                num += 1
            t.err_inj_mul(info=List_err)  ###Record fault injection information

        for i in range(len(self.ARR)):
            if ERR_class == 'bit_err'or ERR_class=='multi_bit':
                if i==self.section:
                    Bit = self.bit
                    if self.Bit_List != []:
                        Bit = self.Bit_List
                    t = part_cul.Covcul(array_w=self.WRR[i], array_i=self.ARR[i], file=file, bias=self.bias,cul_type='is' ,cul_layer='cov',
                                       err_flag=self.err_flag,fault_cyc=self.fault_cyc,fault_inall=cyc_inall,alter_flag=alter_flag,
                                        errtype=self.err_type,tip=self.tip,bitflip=Bit,coord_x=self.X,coord_y=self.Y,times=self.times,
                                        mid_root=self.mid_root,parallel_flag=parallel_flag)
                    T, CYC = t.acc_cul()
                    self.store_partcul_info(t)
                else:
                    t = part_cul.Covcul(array_w=self.WRR[i], array_i=self.ARR[i], file=file, bias=self.bias,cul_type='is',
                                       cul_layer=self.layer_type,times=self.times,mid_root=self.mid_root)
                    T,CYC=t.acc_cul()

            elif ERR_class == 'hard_err':
                if self.err_set == 'hard_fix0':
                    hard_fix = 0
                else:
                    hard_fix = 1
                t = part_cul.Covcul(array_w=self.WRR[i], array_i=self.ARR[i], file=file, bias=self.bias,cul_type='is' ,
                                    cul_layer=self.layer_type, err_flag=self.err_flag, pe_size=self.pe_size,
                                    errtype=self.err_type, tip=self.tip, coord_x=self.X, bitflip=self.bit,coord_y=self.Y,
                                    times=self.times, mid_root=self.mid_root,parallel_flag=parallel_flag, hard_fix=hard_fix)
                #coord_x=self.X, bitflip=self.bit,coord_y=self.Y
                T, CYC = t.acc_cul()

            elif ERR_class=='multi_err' and self.multi_err_List!=[]:
                Listerr = []
                for j in self.multi_err_List:
                    if j.section==i:
                        Listerr.append(j)
                t = part_cul.Covcul(array_w=self.WRR[i], array_i=self.ARR[i], file=file, bias=self.bias,cul_layer='cov',cul_type='is',
                                    err_flag=True,errtype=self.err_type,Multi_err_list=Listerr,
                                    tip=self.tip, times=self.times, mid_root=self.mid_root, pe_size=self.pe_size)
                T, CYC = t.acc_cul_mulerr()

            else:
                t = part_cul.Covcul(array_w=self.WRR[i], array_i=self.ARR[i], file=file, bias=self.bias, cul_type='is',
                                    cul_layer=self.layer_type, times=self.times, mid_root=self.mid_root)
                T, CYC = t.acc_cul()

            if isinstance(T, int):
                return -1,''


            T_process=T.swapaxes(0,2)
            if tip_union>1:
                if tip_add>1:
                    his_sum = save_Is(T_process, self.save_array, his_sum,flag='add')
                    tip_add-=1
                else:
                    his_sum = save_Is(T_process, self.save_array, his_sum,flag='union')
                    tip_add=self.tip_add
                    tip_union-=1
            else:
                if tip_add>1:
                    his_sum = save_Is(T_process, self.save_array, his_sum,flag='add')
                    tip_add-=1
                else:
                    his_sum = save_Is(T_process, self.save_array, his_sum,flag='union')

            Cul_cyc[i][0] = CYC[0]  # Computation cycle
            Cul_cyc[i][1] = CYC[1]  # Data replacement cycle


        result=self.save_array
        if flag:
            result=softmax_process(alter_fc_form(result),self.bias)
        else:
            result= relu_process(result, self.bias)
        return result,Cul_cyc

    def mul_cul_is_fc(self,cyc_inall=0,valid_flag=True,alter_flag=False):
        #Check whether the fault injection is effective
        if valid_flag==False:
            t=part_cul.Covcul(valid_flag=False,mid_root=self.mid_root,times=self.times)
            t.err_inj(info=self.err_info)###Record fault injection information
            return -1,''

        ERR_class = self.switch_err()

        #######The intermediate results
        flag = self.fin_layer
        file = "test/inter_out.txt"
        doc = open(file, 'w')
        tap_array = []

        Cul_cyc = np.zeros((len(self.ARR), 2))
        first_record = True

        #Test whether the x,y coordinates in fault injection are valid
        if self.err_flag and ERR_class!='hard_err':
            if self.section>3:
                t = part_cul.Covcul(array_w=self.WRR[self.section], array_i=self.ARR[self.section], file=file, bias=self.bias,
                                   cul_layer='fc',cul_type='is',err_flag=True, fault_cyc=self.fault_cyc, fault_inall=cyc_inall,
                                   alter_flag=alter_flag,errtype=self.err_type,test_valid=True,tip=self.tip,bitflip=self.bit,
                                   coord_x=self.X,coord_y=self.Y,times=self.times,mid_root=self.mid_root)
                T, CYC = t.acc_cul()
                if isinstance(T, int):
                    return -1, ''

        for i in range(len(self.ARR)):
            if ERR_class == 'bit_err'or ERR_class=='multi_bit':
                if i==self.section:
                    Bit = self.bit
                    if self.Bit_List != []:
                        Bit = self.Bit_List
                    t = part_cul.Covcul(array_w=self.WRR[i], array_i=self.ARR[i], file=file, bias=self.bias,cul_type='is' ,cul_layer='fc',
                                       err_flag=self.err_flag,fault_cyc=self.fault_cyc,fault_inall=cyc_inall,alter_flag=alter_flag,
                                        errtype=self.err_type,tip=self.tip,bitflip=Bit,coord_x=self.X,coord_y=self.Y,times=self.times,
                                        mid_root=self.mid_root)
                    T, CYC = t.acc_cul()
                    self.store_partcul_info(t)

                else:
                    t = part_cul.Covcul(array_w=self.WRR[i], array_i=self.ARR[i], file=file, bias=self.bias, cul_type='is', cul_layer='fc',
                                       tip=self.tip,times=self.times,mid_root=self.mid_root)
                    T,CYC = t.acc_cul()
            elif ERR_class == 'hard_err':
                if self.err_set == 'hard_fix0':
                    hard_fix = 0
                else:
                    hard_fix = 1
                t = part_cul.Covcul(array_w=self.WRR[i], array_i=self.ARR[i], file=file, bias=self.bias,cul_type='is' ,
                                    cul_layer='fc', err_flag=self.err_flag, pe_size=self.pe_size,
                                    errtype=self.err_type, tip=self.tip, bitflip=self.bit, coord_x=self.X,
                                    coord_y=self.Y, times=self.times, mid_root=self.mid_root, hard_fix=hard_fix)
                T, CYC = t.acc_cul()
            else:
                pass

            if isinstance(T, int):
                return -1,''

            T_process = T.swapaxes(0, 2)
            if (first_record):
                tap_array = T_process
                first_record = False
            else:
                tap_array += T_process
            Cul_cyc[i][0] = CYC[0]  # Computation cycle
            Cul_cyc[i][1] = CYC[1]  # Data replacement cycle

        result = tap_array

        if flag:
            result = softmax_process(alter_fc_form(result), self.bias)
        else:
            result = relu_process(result, self.bias)
        print(result, file=doc)
        doc.close()
        return result,Cul_cyc

    def mul_cul_os(self,cyc_inall=0,valid_flag=True,alter_flag=False):
        if valid_flag==False and self.multi_err_List==[]:
            t=part_cul.Covcul(valid_flag=False,mid_root=self.mid_root,times=self.times)
            t.err_inj(info=self.err_info)
            return -1,''

        ERR_class = self.switch_err()
        flag = self.fin_layer
        file = "test/inter_out.txt"
        doc = open(file, 'w')

        if self.layer_type == 'fc':
            self.k_filt = 1

        if self.layer_type == 'cov' and self.multi_err_List == [] and not self.sup_set_flag:
            if self.Y>=self.num_filt:
                t = part_cul.Covcul(valid_flag=False, mid_root=self.mid_root, times=self.times)
                t.err_inj(info='Unused PE')  
                return -1, ''  

        if self.multi_err_List != []:#Record the fault information in the csv file
            t = part_cul.Covcul(valid_flag=False, mid_root=self.mid_root, times=self.times,
                                Multi_err_list=self.multi_err_List)
            num = 1
            List_err = []
            for i in self.multi_err_List:
                info = i.get_bylist()
                List_err.append('err' + str(num))
                List_err += info
                num += 1
            t.err_inj_mul(info=List_err) 

        if ERR_class == 'bit_err' or ERR_class == 'multi_bit':
            Bit = self.bit
            if self.Bit_List != []:
                Bit = self.Bit_List
            t = part_cul.Covcul(array_w=self.WRR[0], array_i=self.ARR[0], bias=self.bias, cul_layer=self.layer_type, err_flag=self.err_flag,
                                os_w_filt=self.k_filt, os_pix_num=self.w_pix_num,num_filt=self.num_filt,cul_type='os',
                                array_o=self.save_array, pe_size=self.pe_size,fault_cyc=self.fault_cyc,fault_inall=cyc_inall,
                                alter_flag=alter_flag, errtype=self.err_type, tip=self.tip,bitflip=Bit,coord_x=self.X,
                                coord_y=self.Y,times=self.times,mid_root=self.mid_root,os_union=self.tip_union)
            rs, cyc = t.os_cul()
        elif ERR_class == 'hard_err':
            if self.err_set == 'hard_fix0':
                hard_fix = 0
            else:
                hard_fix = 1
            t = part_cul.Covcul(array_w=self.WRR[0], array_i=self.ARR[0], bias=self.bias, cul_layer=self.layer_type, err_flag=self.err_flag,
                                os_w_filt=self.k_filt, os_pix_num=self.w_pix_num,num_filt=self.num_filt,cul_type='os',
                                array_o=self.save_array, pe_size=self.pe_size,fault_cyc=self.fault_cyc,fault_inall=cyc_inall,
                                alter_flag=alter_flag, errtype=self.err_type, tip=self.tip,bitflip=self.bit,coord_x=self.X,
                                coord_y=self.Y,times=self.times,mid_root=self.mid_root,os_union=self.tip_union,file=file,hard_fix=hard_fix,
                                normal_return_flag= self.sup_set_flag)
            rs, cyc = t.os_cul()
        elif ERR_class == 'multi_err' and self.multi_err_List != []:
            t = part_cul.Covcul(array_w=self.WRR[0], array_i=self.ARR[0], bias=self.bias, cul_layer=self.layer_type,err_flag=True,
                                os_w_filt=self.k_filt, os_pix_num=self.w_pix_num, num_filt=self.num_filt, cul_type='os',
                                array_o=self.save_array, pe_size=self.pe_size, fault_cyc=self.fault_cyc,fault_inall=cyc_inall,
                                alter_flag=alter_flag, errtype=self.err_type, tip=self.tip, bitflip=self.bit,coord_x=self.X,
                                coord_y=self.Y, times=self.times, mid_root=self.mid_root, os_union=self.tip_union,
                                file=file,Multi_err_list=self.multi_err_List)
            rs, cyc = t.os_cul_mulerr()
        else:
            t = part_cul.Covcul(array_w=self.WRR[0], array_i=self.ARR[0], file=file, bias=self.bias,cul_layer=self.layer_type,
                                os_w_filt=self.k_filt, os_pix_num=self.w_pix_num, num_filt=self.num_filt, cul_type='os',array_o=self.save_array,
                                tip=self.tip, times=self.times, mid_root=self.mid_root, pe_size=self.pe_size)
            rs, cyc = t.os_cul()

        self.store_partcul_info(t)

        if isinstance(rs, int):
            return -1, ''

        print(rs.shape)
        result = rs.reshape(self.save_array.shape)
        if flag:
            result = softmax_process(result, self.bias)
        else:
            result = relu_process(result, self.bias)

        print(result, file=doc)
        doc.close()
        return result, cyc

    def complete_process(self):
        '''send the calculation result of the current layer to Keras for subsequent calculation and then get the final classification result'''
        file = 'AVF_record/'+self.mid_root+'/classification'+str(self.tip)
        inter_rs,CYC = self.mul_cul()

        if isinstance(inter_rs, int):
            write_csv(file, ['Skip Calculation'])
            return -1,''
        if self.fin_layer:
            if self.mode=='vgg16':
                inter_rs=fin_rs_process(inter_rs)
            write_csv(file, inter_rs)
            return inter_rs,CYC
        errin_img = inter_rs.swapaxes(0, 1)
        errin_img = errin_img.swapaxes(1, 2).copy()
        errin_img = np.expand_dims(errin_img, axis=0)

        if self.mode == 'vgg16':
            RS = vgg16_model(errin_img,layer=self.layer) 
        elif self.mode == 'lenet':
            RS = lenet_model(errin_img, layer=self.layer)
        elif self.mode == 'cifar':
            RS = cifar_model(errin_img,layer=self.layer)
        elif self.mode == 'mobilenet':
            RS = mobilenet_model(errin_img,layer=self.layer)
        else:
            RS=''
            print('The model is not currently supported,Try:vgg16,lenet,cifar,mobilenet ')
        del errin_img 
        if self.Cul_cyc=='':
            info = ['layer'+str(self.layer)+' GoldenRs'+str(self.pe_size)+'PE']
        else:
            info=['Running experiments']
        write_csv(file, info+RS[0])

        gc.collect()
        return RS,CYC

    def cross_layer_process(self):
        inter_rs, CYC = self.mul_cul()

        if isinstance(inter_rs, int):
            return -1, ''
        if self.fin_layer:
            if self.mode == 'vgg16':
                inter_rs = fin_rs_process(inter_rs)
            return inter_rs, CYC
        errin_img = inter_rs.swapaxes(0, 1)
        errin_img = errin_img.swapaxes(1, 2).copy()
        errin_img = np.expand_dims(errin_img, axis=0)

        if self.mode == 'lenet':
            RS = lenet_mid(errin_img, layer=self.layer,next_layer=self.next_layer)
        elif self.mode == 'cifar':
            RS = cifar_mid(errin_img, layer=self.layer,next_layer=self.next_layer)
        else:
            RS=''
            print('The model is not currently supported,Try:lenet,cifar')
        del errin_img 

        gc.collect()
        return RS, CYC

    def store_partcul_info(self,partcul):
        self.change0_1=partcul.change0_1

    def get_partcul_info(self):
        return self.change0_1

    def judge_parallel(self,alter_cyc,x,y):
        if self.parallel_wrr=='':
            return False
        if x-alter_cyc+1>0:
            return False
        if np.isnan(self.parallel_wrr[x][y]):
            return False
        real_x = x + self.parallel_wrr.shape[0] - alter_cyc
        self.X=real_x
        return True

    def chack_value(self, w_k=1, i_k=7):
        rs_sum = 0
        list_single_rs=[]
        list_cumulative_rs=[]
        for i in range(len(self.ARR)//2):
            w_item=self.WRR[i][:, w_k]
            in_item=self.ARR[i][:, i_k]
            mul_value=sum(w_item * in_item)
            list_single_rs.append(mul_value)
            rs_sum += mul_value
            list_cumulative_rs.append(rs_sum)

        print(rs_sum)
        print('Result：',list_single_rs)
        print('Cumulative result：',list_cumulative_rs)
        time.sleep(1000)

    def find_max(self,arr):
        listmax=[]
        for i in range(len(arr)):
            listmax.append(np.max(arr[i]))
        print(listmax)
        return listmax



def save_Is(a,sum,his_sum,flag='add'):
    num_filt=a.shape[0]
    line=a.shape[2]
    o_s=sum.shape[1]
    y=(his_sum)%o_s
    x=math.floor(his_sum/o_s)

    for m in range(num_filt):
        i = x
        j = y
        for n in range(line):
            ###because we set 'a' as a[n][1][line]
            sum[m][i][j]+=a[m][0][n]
            j+=1
            if (j>=o_s):
                j=0
                i+=1
    if flag=='union':
        his_sum+=line
    return his_sum

def fin_rs_process(list_rs):
    num=[]
    for i in range(1000):
        num.append(str(i))
    rs=dict(zip(num,list_rs))
    Rs=sorted(rs.items(), key=lambda asd: asd[1], reverse=True)
    tip_rs = []
    for i in range(5):
        tip_rs.append([Rs[i][0],Rs[i][0],round(Rs[i][1],9)])
    return tip_rs

def relu_process(rs,bias):
    ###bias
    for i in range(len(bias)):
        rs[i]=rs[i]+bias[i]
        rs[i][rs[i]<0]=0
    ###relu
    return rs

def softmax_process(rs,bias):
    try:
        for i in range(len(bias)):
            rs[i]=rs[i]+bias[i]
        inter_v=[math.exp(i)for i in rs]
        Sum=sum(inter_v)
        RS=[i/Sum for i in inter_v]
    except:
        print('Unexpected error')
        print(rs)
        RS=rs.tolist()[0]
    return RS

def activation_process(rs, bias):
 ###bias
 for i in range(len(bias)):
  rs[i] = rs[i] + bias[i]
  W=rs[i].shape[0]
  H=rs[i].shape[1]
  for j in range(W):
    for k in range(H):
      if rs[i][j][k]<0:
        # value=0.01*rs[i][j][k]####Leaky ReLU
        value = (math.exp(rs[i][j][k]))-1#### ELU
        rs[i][j][k]=value
 ###relu
 return rs

def alter_fc_form(inter_rs):
    inter_rs=inter_rs.squeeze()
    I_rs=inter_rs.squeeze()
    return I_rs

def alter_cov_form(inter_rs):
    errin_img = inter_rs.swapaxes(0, 1)
    errin_img = errin_img.swapaxes(1, 2).copy()
    errin_img = np.expand_dims(errin_img, axis=0)
    return errin_img


def test_new_ws():
    #########lenet#########
    # pe_size = 32
    # type = 'fc'
    # layer = 7
    # path = 'scale_sim/RS/WS/sram_read7.csv'
    # file='fc_RS.txt'
    #########vgg16#########
    pe_size = 64
    type = 'cov'
    layer = 1
    path = 'scale_sim/RS/WS/vgg16/sram_read1.csv'
    file = 'vgg16_RS.txt'
    mode='vgg16'
    fin_flag=False

    ARR, WRR, bias, tip_add, tip_union, save = value_map.readin_ws(pe_size=pe_size, scale_path=path, layer=layer,type=type,mode=mode)

    mulcul = Mulcul(pe_size=pe_size, mode=mode, cul_type='ws', layer=layer,layer_type=type,
                    ARR=ARR, WRR=WRR, bias=bias, tip_add=tip_add, tip_union=tip_union, save_array=save,
                    fin_layer=fin_flag,err_flag=False)
    rs,cyc=mulcul.complete_process()



if __name__=='__main__':
    start = time.clock()
    test_new_ws()
    end = time.clock()
    print('Running time of FCV_CUL: %s Seconds' % (end - start), '\n')
