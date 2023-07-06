import numpy as np
import numba as nb
#import cupy as cp
import random
import time
import math
import gc
import value_map
#from util_fuc import inclear,getName
from tool.bitflip import bitflip2,bitflip1
from tool.writein_csv import write_csv,write_txt
np.set_printoptions(threshold=np.inf)


class Covcul:
    '''
    Simulate WS dataflow in several steps:
    1. Element moves right to reach
    2. Multiply with the original intermediate sum
    3. Add to Partial Sum
    4. Move Results Up
    In order to unify the input flow of the scale function, it is necessary to invert the calculation matrix w of the input.
    Transform the input i-matrix during initialization.
    '''

    def __init__(self, array_w=np.zeros((18, 3)), array_i=np.zeros((18, 9)), array_o='',strides=1,bias=np.zeros((3,)),
                 file="zinter_out.txt",fault_cyc=0,err_flag=False,fault_inall=25,entire_cul_cyc=20,coord_x='',
                 coord_y='',bitflip='',errtype='',pe_size=32,cul_layer='cov',cul_type='ws',valid_flag=True,
                 alter_flag=False,test_valid=False,tip=0,os_w_filt=3,os_pix_num=9,num_filt=64,times=0,mid_root='',
                 parallel_flag=False,os_union=1,hard_fix='',Multi_err_list=[],normal_return_flag=False):
        self.array_w = array_w
        self.array_i = array_i
        self.bias = bias
        self.w_h = self.array_w.shape[0]
        self.w_w = self.array_w.shape[1]
        if cul_type=='os':
            self.w_h = self.array_w.shape[1]
            self.w_w = self.array_w.shape[2]
        self.i_h = self.array_i.shape[0]
        self.i_w = self.array_i.shape[1]
        self.num_filt = self.w_w  # umber of filters
        #The dimensional information of the matrix
        if cul_type=='ws':
            self.array_o = np.zeros((self.w_w, int(self.i_w**0.5), int(self.i_w**0.5)),dtype=np.float32)
        elif cul_type=='is':
            self.array_o=np.zeros((self.w_w,1,self.i_w),dtype=np.float32)#notice:because of the deffierence of shape of input and weight array
        else:###os
            self.num_filt = num_filt
            self.array_o=np.full((self.num_filt,array_o.shape[1]*array_o.shape[2],1),np.nan)#32,676,1
        self.o_h = self.array_o.shape[1]
        self.o_w = self.array_o.shape[2]
        if cul_type=='os':
            self.array_sum=np.zeros((self.i_h,self.w_w),dtype=np.float32)
        else:
            self.array_sum = np.zeros((self.w_h, self.w_w),dtype=np.float32)  # Save intermediate results
        # Record the index array stored in the output result, whether updated, the number of stored values, x, y coordinates [tf, n, x, y]
        self.idxrc = np.zeros((self.num_filt, 4))
        self.idxrow = np.zeros((self.w_w,))  # Vector for whether line writing occurs

        #Used to store the output of the OS
        self.os_w_filt=os_w_filt
        self.os_pix_num = os_pix_num
        self.idxrc_os=np.full((2,self.num_filt,int(self.os_w_filt)),np.nan)#22

        self.file=file
        #Used to store the output of the OS
        self.fault_cyc=fault_cyc
        self.err_flag=err_flag#Whether to perform error injection
        self.fault_inall=fault_inall#Which of the total cycles is
        self.entire_cul_cyc=entire_cul_cyc
        self.coord_x=coord_x
        self.coord_y=coord_y
        self.bitflip=bitflip
        self.errtype=errtype#0,w; 1,sum; 2,input
        self.pe_size=pe_size
        #others
        self.cul_layer=cul_layer
        self.cul_type=cul_type
        self.valid_flag=valid_flag
        self.alter_flag=alter_flag
        self.test_valid=test_valid
        self.tip=tip
        self.pe_x=0
        ###label
        self.times=times
        self.change0_1=''
        self.mid_root=mid_root
        self.parallel_flag=parallel_flag
        #Record whether the is, WS mode generates the process of adding nan when extracting matrix
        self.wsisfill_tag=''
        #Record whether the OS needs to be divided because the channel is larger than the PE
        self.os_union=os_union
        ######hard fault
        self.hard_fix=hard_fix
        self.ace_bit=0
        ######multi fault
        self.Multi_err_list=Multi_err_list

        self.normal_return_flag=normal_return_flag

    def acc_cul(self):
        '''
        Calculate from the bottom right corner. The bottom right corner of i is the first input to arrive, and the top left corner of the i matrix
        is the last calculation to be made.
        :return:
        array_o
        '''
        doc=open(self.file,'w')
        ## Step height -1
        np_k = -1
        ERR_clas='bit_err'
        if self.hard_fix!='' and self.err_flag:
            ERR_clas='hard_err'
            if self.errtype==0:
                vaild=self.err_hard_inj(x=self.coord_x,y=self.coord_y,wt=self.array_w,clas='w')
                if not vaild: self.err_flag=False

        while (self.idxrc[self.num_filt - 1][1] < self.o_h * self.o_w):
            # If the data on the last channel is not fully filled, the loop does not end
            np_k += 1
            '''The matrix is intercepted periodially and then calculated'''
            par_arr_i=self.get_par_arr(np_k)

            if (np_k+1==self.fault_cyc or self.alter_flag==True)and ERR_clas=='bit_err':
                if self.err_flag:
                    par_arr_i=self.err_inj(cycles=np_k+1,par_ifmap=par_arr_i,X=self.coord_x,Y=self.coord_y,
                                 f_bit=self.bitflip,err_type_set=self.errtype)
                    self.alter_flag=False
                    #If the error injection of this time does not take effect, the calculation procedure is omitted
                    if isinstance(par_arr_i, int):
                        return -1,''

            if ERR_clas=='hard_err' and self.errtype==1 and self.err_flag:
                vaild=self.err_hard_inj(x=self.coord_x, y=self.coord_y, psum=self.array_sum,clas='psum')
                if not vaild: self.err_flag=False

            '''PSUM passes the data at the beginning of the period'''
            if (np_k + 1 > self.w_h):
                self.sto_res(row=self.array_sum[0], cycles=np_k)
            self.array_sum = np.insert(self.array_sum, self.i_h, values=0, axis=0)  # Insert a row at the end
            self.array_sum = np.delete(self.array_sum, 0, 0)  # Delete first line from header
            if (self.idxrc[self.num_filt - 1][1] >= self.o_h * self.o_w):
                # In the last period, only the number is passed, not calculated
                break


            par_arr_i=nan_process(par_arr_i[0])

            if ERR_clas=='hard_err' and self.errtype==2 and self.err_flag:
                vaild=self.err_hard_inj(x=self.coord_x, y=self.coord_y, input=par_arr_i,clas='i')
                if not vaild: self.err_flag=False

            mul_arr_i = par_arr_i * self.array_w  #The product of the current calculation

            self.array_sum = mul_arr_i + self.array_sum  #Add to the previous partial sum

            self.array_sum=self.array_sum.astype(np.float32)

            del mul_arr_i,par_arr_i       

        cycles = np_k + 1  #Calculate all cycles
        doc.close()
        #The section's period is returned as a list
        CYC=[cycles,self.w_h]
        return self.array_o,CYC

    def acc_cul_mulerr(self):
        '''
        Calculate from the bottom right corner. The bottom right corner of i is the first input to arrive, and the top left corner of the i matrix
        is the last calculation to be made.
        :return:
        array_o
        '''
        doc=open(self.file,'w')
        np_k = -1
        ###Determine the type of experiment
        exceed_edge = False
        fault_setting=''
        fault_no = 0
        over=False

        while (self.idxrc[self.num_filt - 1][1] < self.o_h * self.o_w):
            # If the data on the last channel is not fully filled, the loop does not end
            np_k += 1

            # Inject faults starting from the first one
            if not exceed_edge and self.Multi_err_list!=[]:
                fault_setting = self.Multi_err_list[fault_no]

            '''The matrix is intercepted periodially and then calculated'''
            par_arr_i=self.get_par_arr(np_k)

            if fault_setting!='' and (np_k+1==fault_setting.fault_cyc or fault_setting.alter_flag==True):
                if self.err_flag and not over:
                    while(self.Multi_err_list[fault_no].fault_cyc==np_k+1):
                        par_arr_i=self.err_inj_mul(par_ifmap=par_arr_i,X=fault_setting.X,Y=fault_setting.Y,
                                 f_bit=fault_setting.bit,err_type_set=fault_setting.err_type)
                        fault_no+=1
                        if fault_no>=len(self.Multi_err_list):
                            exceed_edge=True
                            over=True
                            break

            '''PSUM passes the data at the beginning of the period'''
            if (np_k + 1 > self.w_h):
                self.sto_res(row=self.array_sum[0], cycles=np_k)
            self.array_sum = np.insert(self.array_sum, self.i_h, values=0, axis=0)
            self.array_sum = np.delete(self.array_sum, 0, 0)
            if (self.idxrc[self.num_filt - 1][1] >= self.o_h * self.o_w):
                # In the last period, only the number is passed, not calculated
                break

            par_arr_i=nan_process(par_arr_i[0])

            mul_arr_i = par_arr_i * self.array_w
            self.array_sum = mul_arr_i + self.array_sum
            self.array_sum=self.array_sum.astype(np.float32)
            del mul_arr_i,par_arr_i

        cycles = np_k + 1
        doc.close()

        CYC=[cycles,self.w_h]
        return self.array_o,CYC

    def os_cul(self):
        '''
        Because the computing flow of OS mode is fixed psum matrix, which is different from WS and IS,
        this function is implemented independently
        '''
        doc = open(self.file, 'w')
        np_k = -1
        #Preprocessing of data
        self.os_pretreat(com_size=self.i_h)
        all_times=self.o_h * self.o_w
        check_point=self.num_filt - 1

        ERR_clas = 'bit_err'
        if self.hard_fix != '' and self.err_flag:
            ERR_clas = 'hard_err'

        while (self.idxrc[check_point][1] < all_times):
            # If the data on the last channel is not fully filled, the loop does not end
            np_k += 1
            '''The matrix is intercepted periodially and then calculated'''
            par_arr_i, par_arr_w = get_arr_wrr(np_k,array_i=self.array_i,array_w=self.array_w,w_w=self.w_w)

            if np_k + 1 == self.fault_cyc:
                if self.err_flag:
                    par_arr_i,par_arr_w = self.err_inj_os(cycles=np_k + 1, par_ifmap=par_arr_i,par_wrr=par_arr_w,
                                                          X=self.coord_x,Y=self.coord_y,f_bit=self.bitflip,
                                                          err_type_set=self.errtype)
                    # If the error injection of this time does not take effect, the calculation procedure is omitted
                    if isinstance(par_arr_i, int)or isinstance(par_arr_w, int) :
                        return -1, ''


            #par_arr_w[1] = nan_process(par_arr_w[1])

            if ERR_clas == 'hard_err' and self.errtype == 1 and self.err_flag:
                vaild = self.err_hard_inj(x=self.coord_x, y=self.coord_y, psum=self.array_sum, clas='psum_os')
                if not vaild: self.err_flag = False

            '''PSUM passes the data at the beginning of the period'''
            if self.cul_layer=='fc':
                if (self.ju_os_out_fc(par_wrr=par_arr_w,par_irr=par_arr_i)):
                    self.sto_res_os_fc()
            else:
                if (ju_os_out(par_wrr=par_arr_w,idxrc=self.idxrc,idxrc_os=self.idxrc_os,array_sum=self.array_sum,os_union=self.os_union)):
                    self.sto_res_os()

            if (self.idxrc[self.num_filt - 1][1] >= self.o_h * self.o_w):
                # In the last period, only the number is passed, not calculated
                break

            if par_arr_i.shape[2]<self.w_w:#32
                break
            par_arr_i = nan_process(par_arr_i)
            par_arr_w[0] = nan_process(par_arr_w[0])

            if ERR_clas == 'hard_err' and self.err_flag and (self.errtype == 0 or self.errtype == 2):
                if self.errtype==0:
                    vaild=self.err_hard_inj(x=self.coord_x,y=self.coord_y,wt=par_arr_w[0],clas='w')
                else:
                    vaild=self.err_hard_inj(x=self.coord_x, y=self.coord_y, input=par_arr_i[0],clas='i')
                if not vaild: self.err_flag = False

            mul_arr_i = par_arr_i[0] * par_arr_w[0]
            self.array_sum = mul_arr_i + self.array_sum
            self.array_sum = self.array_sum.astype(np.float32)

        cycles = np_k + 1
        doc.close()
        # The section's period is returned as a list
        CYC = [cycles, self.w_h]
        self.array_o=nan_process(self.array_o)
        return self.array_o, CYC

    def os_cul_mulerr(self):
        '''
        Multiple error injection mode for OS dataflow
        '''
        doc = open(self.file, 'w')
        np_k = -1
        #Preprocessing of data
        self.os_pretreat(com_size=self.i_h)
        all_times=self.o_h * self.o_w
        check_point=self.num_filt - 1

        exceed_edge=False
        fault_setting= ''
        fault_no = 0

        while (self.idxrc[check_point][1] < all_times):
            # If the data on the last channel is not fully filled, the loop does not end
            np_k += 1

            if not exceed_edge and self.Multi_err_list != []:
                fault_setting = self.Multi_err_list[fault_no]

            '''The matrix is intercepted periodially and then calculated'''
            par_arr_i, par_arr_w = get_arr_wrr(np_k,array_i=self.array_i,array_w=self.array_w,w_w=self.w_w)

            if fault_setting!='' and np_k + 1 == fault_setting.fault_cyc:
                if self.err_flag:
                    while (self.Multi_err_list[fault_no].fault_cyc == np_k + 1):
                        par_arr_i,par_arr_w = self.err_inj_os_mul(par_ifmap=par_arr_i,par_wrr=par_arr_w,
                                                              X=fault_setting.X,Y=fault_setting.Y,f_bit=fault_setting.bit,
                                                              err_type_set=fault_setting.err_type)
                        fault_no += 1
                        if fault_no >= len(self.Multi_err_list):
                            exceed_edge = True
                            break

            '''PSUM passes the data at the beginning of the period'''
            if self.cul_layer=='fc':
                if (self.ju_os_out_fc(par_wrr=par_arr_w,par_irr=par_arr_i)):
                    self.sto_res_os_fc()
            else:
                if (ju_os_out(par_wrr=par_arr_w,idxrc=self.idxrc,idxrc_os=self.idxrc_os,array_sum=self.array_sum,os_union=self.os_union)):
                    self.sto_res_os()

            if (self.idxrc[self.num_filt - 1][1] >= self.o_h * self.o_w):
                # In the last period, only the number is passed, not calculated
                break

            if par_arr_i.shape[2]<self.w_w:#32
                break
            par_arr_i = nan_process(par_arr_i)
            par_arr_w[0] = nan_process(par_arr_w[0])

            mul_arr_i = par_arr_i[0] * par_arr_w[0]
            self.array_sum = mul_arr_i + self.array_sum
            self.array_sum = self.array_sum.astype(np.float32)

        cycles = np_k + 1
        doc.close()
        # The section's period is returned as a list
        CYC = [cycles, self.w_h]
        self.array_o=nan_process(self.array_o)
        return self.array_o, CYC

    def get_par_arr(self,np_k):
        '''
        Obtain the input matrix of the current period to participate in the calculation
        '''
        #The purpose of setting PI and pj is to get the index of the element in the actual matrix
        if (self.cul_type=='ws'):
            if (self.cul_layer=='cov'):
                if self.w_w<=self.i_w:
                    par_arr_i = acc_map(cycles=np_k + 1, w_h=self.w_h, w_w=self.w_w, i_h=self.i_h, i_w=self.i_w,
                                    ifmap=self.array_i)
                else:
                    self.wsisfill_tag=self.w_w-self.i_w
                    ifmap_tap = np.full((self.w_h, self.w_w - self.i_w), np.nan, dtype=np.float32)
                    ifmap_tap = np.c_[ifmap_tap, self.array_i]
                    par_arr_i = acc_map(cycles=np_k + 1, w_h=self.w_h, w_w=self.w_w, i_h=self.w_h, i_w=self.w_w,
                                        ifmap=ifmap_tap)

            else:##'fc'
                ifmap_tap = np.full((self.w_h, self.w_w - 1), np.nan,dtype=np.float32)
                ifmap_tap = np.c_[ifmap_tap, self.array_i]
                par_arr_i = acc_map(cycles=np_k + 1, w_h=self.w_h, w_w=self.w_w, i_h=self.w_h, i_w=self.w_w,
                                    ifmap=ifmap_tap)


        elif (self.cul_type == 'is'):
            if (self.cul_layer=='cov' and self.w_w>self.i_w):
                self.wsisfill_tag = self.w_w - self.i_w
                ifmap_tap = np.full((self.w_h, self.w_w - self.i_w), np.nan,dtype=np.float32)
                ifmap_tap = np.c_[ifmap_tap, self.array_i]
                par_arr_i = acc_map(cycles=np_k + 1, w_h=self.w_h, w_w=self.w_w, i_h=self.w_h, i_w=self.w_w,
                                    ifmap=ifmap_tap)

            else:
                par_arr_i = acc_map(cycles=np_k + 1, w_h=self.w_h, w_w=self.w_w, i_h=self.i_h, i_w=self.i_w,
                                    ifmap=self.array_i)


        return par_arr_i

    def os_pretreat(self,com_size):
        '''Preprocessing to facilitate matrix truncation'''
        tap_arr_w = np.full((com_size , com_size), 0)
        tap_arr_w1 = np.full((com_size+1 , com_size), 0)
        tap_arr_w2 = np.full((com_size-1, com_size), 0)
        tap_arr_i = np.full((com_size, com_size ), 0)
        self.array_i=np.concatenate((tap_arr_i,self.array_i,tap_arr_i),axis=1)
        array_w1=np.concatenate((tap_arr_w,self.array_w[0],tap_arr_w),axis=0)
        array_w2= np.concatenate((tap_arr_w1, self.array_w[1], tap_arr_w2), axis=0)
        wrr1 = np.expand_dims(array_w1, axis=0)
        wrr2 = np.expand_dims(array_w2, axis=0)
        WRR = np.concatenate((wrr1, wrr2), axis=0)
        self.array_w=WRR
        self.idxrc_os[1]=0

    def sto_res(self, row, cycles):
        '''

        :param row: Rows to be deleted
        :param arr: Store in the final output position
        :return:
        '''
        self.ju_modf_tfe(cycles=cycles)
        for i in range(0, self.num_filt):
            if (self.idxrc[i][0] == 1):
                x = int(self.idxrc[i][2])
                y = int(self.idxrc[i][3])
                self.array_o[i][x][y]=row[i]

        for i2 in range(self.idxrc.shape[0]):
            if (self.idxrc[i2][0] == 1):
                self.idxrc[i2][1] += 1 
                self.idxrc[i2][3] += 1  
                self.idxrc[i2][0] = 0  
                if (self.idxrc[i2][1] >= self.o_h * self.o_w):
                    self.idxrc[i2][0] = -1  
                if (self.idxrc[i2][3] > self.o_w - 1):  
                    self.idxrc[i2][2] += 1
                    self.idxrc[i2][3] = 0

    def ju_modf_tfe(self, cycles):
        '''
        :param cycles: Use the current number of cycles to determine whether the top deleted row needs to be written back to the true/false/eof
        :return:Write records to object idxrow
        '''
        if cycles == self.w_h:  
            self.idxrow[0] = 1
        elif cycles > self.w_h:
            for k in range(min(cycles - self.w_h + 1, self.w_w)): 
                self.idxrow[k] = 1 
        else:
            return
        for p in range(self.num_filt):
            if (self.idxrow[p] == 1 and self.idxrc[p][0] != -1): 
                self.idxrc[p][0] = 1

    def ju_os_out_fc(self,par_wrr,par_irr):
        '''
        This step passes the value to be updated into the idxrc_os array,
        placing the actual index that needs to be filled in to the final output position in the idxrc array
        '''
        flag=False
        flag_fc=False
        if np.max(par_wrr[1])==0:
            return flag
        else:
            flag=True
        if self.num_filt>self.pe_size:
            flag_fc=True
        for j in range(par_wrr[1].shape[1]):

            for i in range(par_wrr[1].shape[0]):
                output_sign=par_wrr[1][i][j]
                if output_sign!=0:
                    J=j
                    Y = int(par_wrr[1][i][j] - 1)
                    if self.idxrc[j][0]==-1:
                        if flag_fc and (not np.isnan(par_irr[0][i][j])):
                            J=(par_wrr[1][i][j]-1)*self.pe_size+j
                            J = int(J)
                            Y=0
                        else:
                            continue
                    try:
                        if self.idxrc[J][0] == -1: continue
                    except:
                        print('ERROR!')
                    self.idxrc_os[0][J][Y]=self.array_sum[i][j]
                    self.array_sum[i][j]=0
                    self.idxrc[J][0]=1

        return flag

    def sto_res_os(self):
        for i in range(self.idxrc_os[0].shape[0]):
            if (self.idxrc[i][0]==-1):
                continue
            for j in range(self.idxrc_os[0].shape[1]):
                if (self.idxrc[i][0] == 1):
                    if not(np.isnan(self.idxrc_os[0][i][j])):
                        x = int(self.idxrc_os[1][i][j]+j*self.pe_size)
                        y=0
                        if x>=self.array_o.shape[1]:
                            continue
                        self.array_o[i][x][y]=self.idxrc_os[0][i][j]

                        self.idxrc[i][1]+=1
                        self.idxrc_os[1][i][j] += 1
                        if (self.idxrc[i][1] >= self.o_h * self.o_w):
                            self.idxrc[i][0] = -1  

        self.idxrc_os[0]=np.full(self.idxrc_os[0].shape,np.nan)

    def sto_res_os_fc(self):
        for i in range(self.idxrc_os[0].shape[0]):
            if (self.idxrc[i][0]==-1):
                continue
            for j in range(self.idxrc_os[0].shape[1]):
                if (self.idxrc[i][0] == 1):
                    if not(np.isnan(self.idxrc_os[0][i][j])):

                        self.array_o[i][0][0]=self.idxrc_os[0][i][j]
                        self.idxrc[i][1]+=1
                        self.idxrc_os[1][i][j] += 1
                        if (self.idxrc[i][1] >= self.o_h * self.o_w):
                            self.idxrc[i][0] = -1 

        self.idxrc_os[0]=np.full(self.idxrc_os[0].shape,np.nan)


    def err_inj(self,cycles=0,par_ifmap='',err_type_set='',X='',Y='',f_bit='',info='Invalid cycle',INFO=[]):
        '''

        :param cycles: Current cycle
        :param fault_cyc: The target period of injection, which can also be selected as a random period
        :param par_ifmap: The ifmap matrix captured this time
        :return:
        '''
        file = "AVF_record/"+self.mid_root+"Fault_Info"+str(self.tip)
        flagg=0
        if not(self.valid_flag):
            times=[self.times]
            if INFO!=[]:
                write_csv(file_name=file, data=times+INFO)
            else:
                write_csv(file_name=file, data=times+[info])
            return -1

        print('Fault injection is being performed')
        pos = random.randint(0, 31)
        pe_x = random.randint(0, self.pe_size)  # row
        y = random.randint(0, self.pe_size)  # column
        Err_type_set = random.randint(0, 2)  #Setting error type:0,w;1,sum;2,input

        if (err_type_set!=''):
            Err_type_set=err_type_set
            flagg+=1
        if (X!=''):
            pe_x=X
            y=Y
            flagg+=1
        if (f_bit!=''):
            pos=f_bit
            flagg+=1
        if (flagg==3):
            print("The ifmap matrix captured this time")
        else:
            print("Problem with fault parameters")

        self.coord_x = pe_x
        self.coord_y = y
        self.bitflip = pos
        alter_print_flag=False

        if pe_x >= self.w_h and self.parallel_flag==False:
            times = [self.times]
            write_csv(file_name=file,data=times+['Unused PE'])
            return -1

        if y >= self.w_w:
            times = [self.times]
            write_csv(file_name=file,data=times+['Unused PE'])
            return -1

        if self.alter_flag and self.parallel_flag == False:
            mid_info,midx,midy=self.err_alter_data(cycles=self.fault_cyc, X=pe_x, Y=y)
            if self.valid_flag==False:
                times = [self.times]
                write_csv(file_name=file, data=times+[mid_info])
                return -1
            else:
                pe_x = midx
                y = midy
                alter_print_flag=True


        x=self.reverse_x(pe_x)
        if self.cul_type == 'is' and self.parallel_flag:
            #'%' only if it's parallel
            ix=pe_x%self.w_h
            x = self.reverse_x(ix)

        if Err_type_set == 0:  # w
            v_now = self.array_w[x][y]
            v_later,flag0_1 = bitflip2(v_now, pos)
            minus = math.fabs(v_now - v_later) 
            if not self.test_valid:
                self.array_w[x][y] = v_later
            if alter_print_flag:#
                record = [self.times, 'Effective', self.fault_inall + cycles, 'W', pe_x, y, str(pos), cycles, v_now, v_later,minus, str(flag0_1)]
            else:
                record=[self.times,'Effective',self.fault_inall + cycles,'W',pe_x,y,str(pos),cycles+self.w_h,v_now,v_later,minus,str(flag0_1)]
            if not self.test_valid:
                write_csv(file,record)

        if Err_type_set == 1:  # s
            if np.isnan(par_ifmap[0][x][y]):
                write_csv(file,["Invalid Injection S"])
                self.err_flag = False
                return -1
            #The ifMap and psum shapes differ in the first line
            x+=1
            if x>=self.w_h:
                write_csv(file, ["Invalid Injection S"])
                self.err_flag = False
                return -1

            v_now = self.array_sum[x][y]
            v_later ,flag0_1= bitflip2(v_now, pos)
            minus = math.fabs(v_now - v_later)  
            self.array_sum[x][y] = v_later
            record = [self.times,'Effective',self.fault_inall + cycles, 'S', pe_x, y, str(pos), cycles, v_now, v_later, minus,str(flag0_1)]
            if not self.test_valid:
                write_csv(file, record)

        if Err_type_set == 2:  # i
            if (np.isnan(par_ifmap[0][x][y])):
                write_csv(file,["Invalid Injection I"])
                self.err_flag = False
                return -1
            v_now = par_ifmap[0][x][y]
            v_later ,flag0_1= bitflip2(v_now, pos)
            minus = math.fabs(v_now - v_later) 
            par_ifmap[0][x][y] = v_later
            record = [self.times,'Effective',self.fault_inall + cycles, 'I', pe_x, y, str(pos), cycles, v_now, v_later, minus,str(flag0_1)]
            if not self.test_valid:
                write_csv(file, record)
            actual_x=x
            actual_y=int(par_ifmap[1][x][y])
            if self.cul_layer=='fc' and self.cul_type=='ws':
                if (self.array_i[actual_x][0] != v_now):
                    print("Error, please stop!")
                else:
                    print('Fault injection success!')
                self.array_i[actual_x][0] = v_later
            elif self.cul_layer=='cov' and self.cul_type=='is':
                if self.wsisfill_tag!='':
                    actual_y-=self.wsisfill_tag
                if (self.array_i[x][actual_y] != v_now):
                    print("Error, please stop!")
                else:
                    print('Fault injection success!')
                self.array_i[x][actual_y] = v_later
            else:
                if self.wsisfill_tag!='':
                    actual_y-=self.wsisfill_tag
                if (self.array_i[actual_x][actual_y] != v_now):
                    print("Error, please stop!")
                else:
                    print('Fault injection success!')
                self.array_i[actual_x][actual_y]=v_later
        self.err_flag = False 
        self.change0_1=flag0_1
        return par_ifmap

    def err_inj_os(self,par_ifmap,par_wrr,cycles=0,err_type_set='',X='',Y='',f_bit='',info='Invalid cycle'):
        file = "AVF_record/"+self.mid_root+"Fault_Info"+str(self.tip)
        if not(self.valid_flag):
            times = [self.times]
            write_csv(file_name=file, data=times+[info])
            return -1,''

        print('Fault injection is being performed')
        Err_type_set=err_type_set
        pe_x=X
        y=Y
        pos=f_bit

        x=self.reverse_x(pe_x)

        if Err_type_set == 0:  # w
            if (np.isnan(par_wrr[0][x][y])):
                write_csv(file,["Invalid Injection W"])
                self.err_flag = False
                return -1,''
            v_now = par_wrr[0][x][y]
            v_later ,flag0_1= bitflip2(v_now, pos)
            minus = math.fabs(v_now - v_later) 
            par_wrr[0][x][y]=v_later
            actual_x = par_wrr[2][x][y]
            if (self.array_w[0][int(actual_x)][y] != v_now):
                print("Error, please stop!")
            else:
                print('Fault injection success!')
            self.array_w[0][int(actual_x)][y] = v_later
            record=[self.times,'Effective',self.fault_inall + cycles,'W',pe_x,y,pos,cycles,v_now,v_later,minus,flag0_1]
            write_csv(file,record)

        if Err_type_set == 1:  # s
            if np.isnan(par_ifmap[0][x][y]):
                write_csv(file,["Invalid Injection S"])
                self.err_flag = False
                return -1,''
            v_now = self.array_sum[x][y]
            v_later ,flag0_1= bitflip2(v_now, pos)
            minus = math.fabs(v_now - v_later) 
            self.array_sum[x][y] = v_later
            record = [self.times,'Effective',self.fault_inall + cycles, 'S', pe_x, y, pos, cycles, v_now, v_later, minus,flag0_1]
            write_csv(file, record)

        if Err_type_set == 2:  # i
            if (np.isnan(par_ifmap[0][x][y])):
                write_csv(file,["Invalid Injection I"])
                self.err_flag = False
                return -1,''
            v_now = par_ifmap[0][x][y]
            v_later ,flag0_1= bitflip2(v_now, pos)
            minus = math.fabs(v_now - v_later) 
            par_ifmap[0][x][y] = v_later
            record = [self.times,'Effective',self.fault_inall + cycles, 'I', pe_x, y, pos, cycles, v_now, v_later, minus,flag0_1]
            write_csv(file, record)
            actual_y=par_ifmap[1][x][y]

            if (self.array_i[x][int(actual_y)] != v_now):
                print("Error, please stop!")
            else:
                print('Fault injection success!')
            self.array_i[x][int(actual_y)]=v_later
        self.err_flag = False 

        self.change0_1=flag0_1
        return par_ifmap,par_wrr

    def err_hard_inj(self,x,y,clas,wt='',input='',psum=''):
        pe_x=x
        x = self.reverse_pe(pe_x)
        if clas=='w':map=wt
        elif clas=='i':map=input
        else:map=psum
        try:
            test=map[x][y]
        except:
            print(' Assignment error!', x, y, ' Effective range:', self.array_w.shape)
            return False
        if clas=="w":
            i=x
            count=0
            while i>=0:
                try:
                    wt[i][y]=bitflip1(wt[i][y],self.bitflip,fix=self.hard_fix)
                except:
                    print(' Assignment error!', i, y, ' Effective range:',self.array_w.shape)
                    return False
                i-=1
                count+=1
        elif clas=='i':
            i = y
            count = 0
            while i < input.shape[1]:
                try:
                    input[x][i] = bitflip1(input[x][i], self.bitflip, fix=self.hard_fix)
                except:
                    print(' Assignment error!', x, i, ' Effective range:',self.array_w.shape)
                    return False
                i += 1
                count += 1
        elif clas=='psum':
            i = x
            count=0
            while i >= 0:
                try:
                    psum[i][y] = bitflip1(psum[i][y], self.bitflip, fix=self.hard_fix)
                except:
                    print(' Assignment error!', i, y, ' Effective range:',self.array_w.shape)
                    return False
                i -= 1
                count += 1
        else:
            try:
                psum[x][y] = bitflip1(psum[x][y], self.bitflip, fix=self.hard_fix)
            except:
                print(' Assignment error!', x, y, ' Effective range:', self.array_w.shape)
                return False
        return True

    def err_inj_mul(self, par_ifmap='', err_type_set='', X='', Y='', f_bit='', info='Invalid cycle', INFO=[]):
        '''
        Suitable for multi-error fault injection
        '''
        file = "AVF_record/" + self.mid_root + "Fault_Info" + str(self.tip)
        if not (self.valid_flag):
            times = [self.times]
            if INFO != []:
                write_csv(file_name=file, data=times + INFO)
            else:
                write_csv(file_name=file, data=times + info)
            return -1

        print('Fault injection is being performed')

        pe_x = X
        y = Y
        pos = f_bit
        Err_type_set = err_type_set

        self.coord_x = pe_x
        self.coord_y = y
        self.bitflip = pos

        if pe_x >= self.w_h and self.parallel_flag == False:
            return par_ifmap

        if y >= self.w_w:
            return par_ifmap

        if self.alter_flag and self.parallel_flag == False:
            mid_info, midx, midy = self.err_alter_data(cycles=self.fault_cyc, X=pe_x, Y=y)
            if self.valid_flag == False:
                return par_ifmap
            else:
                pe_x = midx
                y = midy

        x = self.reverse_x(pe_x)
        if self.cul_type == 'is' and self.parallel_flag:
            # '%' only if it's parallel
            ix = pe_x % self.w_h
            x = self.reverse_x(ix)

        if Err_type_set == 0:  # w
            v_now = self.array_w[x][y]
            v_later, flag0_1 = bitflip2(v_now, pos)
            minus = math.fabs(v_now - v_later) 
            self.array_w[x][y] = v_later

            if not self.test_valid:
                return par_ifmap

        if Err_type_set == 1:  # s
            if np.isnan(par_ifmap[0][x][y]):
                return par_ifmap
            x += 1
            if x >= self.w_h:
                return par_ifmap

            v_now = self.array_sum[x][y]
            v_later, flag0_1 = bitflip2(v_now, pos)
            minus = math.fabs(v_now - v_later)
            self.array_sum[x][y] = v_later

            if not self.test_valid:
                return par_ifmap

        if Err_type_set == 2:  # i
            if (np.isnan(par_ifmap[0][x][y])):
                return par_ifmap
            v_now = par_ifmap[0][x][y]
            v_later, flag0_1 = bitflip2(v_now, pos)
            minus = math.fabs(v_now - v_later)
            par_ifmap[0][x][y] = v_later

            if not self.test_valid:
                return par_ifmap
            actual_x = x
            actual_y = int(par_ifmap[1][x][y])
            if self.cul_layer == 'fc' and self.cul_type == 'ws':
                if (self.array_i[actual_x][0] != v_now):
                    print("Error, please stop!")
                else:
                    print('Fault injection success!')
                self.array_i[actual_x][0] = v_later
            elif self.cul_layer == 'cov' and self.cul_type == 'is':
                if self.wsisfill_tag != '':
                    actual_y -= self.wsisfill_tag
                if (self.array_i[x][actual_y] != v_now):
                    print("Error, please stop!")
                else:
                    print('Fault injection success!')
                self.array_i[x][actual_y] = v_later
            else:
                if self.wsisfill_tag != '':
                    actual_y -= self.wsisfill_tag
                if (self.array_i[actual_x][actual_y] != v_now):
                    print("Error, please stop!")
                else:
                    print('Fault injection success!')
                self.array_i[actual_x][actual_y] = v_later

        return par_ifmap

    def err_inj_os_mul(self,par_ifmap,par_wrr,err_type_set='',X='',Y='',f_bit='',info='Invalid cycle'):
        file = "AVF_record/"+self.mid_root+"Fault_Info"+str(self.tip)
        if not(self.valid_flag):
            times = [self.times]
            write_csv(file_name=file, data=times+[info])
            return -1,''

        print('Fault injection is being performed')
        Err_type_set=err_type_set
        pe_x=X
        y=Y
        pos=f_bit

        x=self.reverse_x(pe_x)

        if Err_type_set == 0:  # w
            if (np.isnan(par_wrr[0][x][y])):
                return par_ifmap,par_wrr
            v_now = par_wrr[0][x][y]
            v_later ,flag0_1= bitflip2(v_now, pos)
            minus = math.fabs(v_now - v_later)
            par_wrr[0][x][y]=v_later
            actual_x = par_wrr[2][x][y]
            if (self.array_w[0][int(actual_x)][y] != v_now):
                print("Error, please stop!")
            else:
                print('Fault injection success!')
            self.array_w[0][int(actual_x)][y] = v_later

        if Err_type_set == 1:  # s
            if np.isnan(par_ifmap[0][x][y]):
                return par_ifmap,par_wrr
            v_now = self.array_sum[x][y]
            v_later ,flag0_1= bitflip2(v_now, pos)
            self.array_sum[x][y] = v_later

        if Err_type_set == 2:  # i
            if (np.isnan(par_ifmap[0][x][y])):
                return par_ifmap,par_wrr
            v_now = par_ifmap[0][x][y]
            v_later ,flag0_1= bitflip2(v_now, pos)
            par_ifmap[0][x][y] = v_later
            actual_y=par_ifmap[1][x][y]

            if (self.array_i[x][int(actual_y)] != v_now):
                print("Error, please stop!")
            else:
                print('Fault injection success!')
            self.array_i[x][int(actual_y)]=v_later

        self.change0_1=flag0_1
        return par_ifmap,par_wrr

    #If the data to be loaded into the PE matrix exceeds the size of the PE,
    #a process of data substitution occurs: that is, only part of the data is loaded in each unit calculation
    #However, errors in the data replacement process may have subsequent effects.
    def err_alter_data(self,cycles,X,Y):
        '''
        ermines whether the error injected in the current period is valid.
        If it is valid, it will be replaced. If it is invalid, it will return an invalid flag
        '''
        info=''
        real_x=''
        real_y=''

        if (X-cycles+1>0):
            self.valid_flag=False
            info='Invalid alter data cycle'
            print('Idle PE')
        else:
            real_x=X+self.w_h-cycles##
            real_y=Y
            print('Effective PE')
        return info,real_x,real_y

    def reverse_x(self,x):
        '''
        Because the matrix data was reversed in the calculation, the position and data should correspond correctly when
        have fault injection.
        In the reverse process, the y coordinate remains the same, so there is no need to modify it
        '''
        if self.cul_type=='os':
            return self.i_h - x - 1
        return self.w_h-x-1

    def reverse_pe(self,x):
        return self.pe_size-x-1

    def is_actualy(self,x,y,par_ifmap):
        count=0
        width=par_ifmap.shape[1]
        for i in range(y,width):
            if not np.isnan(par_ifmap[x][i]):
                count+=1
            else:
                break
        actual_y=self.i_w-count
        return actual_y




def nan_process(arr):
    arr[np.isnan(arr)]=0
    return arr

def change_input(ifmap):
    i_h=ifmap.shape[0]
    i_w=ifmap.shape[1]
    ifmap_tap = np.full((i_h, i_h-1), np.nan, dtype=np.float32)
    ifmap_tap = np.c_[ifmap,ifmap_tap]
    for i in range(i_h):
        if i==0:continue
        ifmap_tap[i]=shiftROW(ifmap_tap[i],i_w,i)
    return ifmap_tap

def edge_map(ifmap,w_h,w_w):
    edge=np.full((w_h,w_w-1),np.nan,dtype=np.float32)
    arr=np.c_[edge,ifmap,edge]
    return arr

def shiftROW(row,l,k):
    #Shift the data to the right by k units
    #Effective data length: l
    for i in range(l):
        row[l+k-1-i]=row[l-1-i]
        row[l-1-i]=np.nan
    return row

def aquire_map(cycles,w_h,w_w,ifmap):
    i_h=ifmap.shape[0]
    i_w=ifmap.shape[1]
    par_arr=ifmap[:i_w-1-cycles-w_w,i_w-1-cycles]
    return par_arr

@nb.jit(nopython=True)
def aq_os_part_num(x):
    #Returns a positive power
    rs=0
    while(x>0):
        x-=1000
        rs+=1
    return rs

@nb.jit(nopython=True)
def shiftrow(row, l, r_l=1):
    '''

    :param row: Matrix Row
    :param l: Effective data length
    :param r_l:Shift data to the left or right, default to 1 as shift to the left
    :return:
    '''
    L = len(row)  
    if r_l == 1:
        if l < L:
            shfk = L - l
            for i in range(l):
                row[i] = row[i + shfk]
                row[i + shfk] = np.nan
            return row
        else: 
            return row
    else:
        if l <= 0 or l > L:
            row=[np.nan for i in range(L)]
            row=np.array(row)
            return row 
        shfk = L - l
        for i in range(l):
            row[L - i - 1] = row[L - i - shfk - 1]
            row[L - i - shfk - 1] = np.nan
        for k in range(shfk): 
            row[k] = np.nan
        return row

@nb.jit(nopython=True)
def acc_map(cycles,w_h,w_w,i_h,i_w,ifmap):
    '''This step organizes the numbers between the upper bound and lower bound of the original matrix into the matrix
    that participates in the operation'''
    pi = i_h - 1  # Take the coordinates of the lower right corner
    pj = i_w - 1
    par_arr_i = np.full((2,w_h, w_w),np.nan)
    upper_b = pi + pj - cycles+1
    lower_b = upper_b + w_w
    flag = 0
    sign = 0
    for ip1 in range(pi + 1):
        k_j = 0
        for ip2 in range(pj + 1):
            if (ip1 + ip2 >= upper_b) and ip1 + ip2 < lower_b:
                par_arr_i[0][ip1][k_j] = ifmap[ip1][ip2] 
                if not np.isnan(ifmap[ip1][ip2]):
                    par_arr_i[1][ip1][k_j]=ip2
                k_j += 1
                if ip1 == 0 and ip2 == 0: 
                    sign = 1

        if k_j == w_w:
            flag = 1
        if k_j < w_w and flag == 1:
            flag = -1
        if k_j < w_w and sign == 1:
            flag = -1
        if cycles > i_w and k_j < w_w and flag == -1 and k_j > 0:
            par_arr_i[0][ip1] = shiftrow(par_arr_i[0][ip1], k_j, 0)
            par_arr_i[1][ip1] = shiftrow(par_arr_i[1][ip1], k_j, 0)
    return par_arr_i

@nb.jit(nopython=True)
def ju_os_out(par_wrr,idxrc,idxrc_os,array_sum,os_union=1):
    '''
    This step passes the value to be updated into the idxrc_os array,
    placing the actual index that needs to be filled in to the final output position in the idxrc array
    '''
    flag=False
    if np.max(par_wrr[1])==0:
        return flag
    else:
        flag=True
    for j in range(par_wrr[1].shape[1]):
        for i in range(par_wrr[1].shape[0]):
            if par_wrr[1][i][j]!=0:
                J=j
                num_os_part=0
                if j>=idxrc.shape[0]:
                    break
                if idxrc[j][0]==-1:
                    if os_union>1:
                        num_os_part=aq_os_part_num(par_wrr[1][i][j])-1
                        J+=num_os_part*par_wrr[1].shape[1]
                        if idxrc[J][0]==-1:
                            continue
                    else:
                        continue
                Y=int(par_wrr[1][i][j]-1-num_os_part*1000)
                idxrc_os[0][J][Y]=array_sum[i][j]
                array_sum[i][j]=0
                idxrc[J][0]=1
    return flag

@nb.jit(nopython=True)
def get_arr_wrr(np_k,array_i,array_w,w_w):
    '''Intercept the matrix, I from right to left, W from top to bottom'''
    I_w = array_i.shape[1]
    I_h = array_i.shape[0]
    W_w = w_w
    arr_w = array_w
    arr_i = array_i
    par_arr_i1 = arr_i[:, I_w - I_h - np_k:I_w - np_k]
    par_arr_i2=np.zeros(par_arr_i1.shape)
    for i in range(par_arr_i1.shape[1]):
        for j in range(par_arr_i1.shape[0]):
            par_arr_i2[j][i]=I_w - I_h - np_k+i

    par_arr_w1 = arr_w[0][0 + np_k:W_w + np_k, :]
    par_arr_w2 = arr_w[1][0 + np_k:W_w + np_k, :]
    par_arr_w3=np.zeros(par_arr_w1.shape)
    for i in range(par_arr_w3.shape[0]):
        par_arr_w3[i]=np_k+i

    par_arr_i1 = np.expand_dims(par_arr_i1, axis=0)
    par_arr_i2 = np.expand_dims(par_arr_i2, axis=0)
    par_arr_w1 = np.expand_dims(par_arr_w1, axis=0)
    par_arr_w2 = np.expand_dims(par_arr_w2, axis=0)
    par_arr_w3 = np.expand_dims(par_arr_w3, axis=0)
    par_arr_i = np.concatenate((par_arr_i1, par_arr_i2), axis=0)
    par_arr_w = np.concatenate((par_arr_w1, par_arr_w2,par_arr_w3), axis=0)
    return par_arr_i, par_arr_w



































