import os
from tool.file_manage import delete_target

#Reads information from a configuration file and places it in a class for the program to call

#####Experimental parameters
stride=1
#########data flow ##########
# data_flow = 'ws'############################################parameter1
# data_flow = 'is'
data_flow = 'os'

class Config:
    def __init__(self,mod_type='lenet',mod_path='',data='',pe_size=64,times=3000,layer=1,layer_type='cov',
                 data_flow='ws',err_set='random',expr_set='random',fin_layer=False,err_flag=False,
                 scale_path='',csv_exist=False):

        self.mod_type=mod_type
        self.mod_path=mod_path
        self.data=data
        self.pe_size=pe_size
        self.times=times
        self.layer=layer
        self.layer_type=layer_type
        self.data_flow=data_flow
        self.err_set=err_set
        self.expr_set=expr_set
        self.fin_layer=fin_layer
        self.err_flag=err_flag
        self.scale_path=scale_path
        self.csv_exist=csv_exist

    def get_config(self):
        return self.mod_type,self.mod_path,self.data,self.pe_size,\
               self.times,self.layer,self.layer_type,self.data_flow,\
               self.err_set,self.expr_set,self.fin_layer,self.err_flag,\
               self.scale_path,self.csv_exist




def loadin_config():
    my_config=Config()
    return my_config


def aquire_config():#Read experiment configuration
    delete_target(os.getcwd(), '.csv')
    ####Whether the scale_sim csv file already exists
    csv_exist=True###############################################parameter2
    # csv_exist = False
    ####pe_setting
    pe_size = 32
    pe_size = 64
    pe_size = 128
    pe_size = 256
    # pe_size = 512
    ######### layer type ######
    layer_type='cov'##############################################parameter3
    # layer_type = 'fc'

    #####Failure types and experiments
    err_set = 'random_ws'#########################################parameter4
    # err_set = 'random_is'
    err_set='random_os'
    # ######hard_err
    # err_set='hard_fix0'
    # err_set='hard_fix1'
    # # ######multi_err
    # err_set='multi_err'
    # err_set='multi_bit'

    err_flag = True
    # err_flag = False
    #########data flow ##########
    cul_type = data_flow
    ########## mode&layer #########
    ########lenet
    # mode = 'lenet'##############################################parameter5
    # layer = 0#0;1;5;7###########################################parameter6
    # fin_layer = False###########################################parameter7
    # fin_layer = True

    ########cifar
    mode= 'cifar'
    layer=1#0,1,4,5;9,11
    fin_layer=False
    # # fin_layer=True

    ########vgg16
    # mode = 'vgg16'
    # layer = 7#1,2,4,5,7,8,9,11,12,13,15,16,17;fc:20,21,22
    # fin_layer = False
    #fin_layer = True

    ########mobilenet
    # mode = 'mobilenet'
    # layer = 28#2,8,15,21,28,34,41,47,53,59,65,71,78,84,90
    # fin_layer = False


    #####file_path:sacle csv&mode
    scale_path = 'scale_sim/RS/'+cul_type+'/'+mode+'/sram_read'+str(layer)+'pe'+str(pe_size)+'.csv'

    mode_path='mode/vgg16_weights.h5'

    #####The total number of experiments
    times=10000

    my_config = Config(pe_size=pe_size,layer_type=layer_type,data_flow=cul_type,err_set=err_set,
                       layer=layer,mod_type=mode,err_flag=err_flag,fin_layer=fin_layer,scale_path=scale_path,
                       mod_path=mode_path,times=times,csv_exist=csv_exist)
    return my_config
