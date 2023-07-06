import os

def delete_file(PATH):
    for i in range(len(PATH)):
        os.remove(PATH[i])
    print('Delete %d files successfuly'%(i+1))

def delete_target(dir_name,target_type):
    for files in os.listdir(dir_name):
        if files.endswith(target_type):
            os.remove(os.path.join(dir_name,files))

def loc_file():
    path=[]
    a=r"D:\PaperAndWork\newProject\FI-scale-wqx\scale_sim\RS\sram_read.csv"
    b=r"D:\PaperAndWork\newProject\FI-scale-wqx\scale_sim\RS\sram_write.csv"
    path.append(a)
    path.append(b)
    return path

def create_dir(data_flow,name,root=''):
    base=root+data_flow+'/'
    path=base+name
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)

def mode_dir(data_flow='ws',root=''):
    mode=['lenet','vgg16','cifar']
    for i in mode:
        create_dir(data_flow,i,root)

def dataflow_dir(root):
    dataflow=['is','os','ws']
    base = root
    for i in dataflow:
        path = base + i
        isExists = os.path.exists(path)
        if not isExists:
            os.makedirs(path)

    for i in dataflow:
        mode_dir(data_flow=i,root=base)

def deal_with_dir():
    ##Create the directory to store the Scalesim files
    root='../scale_sim/RS/'
    ##The folder where the results are recorded
    root2='../AVF_record/'
    dataflow_dir(root)
    dataflow_dir(root2)

if __name__=="__main__":
    deal_with_dir()##Create a new records folder
