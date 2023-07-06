import csv
import numpy as np

def write_csv(file_name,data):
    a = list(data)
    file=open(file_name+'.csv','a',encoding='utf-8',newline="")
    pointer=csv.writer(file)
    pointer.writerow(a)
    file.close()

def rs_name():
    avf='../AVF_record/AVF-rate'
    err='../AVF_record/Err_info'
    rs_info='../AVF_record/compare_info'
    return avf,err,rs_info

def write_header(avf,err,rs_info,type='lenet',exp_type='bit'):
    # avf,err,rs_info=avf,err,rs_info
    if type=='lenet':
        err_header = ['Times','Info', 'Cyc', 'type', 'X', 'Y', 'Bit', 'cyc', 'v_ori', 'v_lat', 'DEC','1;0change']
        avf_header=['Times','Avf_all_acc','Avf_16_acc','Avf_8_acc','Avf_clas_acc']
        compare_header=['Times','Avf_all_acc','Avf_16_acc','Avf_8_acc','Avf_clas_acc']
    elif type=='vgg16':
        err_header = ['Times', 'Info', 'Cyc', 'type', 'X', 'Y', 'Bit', 'cyc', 'v_ori', 'v_lat', 'DEC', '1;0change']
        avf_header = ['Times', 'Avf_all_acc', 'Avf_6_acc', 'Avf_4_acc', 'Avf_clas_acc']
        compare_header = ['Times', 'Avf_all_acc', 'Avf_6_acc', 'Avf_4_acc', 'Avf_clas_acc']
    else:###cifar
        if 'hard' in exp_type:
            err_header = ['Times', 'Type', 'Loc', 'X', 'Y', 'Bit']
        else:
            err_header = ['Times', 'Info', 'Cyc', 'type', 'X', 'Y', 'Bit', 'cyc', 'v_ori', 'v_lat', 'DEC', '1;0change']
        avf_header = ['Times', 'Avf_all_acc', 'Avf_16_acc', 'Avf_8_acc', 'Avf_clas_acc','top1_16_acc','top1_8_acc','top1_clas_acc']
        compare_header = ['Times', 'Avf_all_acc', 'Avf_16_acc', 'Avf_8_acc', 'Avf_clas_acc']
    write_csv(err,err_header)
    write_csv(avf,avf_header)
    write_csv(rs_info, compare_header)


def write_txt(data,name='Test1.txt'):
    file=name
    doc=open(file,'w')
    print(data,file=doc)
    doc.close()

def write_txt2(data,name='Test2.txt'):
    file=name
    doc=open(file,'w')
    for i in range(len(data)):
        print(data[i],file=doc)
    doc.close()