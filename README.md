# saca-FI
In this work, we develop an architecture-level fault injection framework, saca-FI, to analyze the reliability of systolic array based CNN accelerators. Saca-FI is flexible to be applied to evaluate multiple CNN architectures and models. 

# Runtime Environment: 
Operating system: Windows10 
Deep Learning Framework: python3.6.6 + keras2.2.4 + tensorflow1.8.0 
Scientific computing library: numpy1.17.0 + numba0.53.1

# Operating Instructions:
Run the Auto_proc.py file. Make sure you have a csv mapping file for the configuration in the scale_sim/RS/.. folder. If it doesn't exist, set the csv_exist parameter to False in Config_info.py. 
The parameters are changed in Config_info.py.
The models are stored in the mode folder, which contains the trained LeNet-5 model (modelnew.h5) and CIFAR-10 CNN model (cifar.h5). 
The input images are stored in the pic folder.
Path description: You need to change the a and b paths in the loc_file function of tool/file_manage.py to local paths.
Results: Saved in the AVF_record folder.

# Paper: More details can be found here:
Saca-FI: A microarchitecture-level fault injection framework for reliability analysis of systolic array based CNN accelerator，https://doi.org/10.1016/j.future.2023.05.009

It would be appreciated if you could share your improvement suggestions with us.
