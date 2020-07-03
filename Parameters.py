"""
parameters
"""
threshold = 0.15
subject_number = 100
remove_length = 1
window_length = 1199
# bn_RS_Regressed.mat： 246
# aal_RS_Regressed.mat： 90
main_dir = 'F:\Data\REST1'
# main_dir = "D:\\fMRI_project"

RS_dir = 'aal_RS_Regressed.mat'
node_number = 90
Network_Dir = '/home/ren/fMRI-Gao/Data/Result_shen268/Network/'
# Network_Dir = "D:\\fMRI_project\\fMRI"

# RS_dir = 'bn_RS_Regressed.mat'
# node_number = 246
# Network_Dir = '/home/imagetech/Research/ID_Network/Data/Result_BN/Network'

# RS_dir = 'aal_RS_Regressed.mat'
# node_number = 90
# Network_Dir = '/home/imagetech/Research/ID_Network/Data/Result_AAL/Network'

feature_number = int(node_number * (node_number - 1) / 2)


# model
hidden_1 = 2000
hidden_2 = 2000
hidden_3 = 2000

batch_size = 2000
epochs = 100
drop_out = 10
act_model = 'sigmoid'
final_model = 'sigmoid'

model_folder = '/home/ren/fMRI-Gao/Model/'
# model_folder = "D:\\fMRI_project\\AEsModel"
