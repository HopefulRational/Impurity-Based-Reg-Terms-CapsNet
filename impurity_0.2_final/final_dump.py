from utils import *
from resnet_cnnsovnet_dynamic_routing_2_entropy_dec import *
import argparse
from datetime import date
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from resnet_cnnsovnet_dynamic_routing_2 import *
from constants import *
from utils import *
from data_loaders import *
from smallNorb import *

best_accuracy = 0.0
num_epochs = DEFAULT_EPOCHS
loss_criterion = CrossEntropyLoss()
model = ResnetCnnsovnetDynamicRouting().to(DEVICE)

num_classes = 5
trainloader, validloader, testloader = load_small_norb(DEFAULT_BATCH_SIZE)


parser = argparse.ArgumentParser(description='Logging for entropy-regularised CapsNet')
parser.add_argument('--chk_file', metavar='CHKPNT_FILE', type=str, default='./checkpoints/MNIST/CapsNet/chkpoint_adam_ExpLR_run1.pth', help='checkpoint file')
parser.add_argument('--imp_cij_folder', type=str, default='./dumps/impurities_cijs', help='folder for impurity values for c_ij')
parser.add_argument('--conf_file', metavar='COFUSION_FILE', type=str, default='./dumps/confusion.csv', help='confusion_matrix file')
parser.add_argument('--res_file', metavar='RESULTS_FILE', type=str, default='./dumps/results.txt', help='results.txt file')
parser.add_argument('--reg_type', metavar='REG_TYPE', type=str, default='entropy_reg', help='type of reg')
args = parser.parse_args()

today = date.today()

with open(args.res_file, "a") as fl:
    fl.write('\n\n'+args.chk_file+'\n'+str(today)+'\n')
fl.close()

with open(args.conf_file, "a") as conf_fl:
    conf_fl.write('\n\n'+args.chk_file+'\n'+str(today)+'\n')
conf_fl.close()

conf_fl = open(args.conf_file, "a")


chk = torch.load(args.chk_file)
acc = chk['acc']
model.load_state_dict(chk['model'])

PRIMARY_ACTS = 0
CLASS_ACTS = 1
ENTR1 = 0
ENTR2 = 1
ENTR3 = 2
ENTR4 = 3
GINI1 = 4
GINI2 = 5
GINI3 = 6
GINI4 = 7
MCLS1 = 8
MCLS2 = 9
MCLS3 = 10
MCLS4 = 11


def switchFile(num):
    switcher = {
        0:[entr0_1, entr0_2, entr0_3, entr0_4, gini0_1, gini0_2, gini0_3, gini0_4, mcls0_1, mcls0_2, mcls0_3, mcls0_4],
        1:[entr1_1, entr1_2, entr1_3, entr1_4, gini1_1, gini1_2, gini1_3, gini1_4, mcls1_1, mcls1_2, mcls1_3, mcls1_4],
        2:[entr2_1, entr2_2, entr2_3, entr2_4, gini2_1, gini2_2, gini2_3, gini2_4, mcls2_1, mcls2_2, mcls2_3, mcls2_4], 
        3:[entr3_1, entr3_2, entr3_3, entr3_4, gini3_1, gini3_2, gini3_3, gini3_4, mcls3_1, mcls3_2, mcls3_3, mcls3_4],
        4:[entr4_1, entr4_2, entr4_3, entr4_4, gini4_1, gini4_2, gini4_3, gini4_4, mcls4_1, mcls4_2, mcls4_3, mcls4_4] 
    }
    return switcher.get(num)


entr1_cij_f0 = open(args.imp_cij_folder + '/entr1' + '/0.csv', 'w')
entr1_cij_f1 = open(args.imp_cij_folder + '/entr1' + '/1.csv', 'w')
entr1_cij_f2 = open(args.imp_cij_folder + '/entr1' + '/2.csv', 'w')
entr1_cij_f3 = open(args.imp_cij_folder + '/entr1' + '/3.csv', 'w')
entr1_cij_f4 = open(args.imp_cij_folder + '/entr1' + '/4.csv', 'w')

entr2_cij_f0 = open(args.imp_cij_folder + '/entr2' + '/0.csv', 'w')
entr2_cij_f1 = open(args.imp_cij_folder + '/entr2' + '/1.csv', 'w')
entr2_cij_f2 = open(args.imp_cij_folder + '/entr2' + '/2.csv', 'w')
entr2_cij_f3 = open(args.imp_cij_folder + '/entr2' + '/3.csv', 'w')
entr2_cij_f4 = open(args.imp_cij_folder + '/entr2' + '/4.csv', 'w')

entr3_cij_f0 = open(args.imp_cij_folder + '/entr3' + '/0.csv', 'w')
entr3_cij_f1 = open(args.imp_cij_folder + '/entr3' + '/1.csv', 'w')
entr3_cij_f2 = open(args.imp_cij_folder + '/entr3' + '/2.csv', 'w')
entr3_cij_f3 = open(args.imp_cij_folder + '/entr3' + '/3.csv', 'w')
entr3_cij_f4 = open(args.imp_cij_folder + '/entr3' + '/4.csv', 'w')

entr4_cij_f0 = open(args.imp_cij_folder + '/entr4' + '/0.csv', 'w')
entr4_cij_f1 = open(args.imp_cij_folder + '/entr4' + '/1.csv', 'w')
entr4_cij_f2 = open(args.imp_cij_folder + '/entr4' + '/2.csv', 'w')
entr4_cij_f3 = open(args.imp_cij_folder + '/entr4' + '/3.csv', 'w')
entr4_cij_f4 = open(args.imp_cij_folder + '/entr4' + '/4.csv', 'w')

gini1_cij_f0 = open(args.imp_cij_folder + '/gini1' + '/0.csv', 'w')
gini1_cij_f1 = open(args.imp_cij_folder + '/gini1' + '/1.csv', 'w')
gini1_cij_f2 = open(args.imp_cij_folder + '/gini1' + '/2.csv', 'w')
gini1_cij_f3 = open(args.imp_cij_folder + '/gini1' + '/3.csv', 'w')
gini1_cij_f4 = open(args.imp_cij_folder + '/gini1' + '/4.csv', 'w')

gini2_cij_f0 = open(args.imp_cij_folder + '/gini2' + '/0.csv', 'w')
gini2_cij_f1 = open(args.imp_cij_folder + '/gini2' + '/1.csv', 'w')
gini2_cij_f2 = open(args.imp_cij_folder + '/gini2' + '/2.csv', 'w')
gini2_cij_f3 = open(args.imp_cij_folder + '/gini2' + '/3.csv', 'w')
gini2_cij_f4 = open(args.imp_cij_folder + '/gini2' + '/4.csv', 'w')

gini3_cij_f0 = open(args.imp_cij_folder + '/gini3' + '/0.csv', 'w')
gini3_cij_f1 = open(args.imp_cij_folder + '/gini3' + '/1.csv', 'w')
gini3_cij_f2 = open(args.imp_cij_folder + '/gini3' + '/2.csv', 'w')
gini3_cij_f3 = open(args.imp_cij_folder + '/gini3' + '/3.csv', 'w')
gini3_cij_f4 = open(args.imp_cij_folder + '/gini3' + '/4.csv', 'w')

gini4_cij_f0 = open(args.imp_cij_folder + '/gini4' + '/0.csv', 'w')
gini4_cij_f1 = open(args.imp_cij_folder + '/gini4' + '/1.csv', 'w')
gini4_cij_f2 = open(args.imp_cij_folder + '/gini4' + '/2.csv', 'w')
gini4_cij_f3 = open(args.imp_cij_folder + '/gini4' + '/3.csv', 'w')
gini4_cij_f4 = open(args.imp_cij_folder + '/gini4' + '/4.csv', 'w')

mcls1_cij_f0 = open(args.imp_cij_folder + '/mcls1' + '/0.csv', 'w')
mcls1_cij_f1 = open(args.imp_cij_folder + '/mcls1' + '/1.csv', 'w')
mcls1_cij_f2 = open(args.imp_cij_folder + '/mcls1' + '/2.csv', 'w')
mcls1_cij_f3 = open(args.imp_cij_folder + '/mcls1' + '/3.csv', 'w')
mcls1_cij_f4 = open(args.imp_cij_folder + '/mcls1' + '/4.csv', 'w')

mcls2_cij_f0 = open(args.imp_cij_folder + '/mcls2' + '/0.csv', 'w')
mcls2_cij_f1 = open(args.imp_cij_folder + '/mcls2' + '/1.csv', 'w')
mcls2_cij_f2 = open(args.imp_cij_folder + '/mcls2' + '/2.csv', 'w')
mcls2_cij_f3 = open(args.imp_cij_folder + '/mcls2' + '/3.csv', 'w')
mcls2_cij_f4 = open(args.imp_cij_folder + '/mcls2' + '/4.csv', 'w')

mcls3_cij_f0 = open(args.imp_cij_folder + '/mcls3' + '/0.csv', 'w')
mcls3_cij_f1 = open(args.imp_cij_folder + '/mcls3' + '/1.csv', 'w')
mcls3_cij_f2 = open(args.imp_cij_folder + '/mcls3' + '/2.csv', 'w')
mcls3_cij_f3 = open(args.imp_cij_folder + '/mcls3' + '/3.csv', 'w')
mcls3_cij_f4 = open(args.imp_cij_folder + '/mcls3' + '/4.csv', 'w')

mcls4_cij_f0 = open(args.imp_cij_folder + '/mcls4' + '/0.csv', 'w')
mcls4_cij_f1 = open(args.imp_cij_folder + '/mcls4' + '/1.csv', 'w')
mcls4_cij_f2 = open(args.imp_cij_folder + '/mcls4' + '/2.csv', 'w')
mcls4_cij_f3 = open(args.imp_cij_folder + '/mcls4' + '/3.csv', 'w')
mcls4_cij_f4 = open(args.imp_cij_folder + '/mcls4' + '/4.csv', 'w')


# For impurities for c_ij per_class
entr0_1,entr0_2, entr0_3, entr0_4,entr1_1,entr1_2,entr1_3,entr1_4,entr2_1,entr2_2,entr2_3,entr2_4,entr3_1,entr3_2,entr3_3,entr3_4,entr4_1,entr4_2, entr4_3, entr4_4                                   = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]

gini0_1,gini0_2, gini0_3, gini0_4,gini1_1,gini1_2,gini1_3,gini1_4,gini2_1,gini2_2,gini2_3,gini2_4,gini3_1,gini3_2,gini3_3,gini3_4,gini4_1,gini4_2, gini4_3, gini4_4                                   = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]

mcls0_1,mcls0_2, mcls0_3, mcls0_4,mcls1_1,mcls1_2,mcls1_3,mcls1_4,mcls2_1,mcls2_2,mcls2_3,mcls2_4,mcls3_1,mcls3_2,mcls3_3,mcls3_4,mcls4_1,mcls4_2, mcls4_3, mcls4_4                                   = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]

# START FROM HERE...

conf_matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

prim_act_sum = [0,0,0,0,0]
with torch.no_grad():
     for batch_idx, (data, lbls) in tqdm(enumerate(testloader)):

         if batch_idx == 11:
             break

         data, labels = data.to(DEVICE), one_hot(lbls).to(DEVICE)
         outputs, c_ij1, c_ij2, c_ij3, c_ij4 = model(data,labels)
         
         for btch_elem in range(data.shape[0]):
             e1 = get_entropies(c_ij1[btch_elem]).squeeze().detach().cpu().numpy()
             g1 = get_ginis(c_ij1[btch_elem]).squeeze().detach().cpu().numpy()
             m1 = get_mcLosses(c_ij1[btch_elem]).squeeze().detach().cpu().numpy()
             
             e2 = get_entropies(c_ij2[btch_elem]).squeeze().detach().cpu().numpy()
             g2 = get_ginis(c_ij2[btch_elem]).squeeze().detach().cpu().numpy()
             m2 = get_mcLosses(c_ij2[btch_elem]).squeeze().detach().cpu().numpy()

             e3 = get_entropies(c_ij3[btch_elem]).squeeze().detach().cpu().numpy()
             g3 = get_ginis(c_ij3[btch_elem]).squeeze().detach().cpu().numpy()
             m3 = get_mcLosses(c_ij3[btch_elem]).squeeze().detach().cpu().numpy()

             e4 = get_entropies(c_ij4[btch_elem]).squeeze().detach().cpu().numpy()
             g4 = get_ginis(c_ij4[btch_elem]).squeeze().detach().cpu().numpy()
             m4 = get_mcLosses(c_ij4[btch_elem]).squeeze().detach().cpu().numpy()
             
             switchFile(torch.max(labels[btch_elem], dim=0)[1].item())[GINI1].append(g1)
             
             switchFile(torch.max(labels[btch_elem], dim=0)[1].item())[MCLS1].append(m1)
             switchFile(torch.max(labels[btch_elem], dim=0)[1].item())[ENTR1].append(e1)

             switchFile(torch.max(labels[btch_elem], dim=0)[1].item())[GINI2].append(g2)
             switchFile(torch.max(labels[btch_elem], dim=0)[1].item())[MCLS2].append(m2)
             switchFile(torch.max(labels[btch_elem], dim=0)[1].item())[ENTR2].append(e2)

             switchFile(torch.max(labels[btch_elem], dim=0)[1].item())[GINI3].append(g3)
             switchFile(torch.max(labels[btch_elem], dim=0)[1].item())[MCLS3].append(m3)
             switchFile(torch.max(labels[btch_elem], dim=0)[1].item())[ENTR3].append(e3)

             switchFile(torch.max(labels[btch_elem], dim=0)[1].item())[GINI4].append(g4)
             switchFile(torch.max(labels[btch_elem], dim=0)[1].item())[MCLS4].append(m4)
             switchFile(torch.max(labels[btch_elem], dim=0)[1].item())[ENTR4].append(e4)

             conf_matrix[torch.max(labels[btch_elem], dim=0)[1].item()][torch.max(outputs[btch_elem], dim=0)[1].item()] += 1             
             
             prim_act_sum[torch.max(labels[btch_elem], dim=0)[1].item()] += 1             

print(f"\nprim_act_sum : {prim_act_sum}")
#prim_act_sum = np.array(prim_act_sum).reshape(num_classes,1)

#np.savetxt(conf_fl, conf_matrix, delimiter=",")
#conf_fl.write("\n\n\n")
#conf_fl.close()

np.savetxt(gini1_cij_f0, np.transpose(gini0_1), delimiter=",")
gini1_cij_f0.write("\n\n\n")
print(f"np.transpose(gini0_1).shape : {np.transpose(gini0_1).shape} | np.transpose(np.array(gini0_1)).shape : {np.transpose(np.array(gini0_1)).shape}")
np.savetxt(gini1_cij_f0, np.average(np.transpose(np.array(gini0_1)),axis=1), delimiter=",")
#gini1_cij_f0.write(f"Average:\n {np.average(np.transpose(np.array(gini0)),axis=1)} \n")
np.savetxt(gini1_cij_f1, np.transpose(gini1_1), delimiter=",")
gini1_cij_f1.write("\n\n\n")
np.savetxt(gini1_cij_f1, np.average(np.transpose(np.array(gini1_1)),axis=1), delimiter=",")
#gini1_cij_f1.write(f"Average:\n{np.average(np.transpose(np.array(gini1)),axis=1)}\n")
np.savetxt(gini1_cij_f2, np.transpose(gini2_1), delimiter=",")
gini1_cij_f2.write("\n\n\n")
np.savetxt(gini1_cij_f2, np.average(np.transpose(np.array(gini2_1)),axis=1), delimiter=",")
#gini1_cij_f2.write(f"Average:\n{np.average(np.transpose(np.array(gini2)),axis=1)}\n")
np.savetxt(gini1_cij_f3, np.transpose(gini3_1), delimiter=",")
gini1_cij_f3.write("\n\n\n")
np.savetxt(gini1_cij_f3, np.average(np.transpose(np.array(gini3_1)),axis=1), delimiter=",")
#gini1_cij_f3.write(f"Average:\n{np.average(np.transpose(np.array(gini3)),axis=1)}\n")
np.savetxt(gini1_cij_f4, np.transpose(gini4_1), delimiter=",")
gini1_cij_f4.write("\n\n\n")
np.savetxt(gini1_cij_f4, np.average(np.transpose(np.array(gini4_1)),axis=1), delimiter=",")
#gini1_cij_f4.write(f"Average:\n{np.average(np.transpose(np.array(gini4)),axis=1)}\n")


np.savetxt(gini2_cij_f0, np.transpose(gini0_2), delimiter=",")
gini2_cij_f0.write("\n\n\n")
np.savetxt(gini2_cij_f0, np.average(np.transpose(np.array(gini0_2)),axis=1), delimiter=",")
#gini2_cij_f0.write(f"Average:\n {np.average(np.transpose(np.array(gini0)),axis=1)} \n")
np.savetxt(gini2_cij_f1, np.transpose(gini1_2), delimiter=",")
gini2_cij_f1.write("\n\n\n")
np.savetxt(gini2_cij_f1, np.average(np.transpose(np.array(gini1_2)),axis=1), delimiter=",")
#gini2_cij_f1.write(f"Average:\n{np.average(np.transpose(np.array(gini1)),axis=1)}\n")
np.savetxt(gini2_cij_f2, np.transpose(gini2_2), delimiter=",")
gini2_cij_f2.write("\n\n\n")
np.savetxt(gini2_cij_f2, np.average(np.transpose(np.array(gini2_2)),axis=1), delimiter=",")
#gini2_cij_f2.write(f"Average:\n{np.average(np.transpose(np.array(gini2)),axis=1)}\n")
np.savetxt(gini2_cij_f3, np.transpose(gini3_2), delimiter=",")
gini2_cij_f3.write("\n\n\n")
np.savetxt(gini2_cij_f3, np.average(np.transpose(np.array(gini3_2)),axis=1), delimiter=",")
#gini2_cij_f3.write(f"Average:\n{np.average(np.transpose(np.array(gini3)),axis=1)}\n")
np.savetxt(gini2_cij_f4, np.transpose(gini4_2), delimiter=",")
gini2_cij_f4.write("\n\n\n")
np.savetxt(gini2_cij_f4, np.average(np.transpose(np.array(gini4_2)),axis=1), delimiter=",")
#gini2_cij_f4.write(f"Average:\n{np.average(np.transpose(np.array(gini4)),axis=1)}\n")


np.savetxt(gini3_cij_f0, np.transpose(gini0_3), delimiter=",")
gini3_cij_f0.write("\n\n\n")
np.savetxt(gini3_cij_f0, np.average(np.transpose(np.array(gini0_3)),axis=1), delimiter=",")
#gini3_cij_f0.write(f"Average:\n {np.average(np.transpose(np.array(gini0)),axis=1)} \n")
np.savetxt(gini3_cij_f1, np.transpose(gini1_3), delimiter=",")
gini3_cij_f1.write("\n\n\n")
np.savetxt(gini3_cij_f1, np.average(np.transpose(np.array(gini1_3)),axis=1), delimiter=",")
#gini3_cij_f1.write(f"Average:\n{np.average(np.transpose(np.array(gini1)),axis=1)}\n")
np.savetxt(gini3_cij_f2, np.transpose(gini2_3), delimiter=",")
gini3_cij_f2.write("\n\n\n")
np.savetxt(gini3_cij_f2, np.average(np.transpose(np.array(gini2_3)),axis=1), delimiter=",")
#gini3_cij_f2.write(f"Average:\n{np.average(np.transpose(np.array(gini2)),axis=1)}\n")
np.savetxt(gini3_cij_f3, np.transpose(gini3_3), delimiter=",")
gini3_cij_f3.write("\n\n\n")
np.savetxt(gini3_cij_f3, np.average(np.transpose(np.array(gini3_3)),axis=1), delimiter=",")
#gini3_cij_f3.write(f"Average:\n{np.average(np.transpose(np.array(gini3)),axis=1)}\n")
np.savetxt(gini3_cij_f4, np.transpose(gini4_3), delimiter=",")
gini3_cij_f4.write("\n\n\n")
np.savetxt(gini3_cij_f4, np.average(np.transpose(np.array(gini4_3)),axis=1), delimiter=",")
#gini3_cij_f4.write(f"Average:\n{np.average(np.transpose(np.array(gini4)),axis=1)}\n")


np.savetxt(gini4_cij_f0, np.transpose(gini0_4), delimiter=",")
gini4_cij_f0.write("\n\n\n")
np.savetxt(gini4_cij_f0, np.average(np.transpose(np.array(gini0_4)),axis=1), delimiter=",")
#gini4_cij_f0.write(f"Average:\n {np.average(np.transpose(np.array(gini0)),axis=1)} \n")
np.savetxt(gini4_cij_f1, np.transpose(gini1_4), delimiter=",")
gini4_cij_f1.write("\n\n\n")
np.savetxt(gini4_cij_f1, np.average(np.transpose(np.array(gini1_4)),axis=1), delimiter=",")
#gini4_cij_f1.write(f"Average:\n{np.average(np.transpose(np.array(gini1)),axis=1)}\n")
np.savetxt(gini4_cij_f2, np.transpose(gini2_4), delimiter=",")
gini4_cij_f2.write("\n\n\n")
np.savetxt(gini4_cij_f2, np.average(np.transpose(np.array(gini2_4)),axis=1), delimiter=",")
#gini4_cij_f2.write(f"Average:\n{np.average(np.transpose(np.array(gini2)),axis=1)}\n")
np.savetxt(gini4_cij_f3, np.transpose(gini3_4), delimiter=",")
gini4_cij_f3.write("\n\n\n")
np.savetxt(gini4_cij_f3, np.average(np.transpose(np.array(gini3_4)),axis=1), delimiter=",")
#gini4_cij_f3.write(f"Average:\n{np.average(np.transpose(np.array(gini3)),axis=1)}\n")
np.savetxt(gini4_cij_f4, np.transpose(gini4_4), delimiter=",")
gini4_cij_f4.write("\n\n\n")
np.savetxt(gini4_cij_f4, np.average(np.transpose(np.array(gini4_4)),axis=1), delimiter=",")
#gini4_cij_f4.write(f"Average:\n{np.average(np.transpose(np.array(gini4)),axis=1)}\n")

print("... gini done!")



np.savetxt(mcls1_cij_f0, np.transpose(mcls0_1), delimiter=",")
mcls1_cij_f0.write("\n\n\n")
np.savetxt(mcls1_cij_f0, np.average(np.transpose(np.array(mcls0_1)),axis=1), delimiter=",")
#mcls1_cij_f0.write(f"Average:\n {np.average(np.transpose(np.array(mcls0)),axis=1)} \n")
np.savetxt(mcls1_cij_f1, np.transpose(mcls1_1), delimiter=",")
mcls1_cij_f1.write("\n\n\n")
np.savetxt(mcls1_cij_f1, np.average(np.transpose(np.array(mcls1_1)),axis=1), delimiter=",")
#mcls1_cij_f1.write(f"Average:\n{np.average(np.transpose(np.array(mcls1)),axis=1)}\n")
np.savetxt(mcls1_cij_f2, np.transpose(mcls2_1), delimiter=",")
mcls1_cij_f2.write("\n\n\n")
np.savetxt(mcls1_cij_f2, np.average(np.transpose(np.array(mcls2_1)),axis=1), delimiter=",")
#mcls1_cij_f2.write(f"Average:\n{np.average(np.transpose(np.array(mcls2)),axis=1)}\n")
np.savetxt(mcls1_cij_f3, np.transpose(mcls3_1), delimiter=",")
mcls1_cij_f3.write("\n\n\n")
np.savetxt(mcls1_cij_f3, np.average(np.transpose(np.array(mcls3_1)),axis=1), delimiter=",")
#mcls1_cij_f3.write(f"Average:\n{np.average(np.transpose(np.array(mcls3)),axis=1)}\n")
np.savetxt(mcls1_cij_f4, np.transpose(mcls4_1), delimiter=",")
mcls1_cij_f4.write("\n\n\n")
np.savetxt(mcls1_cij_f4, np.average(np.transpose(np.array(mcls4_1)),axis=1), delimiter=",")
#mcls1_cij_f4.write(f"Average:\n{np.average(np.transpose(np.array(mcls4)),axis=1)}\n")


np.savetxt(mcls2_cij_f0, np.transpose(mcls0_2), delimiter=",")
mcls2_cij_f0.write("\n\n\n")
np.savetxt(mcls2_cij_f0, np.average(np.transpose(np.array(mcls0_2)),axis=1), delimiter=",")
#mcls2_cij_f0.write(f"Average:\n {np.average(np.transpose(np.array(mcls0)),axis=1)} \n")
np.savetxt(mcls2_cij_f1, np.transpose(mcls1_2), delimiter=",")
mcls2_cij_f1.write("\n\n\n")
np.savetxt(mcls2_cij_f1, np.average(np.transpose(np.array(mcls1_2)),axis=1), delimiter=",")
#mcls2_cij_f1.write(f"Average:\n{np.average(np.transpose(np.array(mcls1)),axis=1)}\n")
np.savetxt(mcls2_cij_f2, np.transpose(mcls2_2), delimiter=",")
mcls2_cij_f2.write("\n\n\n")
np.savetxt(mcls2_cij_f2, np.average(np.transpose(np.array(mcls2_2)),axis=1), delimiter=",")
#mcls2_cij_f2.write(f"Average:\n{np.average(np.transpose(np.array(mcls2)),axis=1)}\n")
np.savetxt(mcls2_cij_f3, np.transpose(mcls3_2), delimiter=",")
mcls2_cij_f3.write("\n\n\n")
np.savetxt(mcls2_cij_f3, np.average(np.transpose(np.array(mcls3_2)),axis=1), delimiter=",")
#mcls2_cij_f3.write(f"Average:\n{np.average(np.transpose(np.array(mcls3)),axis=1)}\n")
np.savetxt(mcls2_cij_f4, np.transpose(mcls4_2), delimiter=",")
mcls2_cij_f4.write("\n\n\n")
np.savetxt(mcls2_cij_f4, np.average(np.transpose(np.array(mcls4_2)),axis=1), delimiter=",")
#mcls2_cij_f4.write(f"Average:\n{np.average(np.transpose(np.array(mcls4)),axis=1)}\n")


np.savetxt(mcls3_cij_f0, np.transpose(mcls0_3), delimiter=",")
mcls3_cij_f0.write("\n\n\n")
np.savetxt(mcls3_cij_f0, np.average(np.transpose(np.array(mcls0_3)),axis=1), delimiter=",")
#mcls3_cij_f0.write(f"Average:\n {np.average(np.transpose(np.array(mcls0)),axis=1)} \n")
np.savetxt(mcls3_cij_f1, np.transpose(mcls1_3), delimiter=",")
mcls3_cij_f1.write("\n\n\n")
np.savetxt(mcls3_cij_f1, np.average(np.transpose(np.array(mcls1_3)),axis=1), delimiter=",")
#mcls3_cij_f1.write(f"Average:\n{np.average(np.transpose(np.array(mcls1)),axis=1)}\n")
np.savetxt(mcls3_cij_f2, np.transpose(mcls2_3), delimiter=",")
mcls3_cij_f2.write("\n\n\n")
np.savetxt(mcls3_cij_f2, np.average(np.transpose(np.array(mcls2_3)),axis=1), delimiter=",")
#mcls3_cij_f2.write(f"Average:\n{np.average(np.transpose(np.array(mcls2)),axis=1)}\n")
np.savetxt(mcls3_cij_f3, np.transpose(mcls3_3), delimiter=",")
mcls3_cij_f3.write("\n\n\n")
np.savetxt(mcls3_cij_f3, np.average(np.transpose(np.array(mcls3_3)),axis=1), delimiter=",")
#mcls3_cij_f3.write(f"Average:\n{np.average(np.transpose(np.array(mcls3)),axis=1)}\n")
np.savetxt(mcls3_cij_f4, np.transpose(mcls4_3), delimiter=",")
mcls3_cij_f4.write("\n\n\n")
np.savetxt(mcls3_cij_f4, np.average(np.transpose(np.array(mcls4_3)),axis=1), delimiter=",")
#mcls3_cij_f4.write(f"Average:\n{np.average(np.transpose(np.array(mcls4)),axis=1)}\n")


np.savetxt(mcls4_cij_f0, np.transpose(mcls0_4), delimiter=",")
mcls4_cij_f0.write("\n\n\n")
np.savetxt(mcls4_cij_f0, np.average(np.transpose(np.array(mcls0_4)),axis=1), delimiter=",")
#mcls4_cij_f0.write(f"Average:\n {np.average(np.transpose(np.array(mcls0)),axis=1)} \n")
np.savetxt(mcls4_cij_f1, np.transpose(mcls1_4), delimiter=",")
mcls4_cij_f1.write("\n\n\n")
np.savetxt(mcls4_cij_f1, np.average(np.transpose(np.array(mcls1_4)),axis=1), delimiter=",")
#mcls4_cij_f1.write(f"Average:\n{np.average(np.transpose(np.array(mcls1)),axis=1)}\n")
np.savetxt(mcls4_cij_f2, np.transpose(mcls2_4), delimiter=",")
mcls4_cij_f2.write("\n\n\n")
np.savetxt(mcls4_cij_f2, np.average(np.transpose(np.array(mcls2_4)),axis=1), delimiter=",")
#mcls4_cij_f2.write(f"Average:\n{np.average(np.transpose(np.array(mcls2)),axis=1)}\n")
np.savetxt(mcls4_cij_f3, np.transpose(mcls3_4), delimiter=",")
mcls4_cij_f3.write("\n\n\n")
np.savetxt(mcls4_cij_f3, np.average(np.transpose(np.array(mcls3_4)),axis=1), delimiter=",")
#mcls4_cij_f3.write(f"Average:\n{np.average(np.transpose(np.array(mcls3)),axis=1)}\n")
np.savetxt(mcls4_cij_f4, np.transpose(mcls4_4), delimiter=",")
mcls4_cij_f4.write("\n\n\n")
np.savetxt(mcls4_cij_f4, np.average(np.transpose(np.array(mcls4_4)),axis=1), delimiter=",")
#mcls4_cij_f4.write(f"Average:\n{np.average(np.transpose(np.array(mcls4)),axis=1)}\n")

print("... mcls done!")


np.savetxt(entr1_cij_f0, np.transpose(entr0_1), delimiter=",")
entr1_cij_f0.write("\n\n\n")
np.savetxt(entr1_cij_f0, np.average(np.transpose(np.array(entr0_1)),axis=1), delimiter=",")
#entr1_cij_f0.write(f"Average:\n {np.average(np.transpose(np.array(entr0)),axis=1)} \n")
np.savetxt(entr1_cij_f1, np.transpose(entr1_1), delimiter=",")
entr1_cij_f1.write("\n\n\n")
np.savetxt(entr1_cij_f1, np.average(np.transpose(np.array(entr1_1)),axis=1), delimiter=",")
#entr1_cij_f1.write(f"Average:\n{np.average(np.transpose(np.array(entr1)),axis=1)}\n")
np.savetxt(entr1_cij_f2, np.transpose(entr2_1), delimiter=",")
entr1_cij_f2.write("\n\n\n")
np.savetxt(entr1_cij_f2, np.average(np.transpose(np.array(entr2_1)),axis=1), delimiter=",")
#entr1_cij_f2.write(f"Average:\n{np.average(np.transpose(np.array(entr2)),axis=1)}\n")
np.savetxt(entr1_cij_f3, np.transpose(entr3_1), delimiter=",")
entr1_cij_f3.write("\n\n\n")
np.savetxt(entr1_cij_f3, np.average(np.transpose(np.array(entr3_1)),axis=1), delimiter=",")
#entr1_cij_f3.write(f"Average:\n{np.average(np.transpose(np.array(entr3)),axis=1)}\n")
np.savetxt(entr1_cij_f4, np.transpose(entr4_1), delimiter=",")
entr1_cij_f4.write("\n\n\n")
np.savetxt(entr1_cij_f4, np.average(np.transpose(np.array(entr4_1)),axis=1), delimiter=",")
#entr1_cij_f4.write(f"Average:\n{np.average(np.transpose(np.array(entr4)),axis=1)}\n")

np.savetxt(entr2_cij_f0, np.transpose(entr0_2), delimiter=",")
entr2_cij_f0.write("\n\n\n")
np.savetxt(entr2_cij_f0, np.average(np.transpose(np.array(entr0_2)),axis=1), delimiter=",")
#entr2_cij_f0.write(f"Average:\n {np.average(np.transpose(np.array(entr0)),axis=1)} \n")
np.savetxt(entr2_cij_f1, np.transpose(entr1_2), delimiter=",")
entr2_cij_f1.write("\n\n\n")
np.savetxt(entr2_cij_f1, np.average(np.transpose(np.array(entr1_2)),axis=1), delimiter=",")
#entr2_cij_f1.write(f"Average:\n{np.average(np.transpose(np.array(entr1)),axis=1)}\n")
np.savetxt(entr2_cij_f2, np.transpose(entr2_2), delimiter=",")
entr2_cij_f2.write("\n\n\n")
np.savetxt(entr2_cij_f2, np.average(np.transpose(np.array(entr2_2)),axis=1), delimiter=",")
#entr2_cij_f2.write(f"Average:\n{np.average(np.transpose(np.array(entr2)),axis=1)}\n")
np.savetxt(entr2_cij_f3, np.transpose(entr3_2), delimiter=",")
entr2_cij_f3.write("\n\n\n")
np.savetxt(entr2_cij_f3, np.average(np.transpose(np.array(entr3_2)),axis=1), delimiter=",")
#entr2_cij_f3.write(f"Average:\n{np.average(np.transpose(np.array(entr3)),axis=1)}\n")
np.savetxt(entr2_cij_f4, np.transpose(entr4_2), delimiter=",")
entr2_cij_f4.write("\n\n\n")
np.savetxt(entr2_cij_f4, np.average(np.transpose(np.array(entr4_2)),axis=1), delimiter=",")
#entr2_cij_f4.write(f"Average:\n{np.average(np.transpose(np.array(entr4)),axis=1)}\n")


np.savetxt(entr3_cij_f0, np.transpose(entr0_3), delimiter=",")
entr3_cij_f0.write("\n\n\n")
np.savetxt(entr3_cij_f0, np.average(np.transpose(np.array(entr0_3)),axis=1), delimiter=",")
#entr3_cij_f0.write(f"Average:\n {np.average(np.transpose(np.array(entr0)),axis=1)} \n")
np.savetxt(entr3_cij_f1, np.transpose(entr1_3), delimiter=",")
entr3_cij_f1.write("\n\n\n")
np.savetxt(entr3_cij_f1, np.average(np.transpose(np.array(entr1_3)),axis=1), delimiter=",")
#entr3_cij_f1.write(f"Average:\n{np.average(np.transpose(np.array(entr1)),axis=1)}\n")
np.savetxt(entr3_cij_f2, np.transpose(entr2_3), delimiter=",")
entr3_cij_f2.write("\n\n\n")
np.savetxt(entr3_cij_f2, np.average(np.transpose(np.array(entr2_3)),axis=1), delimiter=",")
#entr3_cij_f2.write(f"Average:\n{np.average(np.transpose(np.array(entr2)),axis=1)}\n")
np.savetxt(entr3_cij_f3, np.transpose(entr3_3), delimiter=",")
entr3_cij_f3.write("\n\n\n")
np.savetxt(entr3_cij_f3, np.average(np.transpose(np.array(entr3_3)),axis=1), delimiter=",")
#entr3_cij_f3.write(f"Average:\n{np.average(np.transpose(np.array(entr3)),axis=1)}\n")
np.savetxt(entr3_cij_f4, np.transpose(entr4_3), delimiter=",")
entr3_cij_f4.write("\n\n\n")
np.savetxt(entr3_cij_f4, np.average(np.transpose(np.array(entr4_3)),axis=1), delimiter=",")
#entr3_cij_f4.write(f"Average:\n{np.average(np.transpose(np.array(entr4)),axis=1)}\n")


np.savetxt(entr4_cij_f0, np.transpose(entr0_4), delimiter=",")
entr4_cij_f0.write("\n\n\n")
np.savetxt(entr4_cij_f0, np.average(np.transpose(np.array(entr0_4)),axis=1), delimiter=",")
#entr4_cij_f0.write(f"Average:\n {np.average(np.transpose(np.array(entr0)),axis=1)} \n")
np.savetxt(entr4_cij_f1, np.transpose(entr1_4), delimiter=",")
entr4_cij_f1.write("\n\n\n")
np.savetxt(entr4_cij_f1, np.average(np.transpose(np.array(entr1_4)),axis=1), delimiter=",")
#entr4_cij_f1.write(f"Average:\n{np.average(np.transpose(np.array(entr1)),axis=1)}\n")
np.savetxt(entr4_cij_f2, np.transpose(entr2_4), delimiter=",")
entr4_cij_f2.write("\n\n\n")
np.savetxt(entr4_cij_f2, np.average(np.transpose(np.array(entr2_4)),axis=1), delimiter=",")
#entr4_cij_f2.write(f"Average:\n{np.average(np.transpose(np.array(entr2)),axis=1)}\n")
np.savetxt(entr4_cij_f3, np.transpose(entr3_4), delimiter=",")
entr4_cij_f3.write("\n\n\n")
np.savetxt(entr4_cij_f3, np.average(np.transpose(np.array(entr3_4)),axis=1), delimiter=",")
#entr4_cij_f3.write(f"Average:\n{np.average(np.transpose(np.array(entr3)),axis=1)}\n")
np.savetxt(entr4_cij_f4, np.transpose(entr4_4), delimiter=",")
entr4_cij_f4.write("\n\n\n")
np.savetxt(entr4_cij_f4, np.average(np.transpose(np.array(entr4_4)),axis=1), delimiter=",")
#entr4_cij_f4.write(f"Average:\n{np.average(np.transpose(np.array(entr4)),axis=1)}\n")


print("... entr done!")


##############################################################################3333

entr1_cij_f0.close()
entr1_cij_f1.close()
entr1_cij_f2.close()
entr1_cij_f3.close()
entr1_cij_f4.close()

entr2_cij_f0.close()
entr2_cij_f1.close()
entr2_cij_f2.close()
entr2_cij_f3.close()
entr2_cij_f4.close()

entr3_cij_f0.close()
entr3_cij_f1.close()
entr3_cij_f2.close()
entr3_cij_f3.close()
entr3_cij_f4.close()

entr4_cij_f0.close()
entr4_cij_f1.close()
entr4_cij_f2.close()
entr4_cij_f3.close()
entr4_cij_f4.close()

gini1_cij_f0.close()
gini1_cij_f1.close()
gini1_cij_f2.close()
gini1_cij_f3.close()
gini1_cij_f4.close()

gini2_cij_f0.close()
gini2_cij_f1.close()
gini2_cij_f2.close()
gini2_cij_f3.close()
gini2_cij_f4.close()

gini3_cij_f0.close()
gini3_cij_f1.close()
gini3_cij_f2.close()
gini3_cij_f3.close()
gini3_cij_f4.close()

gini4_cij_f0.close()
gini4_cij_f1.close()
gini4_cij_f2.close()
gini4_cij_f3.close()
gini4_cij_f4.close()


mcls1_cij_f0.close()
mcls1_cij_f1.close()
mcls1_cij_f2.close()
mcls1_cij_f3.close()
mcls1_cij_f4.close()

mcls2_cij_f0.close()
mcls2_cij_f1.close()
mcls2_cij_f2.close()
mcls2_cij_f3.close()
mcls2_cij_f4.close()

mcls3_cij_f0.close()
mcls3_cij_f1.close()
mcls3_cij_f2.close()
mcls3_cij_f3.close()
mcls3_cij_f4.close()

mcls4_cij_f0.close()
mcls4_cij_f1.close()
mcls4_cij_f2.close()
mcls4_cij_f3.close()
mcls4_cij_f4.close()

print("closed all necessary files.")

print(np.min((1/prim_act_sum)*np.array(avg_entropies)), np.max((1/prim_act_sum)*np.array(avg_entropies)))
print(np.min((1/prim_act_sum)*np.array(avg_gini)), np.max((1/prim_act_sum)*np.array(avg_gini)))
print(np.min((1/prim_act_sum)*np.array(avg_mc_loss)), np.max((1/prim_act_sum)*np.array(avg_mc_loss)))

print(np.min((1/prim_act_sum)*np.array(avg_entropies),axis=1), np.max((1/prim_act_sum)*np.array(avg_entropies),axis=1))

res_file = open(args.res_file,'w')

res_file.write(f"acc: {acc}\n\n")
res_file.write(f"entropy: \n")
np.savetxt(res_file,(1/prim_act_sum)*np.array(avg_entropies),delimiter=",")
res_file.write("\n\nMin: \n")
np.savetxt(res_file,np.min((1/prim_act_sum)*np.array(avg_entropies),axis=1),delimiter=",")
res_file.write("\n\nMax: \n")
np.savetxt(res_file,np.max((1/prim_act_sum)*np.array(avg_entropies),axis=1),delimiter=",")
np.savetxt(res_file,(1/prim_act_sum)*np.array(avg_entropies),delimiter=",")
res_file.write("\n\nMin: \n")
np.savetxt(res_file,np.min((1/prim_act_sum)*np.array(avg_entropies),axis=0),delimiter=",")
res_file.write("\n\nMax: \n")
np.savetxt(res_file,np.max((1/prim_act_sum)*np.array(avg_entropies),axis=0),delimiter=",")



res_file.write(f"\n\n\n\ngini: \n")
np.savetxt(res_file,(1/prim_act_sum)*np.array(avg_gini),delimiter=",")
res_file.write("\n\nMin: \n")
np.savetxt(res_file,np.min((1/prim_act_sum)*np.array(avg_gini),axis=1),delimiter=",")
res_file.write("\n\nMax: \n")
np.savetxt(res_file,np.max((1/prim_act_sum)*np.array(avg_gini),axis=1),delimiter=",")
np.savetxt(res_file,(1/prim_act_sum)*np.array(avg_entropies),delimiter=",")
res_file.write("\n\nMin: \n")
np.savetxt(res_file,np.min((1/prim_act_sum)*np.array(avg_gini),axis=0),delimiter=",")
res_file.write("\n\nMax: \n")
np.savetxt(res_file,np.max((1/prim_act_sum)*np.array(avg_gini),axis=0),delimiter=",")

res_file.write(f"\n\n\n\nmc_loss: \n")
np.savetxt(res_file,(1/prim_act_sum)*np.array(avg_mc_loss),delimiter=",")
res_file.write("\n\nMin: \n")
np.savetxt(res_file,np.min((1/prim_act_sum)*np.array(avg_mc_loss),axis=1),delimiter=",")
res_file.write("\n\nMax: \n")
np.savetxt(res_file,np.max((1/prim_act_sum)*np.array(avg_mc_loss),axis=1),delimiter=",")
np.savetxt(res_file,(1/prim_act_sum)*np.array(avg_entropies),delimiter=",")
res_file.write("\n\nMin: \n")
np.savetxt(res_file,np.min((1/prim_act_sum)*np.array(avg_mc_loss),axis=0),delimiter=",")
res_file.write("\n\nMax: \n")
np.savetxt(res_file,np.max((1/prim_act_sum)*np.array(avg_mc_loss),axis=0),delimiter=",")


res_file.close()

'''
with open(args.res_file, "a") as fl:
    fl.write(f"entropies: {avg_entropies} and avg_entropy {(1/prim_act_sum)*np.array(avg_entropies)}\n")
    fl.write(f"ginis: {avg_gini} and avg_gini {(1/prim_act_sum)*np.array(avg_gini)}\n")
    fl.write(f"mc_losses: {avg_mc_loss} and avg_mc_loss {(1/prim_act_sum)*np.array(avg_mc_loss)}\n")
    fl.write(f"Accuracy: {acc}")
fl.close()
'''
