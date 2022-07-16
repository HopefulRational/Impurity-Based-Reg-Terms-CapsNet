from utils import *
from arch import *

parser = argparse.ArgumentParser(description='Logging for entropy-regularised CapsNet')
parser.add_argument('--chk_file', metavar='CHKPNT_FILE', type=str, default='./checkpoints/MNIST/CapsNet/chkpoint_adam_ExpLR_run1.pth', help='checkpoint file')
parser.add_argument('--imp_cij_folder', type=str, default='../analysis/impurities_cijs', help='folder for impurity values for c_ij')
parser.add_argument('--conf_file', metavar='COFUSION_FILE', type=str, default='../../confusion.csv', help='confusion_matrix file')
parser.add_argument('--res_file', metavar='RESULTS_FILE', type=str, default='../results.txt', help='results.txt file')
parser.add_argument('--reg_type', metavar='REG_TYPE', type=str, default='unreg', help='type of reg')
args = parser.parse_args()

today = date.today()

with open(args.res_file, "a") as fl:
    fl.write('\n\n'+args.chk_file+'\n'+str(today)+'\n')
fl.close()

with open(args.conf_file, "a") as conf_fl:
    conf_fl.write('\n\n'+args.chk_file+'\n'+str(today)+'\n')
conf_fl.close()

conf_fl = open(args.conf_file, "a")


model = CapsNet().to(DEVICE)
chk = torch.load(args.chk_file)
acc = chk['acc']
model.load_state_dict(chk['model'])
dset = torchvision.datasets.FashionMNIST(root='../../data', download=False, train=False, transform=transforms.ToTensor())
batch_size = 100
loader = torch.utils.data.DataLoader(dset,batch_size=batch_size, shuffle=False)
model.eval()

f0 = open('../../primary_activations/' + args.reg_type + '/0.csv', 'w')
f1 = open('../../primary_activations/' + args.reg_type + '/1.csv', 'w')
f2 = open('../../primary_activations/' + args.reg_type + '/2.csv', 'w')
f3 = open('../../primary_activations/' + args.reg_type + '/3.csv', 'w')
f4 = open('../../primary_activations/' + args.reg_type + '/4.csv', 'w')
f5 = open('../../primary_activations/' + args.reg_type + '/5.csv', 'w')
f6 = open('../../primary_activations/' + args.reg_type + '/6.csv', 'w')
f7 = open('../../primary_activations/' + args.reg_type + '/7.csv', 'w')
f8 = open('../../primary_activations/' + args.reg_type + '/8.csv', 'w')
f9 = open('../../primary_activations/' + args.reg_type + '/9.csv', 'w')

''' predictions: file: preds_f0,.. '''
pred0,pred1,pred2,pred3,pred4,pred5,pred6,pred7,pred8,pred9 = [],[],[],[],[],[],[],[],[],[]

''' cij(routing coeff): file: Class_f0,.. '''
cij0,cij1,cij2,cij3,cij4,cij5,cij6,cij7,cij8,cij9 = [],[],[],[],[],[],[],[],[],[]

''' calss_acts: file: class_f0,... '''
class_l0,class_l1,class_l2,class_l3,class_l4,class_l5,class_l6,class_l7,class_l8,class_l9 = [],[],[],[],[],[],[],[],[],[]

''' primary_acts: file: f0,... '''
l0,l1,l2,l3,l4,l5,l6,l7,l8,l9 = [],[],[],[],[],[],[],[],[],[]

''' entr  '''
entr0, entr1, entr2, entr3, entr4, entr5, entr6, entr7, entr8, entr9 = [],[],[],[],[],[],[],[],[],[]

''' gini  '''
gini0, gini1, gini2, gini3, gini4, gini5, gini6, gini7, gini8, gini9 = [],[],[],[],[],[],[],[],[],[]

''' mcls  '''
mcls0, mcls1, mcls2, mcls3, mcls4, mcls5, mcls6, mcls7, mcls8, mcls9 = [],[],[],[],[],[],[],[],[],[]

''' files: [preds_f0, Class_f0, class_f0, f0, entr, gini, mcls] - 7 + 1(confusion_mat)''' 

PRIMARY_ACTS = 0
CLASS_ACTS = 1
ENTR = 2
GINI = 3
MCLS = 4
C_IJ = 5
PRED = 6

def switchFile(num):
    switcher = {
        0:[l0, class_l0, entr0, gini0, mcls0, cij0, pred0], 1:[l1, class_l1, entr1, gini1, mcls1, cij1, pred1], 2:[l2, class_l2, entr2, gini2, mcls2, cij2, pred2], 
        3:[l3, class_l3, entr3, gini3, mcls3, cij3, pred3], 4:[l4, class_l4, entr4, gini4, mcls4, cij4, pred4], 5:[l5, class_l5, entr5, gini5, mcls5, cij5, pred5], 
        6:[l6, class_l6, entr6, gini6, mcls6, cij6, pred6], 7:[l7, class_l7, entr7, gini7, mcls7, cij7, pred7], 8:[l8, class_l8, entr8, gini8, mcls8, cij8, pred8], 
        9:[l9, class_l9, entr9, gini9, mcls9, cij9, pred9],
    }
    return switcher.get(num)


entr_cij_f0 = open(args.imp_cij_folder + '/entr' + '/0.csv', 'w')
entr_cij_f1 = open(args.imp_cij_folder + '/entr' + '/1.csv', 'w')
entr_cij_f2 = open(args.imp_cij_folder + '/entr' + '/2.csv', 'w')
entr_cij_f3 = open(args.imp_cij_folder + '/entr' + '/3.csv', 'w')
entr_cij_f4 = open(args.imp_cij_folder + '/entr' + '/4.csv', 'w')
entr_cij_f5 = open(args.imp_cij_folder + '/entr' + '/5.csv', 'w')
entr_cij_f6 = open(args.imp_cij_folder + '/entr' + '/6.csv', 'w')
entr_cij_f7 = open(args.imp_cij_folder + '/entr' + '/7.csv', 'w')
entr_cij_f8 = open(args.imp_cij_folder + '/entr' + '/8.csv', 'w')
entr_cij_f9 = open(args.imp_cij_folder + '/entr' + '/9.csv', 'w')

gini_cij_f0 = open(args.imp_cij_folder + '/gini' + '/0.csv', 'w')
gini_cij_f1 = open(args.imp_cij_folder + '/gini' + '/1.csv', 'w')
gini_cij_f2 = open(args.imp_cij_folder + '/gini' + '/2.csv', 'w')
gini_cij_f3 = open(args.imp_cij_folder + '/gini' + '/3.csv', 'w')
gini_cij_f4 = open(args.imp_cij_folder + '/gini' + '/4.csv', 'w')
gini_cij_f5 = open(args.imp_cij_folder + '/gini' + '/5.csv', 'w')
gini_cij_f6 = open(args.imp_cij_folder + '/gini' + '/6.csv', 'w')
gini_cij_f7 = open(args.imp_cij_folder + '/gini' + '/7.csv', 'w')
gini_cij_f8 = open(args.imp_cij_folder + '/gini' + '/8.csv', 'w')
gini_cij_f9 = open(args.imp_cij_folder + '/gini' + '/9.csv', 'w')

mcls_cij_f0 = open(args.imp_cij_folder + '/mcls' + '/0.csv', 'w')
mcls_cij_f1 = open(args.imp_cij_folder + '/mcls' + '/1.csv', 'w')
mcls_cij_f2 = open(args.imp_cij_folder + '/mcls' + '/2.csv', 'w')
mcls_cij_f3 = open(args.imp_cij_folder + '/mcls' + '/3.csv', 'w')
mcls_cij_f4 = open(args.imp_cij_folder + '/mcls' + '/4.csv', 'w')
mcls_cij_f5 = open(args.imp_cij_folder + '/mcls' + '/5.csv', 'w')
mcls_cij_f6 = open(args.imp_cij_folder + '/mcls' + '/6.csv', 'w')
mcls_cij_f7 = open(args.imp_cij_folder + '/mcls' + '/7.csv', 'w')
mcls_cij_f8 = open(args.imp_cij_folder + '/mcls' + '/8.csv', 'w')
mcls_cij_f9 = open(args.imp_cij_folder + '/mcls' + '/9.csv', 'w')


c_ij_path = '../analysis/cij_files'
preds_path = '../analysis/predictions'

preds_f0 = open(preds_path + '/0.csv', 'w')
preds_f1 = open(preds_path + '/1.csv', 'w')
preds_f2 = open(preds_path + '/2.csv', 'w')
preds_f3 = open(preds_path + '/3.csv', 'w')
preds_f4 = open(preds_path + '/4.csv', 'w')
preds_f5 = open(preds_path + '/5.csv', 'w')
preds_f6 = open(preds_path + '/6.csv', 'w')
preds_f7 = open(preds_path + '/7.csv', 'w')
preds_f8 = open(preds_path + '/8.csv', 'w')
preds_f9 = open(preds_path + '/9.csv', 'w')


Class_f0 = open(c_ij_path + '/0.csv', 'w')
Class_f1 = open(c_ij_path + '/1.csv', 'w')
Class_f2 = open(c_ij_path + '/2.csv', 'w')
Class_f3 = open(c_ij_path + '/3.csv', 'w')
Class_f4 = open(c_ij_path + '/4.csv', 'w')
Class_f5 = open(c_ij_path + '/5.csv', 'w')
Class_f6 = open(c_ij_path + '/6.csv', 'w')
Class_f7 = open(c_ij_path + '/7.csv', 'w')
Class_f8 = open(c_ij_path + '/8.csv', 'w')
Class_f9 = open(c_ij_path + '/9.csv', 'w')


class_f0 = open('../../class_caps_activations/' + args.reg_type + '/0.csv', 'w')
class_f1 = open('../../class_caps_activations/' + args.reg_type + '/1.csv', 'w')
class_f2 = open('../../class_caps_activations/' + args.reg_type + '/2.csv', 'w')
class_f3 = open('../../class_caps_activations/' + args.reg_type + '/3.csv', 'w')
class_f4 = open('../../class_caps_activations/' + args.reg_type + '/4.csv', 'w')
class_f5 = open('../../class_caps_activations/' + args.reg_type + '/5.csv', 'w')
class_f6 = open('../../class_caps_activations/' + args.reg_type + '/6.csv', 'w')
class_f7 = open('../../class_caps_activations/' + args.reg_type + '/7.csv', 'w')
class_f8 = open('../../class_caps_activations/' + args.reg_type + '/8.csv', 'w')
class_f9 = open('../../class_caps_activations/' + args.reg_type + '/9.csv', 'w')





# For impurities for c_ij per_class
entr0, entr1, entr2, entr3, entr4, entr5, entr6, entr7, entr8, entr9 = [],[],[],[],[],[],[],[],[],[]
gini0, gini1, gini2, gini3, gini4, gini5, gini6, gini7, gini8, gini9 = [],[],[],[],[],[],[],[],[],[]
mcls0, mcls1, mcls2, mcls3, mcls4, mcls5, mcls6, mcls7, mcls8, mcls9 = [],[],[],[],[],[],[],[],[],[]



conf_matrix = [[0 for _ in range(10)] for _ in range(10)]

avg_entropies, avg_gini, avg_mc_loss = [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0]

prim_act_sum = [[0. for _ in range(1152)] for _ in range(10)]

with torch.no_grad():
     for batch_idx, (data, lbls) in enumerate(loader):
         data, labels = data.to(DEVICE), one_hot(lbls).to(DEVICE)
         output, masked_output, recnstrcted, c_ij, primary_activations, u_hat_activations = model(data)
         digit_out_activations = torch.norm(output.squeeze(), dim=-1, keepdim=False)
         #print(f"SAI: {output.shape} and {digit_out_activations.shape}")
         #print(f"\nc_ij.shape {c_ij.shape} | u_hat_activations.shape {u_hat_activations.shape}")
         #break
         for btch_elem in range(batch_size):
             switchFile(torch.max(labels[btch_elem], dim=0)[1].item())[C_IJ].append(c_ij[btch_elem].detach().cpu().numpy())
             switchFile(torch.max(labels[btch_elem], dim=0)[1].item())[GINI].append(get_ginis(c_ij[btch_elem]).squeeze().detach().cpu().numpy())
             switchFile(torch.max(labels[btch_elem], dim=0)[1].item())[MCLS].append(get_mcLosses(c_ij[btch_elem]).squeeze().detach().cpu().numpy())
             switchFile(torch.max(labels[btch_elem], dim=0)[1].item())[ENTR].append(get_entropies(c_ij[btch_elem]).squeeze().detach().cpu().numpy())
             switchFile(torch.max(labels[btch_elem], dim=0)[1].item())[CLASS_ACTS].append(digit_out_activations[btch_elem].detach().cpu().numpy())
             switchFile(torch.max(labels[btch_elem], dim=0)[1].item())[PRIMARY_ACTS].append(primary_activations[btch_elem].detach().cpu().numpy())
             switchFile(torch.max(labels[btch_elem], dim=0)[1].item())[PRED].append(u_hat_activations[btch_elem].detach().cpu().numpy())
             
             conf_matrix[torch.max(labels[btch_elem], dim=0)[1].item()][torch.max(digit_out_activations[btch_elem], dim=0)[1].item()] += 1
             
             avg_entropies[torch.max(labels[btch_elem], dim=0)[1].item()] += (primary_activations[btch_elem]*get_entropies(c_ij[btch_elem]).squeeze()).detach().cpu().numpy()
             avg_gini[torch.max(labels[btch_elem], dim=0)[1].item()] += (primary_activations[btch_elem]*get_ginis(c_ij[btch_elem]).squeeze()).detach().cpu().numpy()
             avg_mc_loss[torch.max(labels[btch_elem], dim=0)[1].item()] += (primary_activations[btch_elem]*get_mcLosses(c_ij[btch_elem]).squeeze()).detach().cpu().numpy()
             
             prim_act_sum[torch.max(labels[btch_elem], dim=0)[1].item()] += primary_activations[btch_elem].detach().cpu().numpy()

prim_act_sum = np.array(prim_act_sum)
print(np.min((1/prim_act_sum)*np.array(avg_entropies),axis=1), np.max((1/prim_act_sum)*np.array(avg_entropies),axis=1))
#assert False
np.savetxt(conf_fl, conf_matrix, delimiter=",")
conf_fl.write("\n\n\n")
conf_fl.close()

for i in range(1000):
	np.savetxt(Class_f0, np.array(cij0[i].squeeze()), delimiter=",")
	np.savetxt(Class_f1, np.array(cij1[i].squeeze()), delimiter=",")
	np.savetxt(Class_f2, np.array(cij2[i].squeeze()), delimiter=",")
	np.savetxt(Class_f3, np.array(cij3[i].squeeze()), delimiter=",")
	np.savetxt(Class_f4, np.array(cij4[i].squeeze()), delimiter=",")
	np.savetxt(Class_f5, np.array(cij5[i].squeeze()), delimiter=",")
	np.savetxt(Class_f6, np.array(cij6[i].squeeze()), delimiter=",")
	np.savetxt(Class_f7, np.array(cij7[i].squeeze()), delimiter=",")
	np.savetxt(Class_f8, np.array(cij8[i].squeeze()), delimiter=",")
	np.savetxt(Class_f9, np.array(cij9[i].squeeze()), delimiter=",")

	np.savetxt(preds_f0, np.array(pred0[i]), delimiter=",")
	np.savetxt(preds_f1, np.array(pred1[i]), delimiter=",")
	np.savetxt(preds_f2, np.array(pred2[i]), delimiter=",")
	np.savetxt(preds_f3, np.array(pred3[i]), delimiter=",")
	np.savetxt(preds_f4, np.array(pred4[i]), delimiter=",")
	np.savetxt(preds_f5, np.array(pred5[i]), delimiter=",")
	np.savetxt(preds_f6, np.array(pred6[i]), delimiter=",")
	np.savetxt(preds_f7, np.array(pred7[i]), delimiter=",")
	np.savetxt(preds_f8, np.array(pred8[i]), delimiter=",")
	np.savetxt(preds_f9, np.array(pred9[i]), delimiter=",")

print("... cijs, preds done!")
'''
for i in range(10):
    switchFile(i)[ENTR].append(np.mean(np.array(switchFile(i)[ENTR], axis=0)))
    switchFile(i)[GINI].append(np.mean(np.array(switchFile(i)[GINI], axis=0)))
    switchFile(i)[MCLS].append(np.mean(np.array(switchFile(i)[MCLS], axis=0)))
'''

#print(f"SAI: {np.transpose(entr0).shape}")

np.savetxt(gini_cij_f0, np.transpose(gini0), delimiter=",")
gini_cij_f0.write("\n\n\n")
np.savetxt(gini_cij_f0, np.average(np.transpose(np.array(gini0)),axis=1), delimiter=",")
#gini_cij_f0.write(f"Average:\n {np.average(np.transpose(np.array(gini0)),axis=1)} \n")
print(f"gini0: {len(gini0)}...done")
np.savetxt(gini_cij_f1, np.transpose(gini1), delimiter=",")
gini_cij_f1.write("\n\n\n")
np.savetxt(gini_cij_f1, np.average(np.transpose(np.array(gini1)),axis=1), delimiter=",")
#gini_cij_f1.write(f"Average:\n{np.average(np.transpose(np.array(gini1)),axis=1)}\n")
print(f"gini1: {len(gini1)}...done")
np.savetxt(gini_cij_f2, np.transpose(gini2), delimiter=",")
gini_cij_f2.write("\n\n\n")
np.savetxt(gini_cij_f2, np.average(np.transpose(np.array(gini2)),axis=1), delimiter=",")
#gini_cij_f2.write(f"Average:\n{np.average(np.transpose(np.array(gini2)),axis=1)}\n")
print(f"gini2: {len(gini2)}...done")
np.savetxt(gini_cij_f3, np.transpose(gini3), delimiter=",")
gini_cij_f3.write("\n\n\n")
np.savetxt(gini_cij_f3, np.average(np.transpose(np.array(gini3)),axis=1), delimiter=",")
#gini_cij_f3.write(f"Average:\n{np.average(np.transpose(np.array(gini3)),axis=1)}\n")
print(f"gini3: {len(gini3)}...done")
np.savetxt(gini_cij_f4, np.transpose(gini4), delimiter=",")
gini_cij_f4.write("\n\n\n")
np.savetxt(gini_cij_f4, np.average(np.transpose(np.array(gini4)),axis=1), delimiter=",")
#gini_cij_f4.write(f"Average:\n{np.average(np.transpose(np.array(gini4)),axis=1)}\n")
print(f"gini4: {len(gini4)}...done")
np.savetxt(gini_cij_f5, np.transpose(gini5), delimiter=",")
gini_cij_f5.write("\n\n\n")
np.savetxt(gini_cij_f5, np.average(np.transpose(np.array(gini5)),axis=1), delimiter=",")
#gini_cij_f5.write(f"Average:\n{np.average(np.transpose(np.array(gini5)),axis=1)}\n")
print(f"gini5: {len(gini5)}...done")
np.savetxt(gini_cij_f6, np.transpose(gini6), delimiter=",")
gini_cij_f6.write("\n\n\n")
np.savetxt(gini_cij_f6, np.average(np.transpose(np.array(gini6)),axis=1), delimiter=",")
#gini_cij_f6.write(f"Average:\n{np.average(np.transpose(np.array(gini6)),axis=1)}\n")
print(f"gini6: {len(gini6)}...done")
np.savetxt(gini_cij_f7, np.transpose(gini7), delimiter=",")
gini_cij_f7.write("\n\n\n")
np.savetxt(gini_cij_f7, np.average(np.transpose(np.array(gini7)),axis=1), delimiter=",")
#gini_cij_f7.write(f"Average:\n{np.average(np.transpose(np.array(gini7)),axis=1)}\n")
print(f"gini7: {len(gini7)}...done")
np.savetxt(gini_cij_f8, np.transpose(gini8), delimiter=",")
gini_cij_f8.write("\n\n\n")
np.savetxt(gini_cij_f8, np.average(np.transpose(np.array(gini8)),axis=1), delimiter=",")
#gini_cij_f8.write(f"Average:\n{np.average(np.transpose(np.array(gini8)),axis=1)}\n")
print(f"gini8: {len(gini8)}...done")
np.savetxt(gini_cij_f9, np.transpose(gini9), delimiter=",")
gini_cij_f9.write("\n\n\n")
np.savetxt(gini_cij_f9, np.average(np.transpose(np.array(gini9)),axis=1), delimiter=",")
#gini_cij_f9.write(f"Average:\n{np.average(np.transpose(np.array(gini9)),axis=1)}\n")
print(f"gini9: {len(gini9)}...done")

print("... gini done!")

np.savetxt(mcls_cij_f0, np.transpose(mcls0), delimiter=",")
mcls_cij_f0.write("\n\n\n")
np.savetxt(mcls_cij_f0, np.average(np.transpose(np.array(mcls0)),axis=1), delimiter=",")
#mcls_cij_f0.write(f"Average:\n {np.average(np.transpose(np.array(mcls0)),axis=1)} \n")
print(f"mcls0: {len(mcls0)}...done")
np.savetxt(mcls_cij_f1, np.transpose(mcls1), delimiter=",")
mcls_cij_f1.write("\n\n\n")
np.savetxt(mcls_cij_f1, np.average(np.transpose(np.array(mcls1)),axis=1), delimiter=",")
#mcls_cij_f1.write(f"Average:\n{np.average(np.transpose(np.array(mcls1)),axis=1)}\n")
print(f"mcls1: {len(mcls1)}...done")
np.savetxt(mcls_cij_f2, np.transpose(mcls2), delimiter=",")
mcls_cij_f2.write("\n\n\n")
np.savetxt(mcls_cij_f2, np.average(np.transpose(np.array(mcls2)),axis=1), delimiter=",")
#mcls_cij_f2.write(f"Average:\n{np.average(np.transpose(np.array(mcls2)),axis=1)}\n")
print(f"mcls2: {len(mcls2)}...done")
np.savetxt(mcls_cij_f3, np.transpose(mcls3), delimiter=",")
mcls_cij_f3.write("\n\n\n")
np.savetxt(mcls_cij_f3, np.average(np.transpose(np.array(mcls3)),axis=1), delimiter=",")
#mcls_cij_f3.write(f"Average:\n{np.average(np.transpose(np.array(mcls3)),axis=1)}\n")
print(f"mcls3: {len(mcls3)}...done")
np.savetxt(mcls_cij_f4, np.transpose(mcls4), delimiter=",")
mcls_cij_f4.write("\n\n\n")
np.savetxt(mcls_cij_f4, np.average(np.transpose(np.array(mcls4)),axis=1), delimiter=",")
#mcls_cij_f4.write(f"Average:\n{np.average(np.transpose(np.array(mcls4)),axis=1)}\n")
print(f"mcls4: {len(mcls4)}...done")
np.savetxt(mcls_cij_f5, np.transpose(mcls5), delimiter=",")
mcls_cij_f5.write("\n\n\n")
np.savetxt(mcls_cij_f5, np.average(np.transpose(np.array(mcls5)),axis=1), delimiter=",")
#mcls_cij_f5.write(f"Average:\n{np.average(np.transpose(np.array(mcls5)),axis=1)}\n")
print(f"mcls5: {len(mcls5)}...done")
np.savetxt(mcls_cij_f6, np.transpose(mcls6), delimiter=",")
mcls_cij_f6.write("\n\n\n")
np.savetxt(mcls_cij_f6, np.average(np.transpose(np.array(mcls6)),axis=1), delimiter=",")
#mcls_cij_f6.write(f"Average:\n{np.average(np.transpose(np.array(mcls6)),axis=1)}\n")
print(f"mcls6: {len(mcls6)}...done")
np.savetxt(mcls_cij_f7, np.transpose(mcls7), delimiter=",")
mcls_cij_f7.write("\n\n\n")
np.savetxt(mcls_cij_f7, np.average(np.transpose(np.array(mcls7)),axis=1), delimiter=",")
#mcls_cij_f7.write(f"Average:\n{np.average(np.transpose(np.array(mcls7)),axis=1)}\n")
print(f"mcls7: {len(mcls7)}...done")
np.savetxt(mcls_cij_f8, np.transpose(mcls8), delimiter=",")
mcls_cij_f8.write("\n\n\n")
np.savetxt(mcls_cij_f8, np.average(np.transpose(np.array(mcls8)),axis=1), delimiter=",")
#mcls_cij_f8.write(f"Average:\n{np.average(np.transpose(np.array(mcls8)),axis=1)}\n")
print(f"mcls8: {len(mcls8)}...done")
np.savetxt(mcls_cij_f9, np.transpose(mcls9), delimiter=",")
mcls_cij_f9.write("\n\n\n")
np.savetxt(mcls_cij_f9, np.average(np.transpose(np.array(mcls9)),axis=1), delimiter=",")
#mcls_cij_f9.write(f"Average:\n{np.average(np.transpose(np.array(mcls9)),axis=1)}\n")
print(f"mcls9: {len(mcls9)}...done")

print("... mcls done!")

np.savetxt(entr_cij_f0, np.transpose(entr0), delimiter=",")
entr_cij_f0.write("\n\n\n")
np.savetxt(entr_cij_f0, np.average(np.transpose(np.array(entr0)),axis=1), delimiter=",")
#entr_cij_f0.write(f"Average:\n {np.average(np.transpose(np.array(entr0)),axis=1)} \n")
print(f"{len(entr0)}...done")
np.savetxt(entr_cij_f1, np.transpose(entr1), delimiter=",")
entr_cij_f1.write("\n\n\n")
np.savetxt(entr_cij_f1, np.average(np.transpose(np.array(entr1)),axis=1), delimiter=",")
#entr_cij_f1.write(f"Average:\n{np.average(np.transpose(np.array(entr1)),axis=1)}\n")
print(f"{len(entr1)}...done")
np.savetxt(entr_cij_f2, np.transpose(entr2), delimiter=",")
entr_cij_f2.write("\n\n\n")
np.savetxt(entr_cij_f2, np.average(np.transpose(np.array(entr2)),axis=1), delimiter=",")
#entr_cij_f2.write(f"Average:\n{np.average(np.transpose(np.array(entr2)),axis=1)}\n")
print(f"{len(entr2)}...done")
np.savetxt(entr_cij_f3, np.transpose(entr3), delimiter=",")
entr_cij_f3.write("\n\n\n")
np.savetxt(entr_cij_f3, np.average(np.transpose(np.array(entr3)),axis=1), delimiter=",")
#entr_cij_f3.write(f"Average:\n{np.average(np.transpose(np.array(entr3)),axis=1)}\n")
print(f"{len(entr3)}...done")
np.savetxt(entr_cij_f4, np.transpose(entr4), delimiter=",")
entr_cij_f4.write("\n\n\n")
np.savetxt(entr_cij_f4, np.average(np.transpose(np.array(entr4)),axis=1), delimiter=",")
#entr_cij_f4.write(f"Average:\n{np.average(np.transpose(np.array(entr4)),axis=1)}\n")
print(f"{len(entr4)}...done")
np.savetxt(entr_cij_f5, np.transpose(entr5), delimiter=",")
entr_cij_f5.write("\n\n\n")
np.savetxt(entr_cij_f5, np.average(np.transpose(np.array(entr5)),axis=1), delimiter=",")
#entr_cij_f5.write(f"Average:\n{np.average(np.transpose(np.array(entr5)),axis=1)}\n")
print(f"{len(entr5)}...done")
np.savetxt(entr_cij_f6, np.transpose(entr6), delimiter=",")
entr_cij_f6.write("\n\n\n")
np.savetxt(entr_cij_f6, np.average(np.transpose(np.array(entr6)),axis=1), delimiter=",")
#entr_cij_f6.write(f"Average:\n{np.average(np.transpose(np.array(entr6)),axis=1)}\n")
print(f"{len(entr6)}...done")
np.savetxt(entr_cij_f7, np.transpose(entr7), delimiter=",")
entr_cij_f7.write("\n\n\n")
np.savetxt(entr_cij_f7, np.average(np.transpose(np.array(entr7)),axis=1), delimiter=",")
#entr_cij_f7.write(f"Average:\n{np.average(np.transpose(np.array(entr7)),axis=1)}\n")
print(f"{len(entr7)}...done")
np.savetxt(entr_cij_f8, np.transpose(entr8), delimiter=",")
entr_cij_f8.write("\n\n\n")
np.savetxt(entr_cij_f8, np.average(np.transpose(np.array(entr8)),axis=1), delimiter=",")
#entr_cij_f8.write(f"Average:\n{np.average(np.transpose(np.array(entr8)),axis=1)}\n")
print(f"{len(entr8)}...done")
np.savetxt(entr_cij_f9, np.transpose(entr9), delimiter=",")
entr_cij_f9.write("\n\n\n")
np.savetxt(entr_cij_f9, np.average(np.transpose(np.array(entr9)),axis=1), delimiter=",")
#entr_cij_f9.write(f"Average:\n{np.average(np.transpose(np.array(entr9)),axis=1)}\n")
print(f"{len(entr9)}...done")

print("... entr done!")


##############################################################################3333

np.savetxt(class_f0, np.transpose(class_l0), delimiter=",")
print(f"class_f0 {len(class_l0)}...done")
np.savetxt(class_f1, np.transpose(class_l1), delimiter=",")
print(f"class_f1 {len(class_l1)}...done")
np.savetxt(class_f2, np.transpose(class_l2), delimiter=",")
print(f"class_f2 {len(class_l2)}...done")
np.savetxt(class_f3, np.transpose(class_l3), delimiter=",")
print(f"class_f3 {len(class_l3)}...done")
np.savetxt(class_f4, np.transpose(class_l4), delimiter=",")
print(f"class_f4 {len(class_l4)}...done")
np.savetxt(class_f5, np.transpose(class_l5), delimiter=",")
print(f"class_f5 {len(class_l5)}...done")
np.savetxt(class_f6, np.transpose(class_l6), delimiter=",")
print(f"class_f6 {len(class_l6)}...done")
np.savetxt(class_f7, np.transpose(class_l7), delimiter=",")
print(f"class_f7 {len(class_l7)}...done")
np.savetxt(class_f8, np.transpose(class_l8), delimiter=",")
print(f"class_f8 {len(class_l8)}...done")
np.savetxt(class_f9, np.transpose(class_l9), delimiter=",")
print(f"class_f9 {len(class_l9)}...done")

print("class_caps_activations done...")

np.savetxt(f0, np.transpose(l0), delimiter=",")
print(f"{len(l0)}...done")
np.savetxt(f1, np.transpose(l1), delimiter=",")
print(f"{len(l1)}...done")
np.savetxt(f2, np.transpose(l2), delimiter=",")
print(f"{len(l2)}...done")
np.savetxt(f3, np.transpose(l3), delimiter=",")
print(f"{len(l3)}...done")
np.savetxt(f4, np.transpose(l4), delimiter=",")
print(f"{len(l4)}...done")
np.savetxt(f5, np.transpose(l5), delimiter=",")
print(f"{len(l5)}...done")
np.savetxt(f6, np.transpose(l6), delimiter=",")
print(f"{len(l6)}...done")
np.savetxt(f7, np.transpose(l7), delimiter=",")
print(f"{len(l7)}...done")
np.savetxt(f8, np.transpose(l8), delimiter=",")
print(f"{len(l8)}...done")
np.savetxt(f9, np.transpose(l9), delimiter=",")
print(f"{len(l9)}...done")

print("primary_activations done...")

preds_f0.close()
preds_f1.close()
preds_f2.close()
preds_f3.close()
preds_f4.close()
preds_f5.close()
preds_f6.close()
preds_f7.close()
preds_f8.close()
preds_f9.close()


entr_cij_f0.close()
entr_cij_f1.close()
entr_cij_f2.close()
entr_cij_f3.close()
entr_cij_f4.close()
entr_cij_f5.close()
entr_cij_f6.close()
entr_cij_f7.close()
entr_cij_f8.close()
entr_cij_f9.close()

gini_cij_f0.close()
gini_cij_f1.close()
gini_cij_f2.close()
gini_cij_f3.close()
gini_cij_f4.close()
gini_cij_f5.close()
gini_cij_f6.close()
gini_cij_f7.close()
gini_cij_f8.close()
gini_cij_f9.close()

mcls_cij_f0.close()
mcls_cij_f1.close()
mcls_cij_f2.close()
mcls_cij_f3.close()
mcls_cij_f4.close()
mcls_cij_f5.close()
mcls_cij_f6.close()
mcls_cij_f7.close()
mcls_cij_f8.close()
mcls_cij_f9.close()

Class_f0.close()
Class_f1.close()
Class_f2.close()
Class_f3.close()
Class_f4.close()
Class_f5.close()
Class_f6.close()
Class_f7.close()
Class_f8.close()
Class_f9.close()

class_f0.close()
class_f1.close()
class_f2.close()
class_f3.close()
class_f4.close()
class_f5.close()
class_f6.close()
class_f7.close()
class_f8.close()
class_f9.close()

f0.close()
f1.close()
f2.close()
f3.close()
f4.close()
f5.close()
f6.close()
f7.close()
f8.close()
f9.close()

print(np.min((1/prim_act_sum)*np.array(avg_entropies)), np.max((1/prim_act_sum)*np.array(avg_entropies)))
print(np.min((1/prim_act_sum)*np.array(avg_gini)), np.max((1/prim_act_sum)*np.array(avg_gini)))
print(np.min((1/prim_act_sum)*np.array(avg_mc_loss)), np.max((1/prim_act_sum)*np.array(avg_mc_loss)))


with open(args.res_file, "a") as fl:
    fl.write(f"wted_entropies: {avg_entropies} and avg_entropy {(1/prim_act_sum)*np.array(avg_entropies)}\n")
    fl.write(f"wted_ginis: {avg_gini} and avg_gini {(1/prim_act_sum)*np.array(avg_gini)}\n")
    fl.write(f"wted_mc_losses: {avg_mc_loss} and avg_mc_loss {(1/prim_act_sum)*np.array(avg_mc_loss)}\n")
    fl.write(f"Accuracy: {acc}")
fl.close()

