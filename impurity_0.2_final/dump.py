from utils import *
from arch import *

parser = argparse.ArgumentParser(description='Train a fully connected network with regularization')
parser.add_argument('--chk_file', metavar='CHKPNT_FILE', type=str, default='./checkpoints/MNIST/CapsNet/chkpoint_adam_ExpLR_run1.pth', help='checkpoint file')
#parser.add_argument('--conf_file', metavar='COFUSION_FILE', type=str, default='../../confusion.csv', help='confusion_matrix file')
#parser.add_argument('--res_file', metavar='RESULTS_FILE', type=str, default='../../results.txt', help='results.txt file')
parser.add_argument('--reg_type', metavar='REG_TYPE', type=str, default='unreg', help='type of reg')
args = parser.parse_args()

'''
with open(args.res_file, "a") as fl:
    fl.write(args.chk_file+'\n')
fl.close()

with open(args.conf_file, "a") as conf_fl:
    conf_fl.write(args.chk_file+'\n')
conf_fl.close()

conf_fl = open(args.conf_file, "a")
'''

model = CapsNet().to(DEVICE)
chk = torch.load(args.chk_file)
acc = chk['acc']
model.load_state_dict(chk['model'])
dset = torchvision.datasets.KMNIST(root='../../data', download=False, train=False, transform=transforms.ToTensor())
batch_size = 100
loader = torch.utils.data.DataLoader(dset,batch_size=batch_size, shuffle=False)
model.eval()

f0 = open('../../primary_activations/' + args.reg_type + '/0.csv', 'a')
f1 = open('../../primary_activations/' + args.reg_type + '/1.csv', 'a')
f2 = open('../../primary_activations/' + args.reg_type + '/2.csv', 'a')
f3 = open('../../primary_activations/' + args.reg_type + '/3.csv', 'a')
f4 = open('../../primary_activations/' + args.reg_type + '/4.csv', 'a')
f5 = open('../../primary_activations/' + args.reg_type + '/5.csv', 'a')
f6 = open('../../primary_activations/' + args.reg_type + '/6.csv', 'a')
f7 = open('../../primary_activations/' + args.reg_type + '/7.csv', 'a')
f8 = open('../../primary_activations/' + args.reg_type + '/8.csv', 'a')
f9 = open('../../primary_activations/' + args.reg_type + '/9.csv', 'a')

l0,l1,l2,l3,l4,l5,l6,l7,l8,l9 = [],[],[],[],[],[],[],[],[],[]

def switchFile(num):
    switcher = {
        0: [f0,l0,class_f0,class_l0], 1:[f1,l1,class_f1,class_l1], 2:[f2,l2,class_f2,class_l2], 3:[f3,l3,class_f3,class_l3], 4:[f4,l4,class_f4,class_l4],
        5: [f5,l5,class_f5,class_l5], 6:[f6,l6,class_f6,class_l6], 7:[f7,l7,class_f7,class_l7], 8:[f8,l8,class_f8,class_l8], 9:[f9,l9,class_f9,class_l9],
    }
    return switcher.get(num)


class_f0 = open('../../class_caps_activations/' + args.reg_type + '/0.csv', 'a')
class_f1 = open('../../class_caps_activations/' + args.reg_type + '/1.csv', 'a')
class_f2 = open('../../class_caps_activations/' + args.reg_type + '/2.csv', 'a')
class_f3 = open('../../class_caps_activations/' + args.reg_type + '/3.csv', 'a')
class_f4 = open('../../class_caps_activations/' + args.reg_type + '/4.csv', 'a')
class_f5 = open('../../class_caps_activations/' + args.reg_type + '/5.csv', 'a')
class_f6 = open('../../class_caps_activations/' + args.reg_type + '/6.csv', 'a')
class_f7 = open('../../class_caps_activations/' + args.reg_type + '/7.csv', 'a')
class_f8 = open('../../class_caps_activations/' + args.reg_type + '/8.csv', 'a')
class_f9 = open('../../class_caps_activations/' + args.reg_type + '/9.csv', 'a')

class_l0,class_l1,class_l2,class_l3,class_l4,class_l5,class_l6,class_l7,class_l8,class_l9 = [],[],[],[],[],[],[],[],[],[]

conf_matrix = [[0 for _ in range(10)] for _ in range(10)]

#avg_entropies, avg_gini, avg_mc_loss = [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0]

with torch.no_grad():
     for batch_idx, (data, lbls) in enumerate(loader):
         data, labels = data.to(DEVICE), one_hot(lbls).to(DEVICE)
         output, masked_output, recnstrcted, c_ij, primary_activations, u_hat_activations = model(data)
         digit_out_activations = torch.norm(output.squeeze(), dim=-1, keepdim=False)
         #print(f"SAI: {output.shape} and {digit_out_activations.shape}")
         for btch_elem in range(batch_size):
             #avg_entropies[torch.max(labels[btch_elem], dim=0)[1].item()] += act_wt_entropy(c_ij[btch_elem])
             #avg_gini[torch.max(labels[btch_elem], dim=0)[1].item()] += act_wt_gini(c_ij[btch_elem])
             #avg_mc_loss[torch.max(labels[btch_elem], dim=0)[1].item()] += act_wt_misclassification(c_ij[btch_elem])

             #print(digit_out_activations[btch_elem].shape, torch.max(digit_out_activations[btch_elem], dim=0)[1])
             conf_matrix[torch.max(labels[btch_elem], dim=0)[1].item()][torch.max(digit_out_activations[btch_elem], dim=0)[1].item()] += 1

             switchFile(torch.max(labels[btch_elem], dim=0)[1].item())[3].append(digit_out_activations[btch_elem].detach().cpu().numpy())
             #switchFile(torch.max(labels[btch_elem], dim=0)[1].item())[1].append(primary_activations[btch_elem].detach().cpu().numpy())
             
             #-------------------------------------------------------------------------------------------------------------------------------------------- 
             #np.savetxt(switchFile(torch.max(labels[btch_elem], dim=0)[1].item()), 
             #print(primary_activations[btch_elem].shape)
             #print(torch.max(labels[btch_elem], dim=0)[1].item())
             #np.savetxt(switchFile(torch.max(labels[btch_elem], dim=0)[1].item()), np.c_[primary_activations[btch_elem].detach().cpu().numpy()], delimiter=",")
             #switchFile(torch.max(labels[btch_elem], dim=0)[1].item()).write("\n\n\n\n")
             #print(labels[x])
             #print(c_ij[x,200,:])
             #print(primary_activations[x,200])
         #print(primary_activations.mean()) 
         #c_ij = c_ij.mean(dim=0).squeeze()
         #dump_cij_impurities(c_ij, './cij_files/reg/cij_'+str(lbls[0].item())+'.csv', './cij_files/reg/entropy_'+str(lbls[0].item())+'.csv', './cij_files/reg/gini_'+str(lbls[0].item())+'.csv')
         #print(f"...batch {batch_idx} done")

print(conf_matrix)
#np.savetxt(conf_fl, conf_matrix, delimiter=",")
#conf_fl.write("\n\n\n")
#conf_fl.close()
#assert False

np.savetxt(class_f0, np.transpose(class_l0), delimiter=",")
print(f"{len(class_l0)}...done")
np.savetxt(class_f1, np.transpose(class_l1), delimiter=",")
print(f"{len(class_l1)}...done")
np.savetxt(class_f2, np.transpose(class_l2), delimiter=",")
print(f"{len(class_l2)}...done")
np.savetxt(class_f3, np.transpose(class_l3), delimiter=",")
print(f"{len(class_l3)}...done")
np.savetxt(class_f4, np.transpose(class_l4), delimiter=",")
print(f"{len(class_l4)}...done")
np.savetxt(class_f5, np.transpose(class_l5), delimiter=",")
print(f"{len(class_l5)}...done")
np.savetxt(class_f6, np.transpose(class_l6), delimiter=",")
print(f"{len(class_l6)}...done")
np.savetxt(class_f7, np.transpose(class_l7), delimiter=",")
print(f"{len(class_l7)}...done")
np.savetxt(class_f8, np.transpose(class_l8), delimiter=",")
print(f"{len(class_l8)}...done")
np.savetxt(class_f9, np.transpose(class_l9), delimiter=",")
print(f"{len(class_l9)}...done")

'''
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
'''
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

assert False

with open(args.res_file, "a") as fl:
    fl.write(f"Entropies: {avg_entropies} and avg_entropy {torch.mean(torch.Tensor(avg_entropies))}\n")
    fl.write(f"ginis: {avg_gini} and avg_gini {torch.mean(torch.Tensor(avg_gini))}\n")
    fl.write(f"mc_losses: {avg_mc_loss} and avg_mc_loss {torch.mean(torch.Tensor(avg_mc_loss))}\n")
    fl.write(f"Accuracy: {acc}")
fl.close()
