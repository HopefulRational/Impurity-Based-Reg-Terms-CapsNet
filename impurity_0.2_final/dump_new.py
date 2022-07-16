from utils import *
from arch import *

parser = argparse.ArgumentParser(description='Train a fully connected network with regularization')
parser.add_argument('--chk_file', metavar='CHKPNT_FILE', type=str, default='./checkpoints/MNIST/CapsNet/chkpoint_adam_ExpLR_run1.pth', help='checkpoint file')
parser.add_argument('--res_file', metavar='RESULTS_FILE', type=str, default='../../results.txt', help='results.txt file')
args = parser.parse_args()

with open(args.res_file, "a") as fl:
    fl.write(args.chk_file+'\n')
fl.close()

model = CapsNet().to(DEVICE)
chk = torch.load(args.chk_file)
model.load_state_dict(chk['model'])
dset = torchvision.datasets.KMNIST(root='../../data', download=False, train=False, transform=transforms.ToTensor())
loader = torch.utils.data.DataLoader(dset,batch_size=100)
model.eval()

avg_entropies, avg_gini, avg_mc_loss = [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0]

with torch.no_grad():
     for batch_idx, (data, lbls) in enumerate(loader):
         #print(DEVICE)
         #print(data.type())
         data = data.to(DEVICE)
         labels = one_hot(lbls).to(DEVICE)
         #data, labels = data.to(DEVICE), one_hot(lbls).to(DEVICE)
         #print("data.shape: ", data.shape)
         output, masked_output, recnstrcted, c_ij, primary_activations = model(data)
         for btch_elem in range(32):
             avg_entropies[torch.max(labels[btch_elem], dim=0)[1].item()] += act_wt_entropy(c_ij[btch_elem])
             avg_gini[torch.max(labels[btch_elem], dim=0)[1].item()] += act_wt_gini(c_ij[btch_elem])
             avg_mc_loss[torch.max(labels[btch_elem], dim=0)[1].item()] += act_wt_misclassification(c_ij[btch_elem])

             #print(labels[x])
             #print(c_ij[x,200,:])
             #print(primary_activations[x,200])
         #print(primary_activations.mean()) 
         #c_ij = c_ij.mean(dim=0).squeeze()
         #dump_cij_impurities(c_ij, './cij_files/reg/cij_'+str(lbls[0].item())+'.csv', './cij_files/reg/entropy_'+str(lbls[0].item())+'.csv', './cij_files/reg/gini_'+str(lbls[0].item())+'.csv')
         print(f"...batch {batch_idx} done")
         #break
with open(args.res_file, "a") as fl:
    fl.write(f"Entropies: {avg_entropies} and avg_entropy {torch.mean(torch.Tensor(avg_entropies))}\n")
    fl.write(f"ginis: {avg_gini} and avg_gini {torch.mean(torch.Tensor(avg_gini))}\n")
    fl.write(f"mc_losses: {avg_mc_loss} and avg_mc_loss {torch.mean(torch.Tensor(avg_mc_loss))}\n\n\n")
fl.close()
