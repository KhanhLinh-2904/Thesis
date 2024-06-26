import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import shutil
from tqdm import tqdm
import dataloader
import model
import Myloss
import numpy as np
from torchvision import transforms
import torch

import gc
torch.cuda.empty_cache()
gc.collect()


original_stdout = sys.stdout

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def validate(val_loader, model, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for iteration, (img_lowlight, label_img) in enumerate(val_loader):
            img_lowlight = img_lowlight.cuda()
            label_img = label_img.cuda()

            enhanced_image, A  = model(img_lowlight)
            loss = criterion(enhanced_image, label_img)
            val_loss += loss.item()

    return val_loss / len(val_loader)

def train(config):

	os.environ['CUDA_VISIBLE_DEVICES']='0'
	scale_factor = config.scale_factor
	DCE_net = model.enhance_net_nopool(scale_factor).cuda()

	DCE_net.apply(weights_init)
	if config.load_pretrain == True:
		DCE_net.load_state_dict(torch.load(config.pretrain_dir))

	
	train_data = dataloader.populate_train_list(config.lowlight_images_path)
	train_dataset = dataloader.lowlight_loader(train_data)		
	
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
	
	L_color = Myloss.L_color()
	L_spa = Myloss.L_spa()
	L_exp = Myloss.L_exp(16)
	# L_exp = Myloss.L_exp(16,0.6)
	L_TV = Myloss.L_TV()


	optimizer = torch.optim.Adam(DCE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
	
	DCE_net.train()
	epoch_save = 0
	min_loss_epoch = 0
	for epoch in tqdm(range(config.num_epochs)):
		loss_value = []
		for iteration, train_img_lowlight in enumerate(train_loader):
			train_img_lowlight = train_img_lowlight.cuda()
			E = 0.6

			enhanced_image,A  = DCE_net(train_img_lowlight)
			Loss_TV = 1600*L_TV(A)
			# Loss_TV = 200*L_TV(A)			
			loss_spa = torch.mean(L_spa(enhanced_image, train_img_lowlight))
			loss_col = 5*torch.mean(L_color(enhanced_image))
			loss_exp = 10*torch.mean(L_exp(enhanced_image,E))

			loss =  Loss_TV + loss_spa + loss_col + loss_exp

			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm(DCE_net.parameters(),config.grad_clip_norm)
			optimizer.step()

			if ((iteration+1) % config.display_iter) == 0:
				print("Loss at iteration", iteration+1, ":", loss.item())
			if ((iteration+1) % config.snapshot_iter) == 0:
				torch.save(DCE_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth') 		

			loss_value.append(loss.item())

		if epoch == 0:
			min_loss_epoch = np.mean(loss_value)
			epoch_save = 1
			
		if min_loss_epoch > np.mean(loss_value):
			min_loss_epoch = np.mean(loss_value)
			epoch_save = epoch + 1

		print("Mean training loss eporch ", epoch,": ", np.mean(loss_value))
		
	print("Min training loss at epoch ", epoch_save,": ", min_loss_epoch)

	torch.cuda.empty_cache()

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Input Parameters
	# parser.add_argument('--lowlight_images_path', type=str, default="data/train_data/") #original

	parser.add_argument('--lowlight_images_path', type=str, default="Zero-DCE++/data/SICE_Part1_train")

	parser.add_argument('--lr', type=float, default=0.0001)
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=1000)
	parser.add_argument('--train_batch_size', type=int, default=8)
	parser.add_argument('--val_batch_size', type=int, default=8)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--display_iter', type=int, default=50)
	parser.add_argument('--snapshot_iter', type=int, default=10)
	parser.add_argument('--scale_factor', type=int, default=1)
	parser.add_argument('--snapshots_folder', type=str, default="Zero-DCE++/snapshots_1000/")
	parser.add_argument('--load_pretrain', type=bool, default= True)
	parser.add_argument('--pretrain_dir', type=str, default= "Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth")

	config = parser.parse_args()

	if not os.path.exists(config.snapshots_folder):
		os.mkdir(config.snapshots_folder)


	train(config)










	