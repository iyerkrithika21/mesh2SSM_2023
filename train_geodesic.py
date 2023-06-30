from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR,ReduceLROnPlateau,StepLR
from data import Meshes,MeshesWithFaces
from mesh_vae import VAE
from model_autoencoder import Mesh2SSM_AE
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream, prepare_logger, cd_loss_L1
import sklearn.metrics as metrics
from chamfer_distance import ChamferDistance


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp train_geodesic.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp model_autoencoder.py checkpoints' + '/' + args.exp_name + '/' + 'model_ae.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')

def train(args, io):
    epochs_dir, log_fd, train_writer, val_writer = prepare_logger(args)

    training_data = MeshesWithFaces(directory = args.data_directory, extention=args.extention,k=args.k)
    args.scale = training_data.scale
    print(training_data.scale)
    train_loader = DataLoader(training_data, num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_data = MeshesWithFaces(directory = args.data_directory, extention=args.extention, partition ='val',k=args.k)
    args.test_scale = test_data.scale
    test_loader = DataLoader(test_data, num_workers=8,
                             batch_size=args.batch_size, shuffle=True, drop_last=True)

    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    
   
    model = Mesh2SSM_AE(args)
    args.num_points = model.set_template(args)
    

    model_vae = VAE(args).double().to(device)
    num_steps = int(len(training_data)/args.batch_size)
    print(str(model_vae))
        
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
        opt_vae = optim.SGD(model_vae.parameters(), lr=args.vae_lr, momentum=args.momentum, weight_decay=1e-4)    
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr,betas=(0.9, 0.999))
        opt_vae = optim.Adam(model_vae.parameters(), lr=args.vae_lr,betas=(0.9, 0.999))
        
    
    scheduler_vae = StepLR(opt_vae,  step_size=200, gamma=0.1)
    
    scheduler = StepLR(opt, step_size=200, gamma=0.1)

    criterion = ChamferDistance()

    best_test_dist = 10e5
    step = 0
    val_step = 0
    ae_burnin = 100 # number of epochs for training mesh2ssm
    template_update_interval = 200 # after how many epochs the template should be updated
    first_update_epoch = 400 # first time template is updated
    for epoch in range(args.epochs*2):
        
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        
        
        # hyperparameter alpha
        if step < int((30*args.epochs*num_steps)/100):
            alpha = 0.01
        elif step < int((50*args.epochs*num_steps)/100):
            alpha = 0.1
        elif step < int((80*args.epochs*num_steps)/100):
            alpha = 0.5
        else:
            alpha = 1.0
        

        for data, idx, label, _ in train_loader:
            data, idx, label= data.to(device), idx.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            
            particles, reconstruction = model(data,idx)
            

            if(epoch%2==1):
                if(epoch>ae_burnin):
                    model.eval()
                    model_vae.train()
                    opt_vae.zero_grad()
                    
                    mu,logvar,z,x_recon = model_vae(particles)
                    KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),dim=1),dim=0)
                    MSE =  (torch.sum(torch.sum(torch.sum((particles - x_recon) ** 2, axis=-1),axis=-1)))/batch_size
                    loss = args.vae_mse_weight*MSE + KLD
                    
                    loss.backward()
                    opt_vae.step()
                    scheduler_vae.step()
                    print(f'Epoch: {epoch}  VAE Training Loss: MSE: {MSE.detach().item() }, KLD: {KLD.detach().item() }')
            
            if(epoch%2==0):

                model.train()
                model_vae.eval()
                opt.zero_grad()
                
                dist1, dist2, _, _= criterion(label, particles)
                chamfer_loss = 0.5 * (dist1.mean() + dist2.mean())
                mse_loss = F.mse_loss(data.permute(0, 2, 1), reconstruction)
                    

                loss1 = chamfer_loss + args.mse_weight*mse_loss

                loss = loss1 + alpha*cd_loss_L1(label,particles)
                loss.backward()
                opt.step()         
                train_loss += loss.detach().item() 
                scheduler.step()
                train_writer.add_scalar('training loss', loss.detach().item(), step)
                step+=1
                
                print(f'Epoch: {epoch}  Training Loss: {train_loss}')

            
        if(epoch%template_update_interval==0 and epoch >=first_update_epoch):
            model_vae.eval()
            model.eval()
            with torch.no_grad():
                samples = model_vae.sample(200)
                samples = samples.detach().cpu().numpy()
            for sid in range(10):
                np.savetxt(args.pred_dir+"sample_"+str(sid)+".particles",samples[sid]*args.scale)
           
            template = np.mean(samples,axis=0)
            template = np.reshape(template,(args.num_points,3))
            np.savetxt(args.pred_dir+ "learned_template_"+str(epoch)+".particles", template*args.scale)
            print("Setting Template")
            model.set_template(args,array=template)
            
        ####################
        # Validation to save best model
        ####################
        if(epoch%10==0):

            test_loss = 0.0
            count = 0.0
            model.eval()
            model_vae.eval()
            for data, idx, label, names in test_loader:
                data, idx, label= data.to(device),idx.to(device),label.to(device).squeeze()
                # input(data.shape)
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                with torch.no_grad():
                    particles,reconstruction = model(data)
                    mu,logvar,z,x_recon = model_vae(particles) 
                dist1, dist2, idx1, idx2= criterion(label, particles)
                loss = 0.5 * (dist1.mean() + dist2.mean())
                val_writer.add_scalar('test chamfer loss', loss, val_step)
                val_step += 1
                test_loss += loss.detach().item() #* batch_size
                for i in range(len(particles)):
                    p = particles[i].detach().cpu().numpy()*args.test_scale
                    r = reconstruction[i].detach().cpu().numpy()*args.test_scale
                    n = names[i].split(args.extention)[0] + ".particles"
                    d = data[i].permute(1,0).detach().cpu().numpy()*args.test_scale
                    vae_r = x_recon[i].detach().cpu().numpy()*args.test_scale

                    np.savetxt(args.pred_dir + n, p)
                    np.savetxt(args.recon_dir + n, r)
                    np.savetxt(args.recon_dir + "og"+n, d)
                    np.savetxt(args.pred_dir + "vae_"+n, vae_r)
                    
                
            
            if test_loss <= best_test_dist:
                best_test_dist = test_loss
                torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)
                torch.save(model_vae.state_dict(), 'checkpoints/%s/models/model_vae.t7' % args.exp_name)
                template = model.get_template()
                np.savetxt('checkpoints/%s/models/best_template.txt' % args.exp_name, template)

        
    torch.save(model.state_dict(), 'checkpoints/%s/models/model_last.t7' % args.exp_name)
    torch.save(model_vae.state_dict(), 'checkpoints/%s/models/model_vae_last.t7' % args.exp_name)
    template = model.get_template()
    np.savetxt('checkpoints/%s/models/final_template.txt' % args.exp_name, template)



if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Mesh2SSM: From surface meshes to statistical shape models of anatomy')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=10, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=10, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--vae_lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=128, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--nf', type=int, default=8, metavar='N',
                        help='Dimension of IMnet nf')
    parser.add_argument('--k', type=int, default=10, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--data_directory', type=str,
                        help="data directory")
    parser.add_argument('--model_type', type=str, default = 'autoencoder',
                        help="model type autoencoder or only encoder")
    parser.add_argument('--mse_weight', type=float, default=0.01)
    parser.add_argument('--template', type=str, default = "template")
    parser.add_argument('--extention', type=str, default=".ply")
    parser.add_argument('--gpuid', type=int, default=0)
    parser.add_argument('--vae_mse_weight', type=float, default=10)
    parser.add_argument('--latent_dim', type = int, default = 64)
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))
    args.pred_dir = 'checkpoints/' + args.exp_name + "/output/"
    if not os.path.exists(args.pred_dir):
        os.makedirs(args.pred_dir)
    args.recon_dir = 'checkpoints/' + args.exp_name + "/recon/"
    if not os.path.exists(args.recon_dir):
        os.makedirs(args.recon_dir)
    args.log_dir = 'checkpoints/' 
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpuid)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)



'''
This repo reuses code from: https://github.com/WangYueFt/dgcnn/
'''