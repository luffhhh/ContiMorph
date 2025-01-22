import sys
sys.path.append(".")

import torch
import torch.optim as optim
import os, time
from networks.ContiMorphNet import ContiMorphNet
from losses.train_loss import VM_diffeo_loss
import numpy as np
from data_set.utils import TagDataset
from torch.utils.tensorboard import SummaryWriter
import datetime
import random
import re
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/' + current_time
writer = SummaryWriter(log_dir)
# device configuration

print("device = ", device)

def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from epoch {start_epoch}")
    else:
        print("No checkpoint found, starting from scratch")
        start_epoch = 0

    return model, optimizer, start_epoch

def train_Cardiac_Tagging_ME_net(net, \
                                 data_root, \
                                 batch_size, \
                                 n_epochs, \
                                 learning_rate, \
                                 model_path, \
                                 kl_loss, \
                                 recon_loss, \
                                 smoothing_loss,
                                 steps_per_epoch=100, \
                                 checkpoint_path=None):  
    net.train()
    net.cuda()
    net = net.float()
    # optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    if checkpoint_path:
        net, optimizer, start_epoch = load_checkpoint(net, optimizer, checkpoint_path)
    else:
        start_epoch = 0  
    # training start time
    training_start_time = time.time()

    val_dataset = TagDataset(data_path=data_root,data_type='val')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    epoch_loss = 0
    train_loss = 0
    epoch_rec_loss = 0
    epoch_rec_lag_loss = 0
    epoch_smoothing_loss = 0
    epoch_smoothing_lag_loss = 0
    epoch_img_loss = 0
    epoch_ss_loss = 0
    epoch_img_lag_loss = 0
    for outer_epoch in range(start_epoch, n_epochs):
        # print training log
        print("epochs = ", outer_epoch)
        print("." * 50)

        train_dataset = TagDataset(data_path=data_root,data_type='train')
        train_loader = iter(torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True))


        epoch_loss_0 = 0
        epoch_rec_loss_0 = 0
        epoch_rec_lag_loss_0 = 0
        epoch_smoothing_loss_0 = 0
        epoch_smoothing_lag_loss_0 = 0
        epoch_img_loss_0 = 0
        epoch_ss_loss_0 = 0
        epoch_img_lag_loss_0 = 0
        for step in range(steps_per_epoch):
            name,cine,seq_es = next(train_loader)
            cine = cine.to('cuda')
            cine = cine.float()
            x = cine[:, 1:, ::]  # other frames except the 1st frame
            y = cine[:, 0:-1, ::]  # 1st frame also is the reference frame
            
            shape = x.shape  # batch_size, seq_length, height, width
            batch_size = shape[0]
            seq_length = shape[1]
            height = shape[2]
            width = shape[3]
            x = x.contiguous()
            x = x.view(-1, 1, height, width)  # batch_size * seq_length, channels=1, height, width
            y = y.contiguous()
            y = y.view(-1, 1, height, width)  # batch_size * seq_length, channels=1, height, width
            # set the param gradients as zero
            optimizer.zero_grad()

            registered_cine1, registered_cine1_lag, \
                  deformation_matrix, deformation_matrix_lag, img_loss, ss_loss, lag_img_loss  = net(y, x)

            train_smoothing_loss = smoothing_loss(deformation_matrix)
            train_smoothing_loss_lag = smoothing_loss(deformation_matrix_lag)

            a = 5
            b = 1
            c = 100
            d = 5e8
            e = 100
            training_loss = c * recon_loss(x, registered_cine1) + e * recon_loss(x, registered_cine1_lag) + \
                            a * train_smoothing_loss + b * train_smoothing_loss_lag +  \
                            c * img_loss + d * ss_loss + e * lag_img_loss


            training_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),max_norm=1.0,norm_type=2)
            optimizer.step()
                # statistic
            epoch_loss_0 += training_loss.item()
            epoch_rec_loss_0 += c * recon_loss(x, registered_cine1).item()
            epoch_rec_lag_loss_0 += e * recon_loss(x, registered_cine1_lag).item()
            epoch_smoothing_loss_0 += a * train_smoothing_loss.item()
            epoch_smoothing_lag_loss_0 += b * train_smoothing_loss_lag.item()
            epoch_img_loss_0 += c * img_loss.item()
            epoch_ss_loss_0 += d * ss_loss
            epoch_img_lag_loss_0 += e * lag_img_loss.item()


        epoch_loss_0 = epoch_loss_0 / steps_per_epoch
        epoch_rec_loss_0 = epoch_rec_loss_0 / steps_per_epoch
        epoch_rec_lag_loss_0 = epoch_rec_lag_loss_0 / steps_per_epoch
        epoch_smoothing_loss_0 = epoch_smoothing_loss_0 / steps_per_epoch
        epoch_smoothing_lag_loss_0 = epoch_smoothing_lag_loss_0 / steps_per_epoch
        epoch_img_loss_0 = epoch_img_loss_0 / steps_per_epoch
        epoch_ss_loss_0 = epoch_ss_loss_0 / steps_per_epoch
        epoch_img_lag_loss_0 = epoch_img_lag_loss_0 / steps_per_epoch


        epoch_loss += epoch_loss_0
        epoch_rec_loss += epoch_rec_loss_0
        epoch_rec_lag_loss += epoch_rec_lag_loss_0
        epoch_smoothing_loss += epoch_smoothing_loss_0
        epoch_smoothing_lag_loss += epoch_smoothing_lag_loss_0
        epoch_img_loss += epoch_img_loss_0
        epoch_ss_loss += epoch_ss_loss_0
        epoch_img_lag_loss += epoch_img_lag_loss_0
        
        train_loss = epoch_loss /(outer_epoch +1)
        train_rec_loss = epoch_rec_loss / (outer_epoch + 1)
        train_rec_lag_loss = epoch_rec_lag_loss / (outer_epoch + 1)
        train_smoothing_loss = epoch_smoothing_loss / (outer_epoch + 1)
        train_img_loss = epoch_img_loss / (outer_epoch + 1)
        train_ss_loss = epoch_ss_loss / (outer_epoch + 1)
        train_smoothing_lag_loss = epoch_smoothing_lag_loss / (outer_epoch + 1)
        train_img_lag_loss = epoch_img_lag_loss / (outer_epoch + 1)


        print("training loss: {:.6f}, train_rec_loss: {:.6f}, train_rec_lag_loss: {:.6f}, train_smoothing_loss: {:.6f}, train_smoothing_lag_loss: {:.6f}, train_img_loss: {:.6f}, train_ss_loss: {:.6f}, train_img_lag_loss: {:.6f}".format(train_loss, train_rec_loss, train_rec_lag_loss, train_smoothing_loss, train_smoothing_lag_loss, train_img_loss, train_ss_loss, train_img_lag_loss))
        writer.add_scalar('training loss', train_loss, outer_epoch)
        if (outer_epoch + 1) % 20 == 0:
            checkpoint_path = os.path.join(model_path, f"checkpoint_epoch_{outer_epoch + 1}.pth")
            torch.save({
                'epoch': outer_epoch + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {outer_epoch + 1}.")

        # when the epoch is over do a pass on the validation set
        total_val_loss = 0
        total_val_recon_cine1_loss = 0
        total_val_recon_cine1_lag_loss = 0
        total_val_smoothing_loss = 0
        total_val_smoothing_loss_lag = 0
        total_val_img_loss = 0
        total_val_ss_loss = 0
        total_val_img_lag_loss = 0
        # net.eval()
        with torch.no_grad():
            for data in val_loader:
                name, cine,seq_es = data
                val_batch_num_0 = cine.shape
                val_batch_num = val_batch_num_0[0]
                cine = cine.to('cuda')
                cine = cine.float()
                x = cine[:, 1:, ::]  # other frames except the 1st frame
                y = cine[:, 0:-1, ::]   # 1st frame also is the reference frame
                
                shape = x.shape  # batch_size, seq_length, height, width
                batch_size = shape[0]
                seq_length = shape[1]
                height = shape[2]
                width = shape[3]
                x = x.contiguous()
                x = x.view(-1, 1, height, width)  # batch_size * seq_length, channels=1, height, width
                y = y.contiguous()
                y = y.view(-1, 1, height, width)  # batch_size * seq_length, channels=1, height, width

                val_registered_cine1, val_registered_cine1_lag, \
                  val_deformation_matrix, val_deformation_matrix_lag, val_img_loss, val_ss_loss, val_lag_img_loss = net(y, x)
                val_smoothing_loss = smoothing_loss(val_deformation_matrix)
                val_smoothing_loss_lag = smoothing_loss(val_deformation_matrix_lag)
                a = 5
                b = 1
                c = 100
                d = 5e8
                e = 100
            
                val_recon_cine1_loss = c * recon_loss(x, val_registered_cine1)
                val_recon_cine1_lag_loss = e * recon_loss(x, val_registered_cine1_lag)
                val_smoothing_loss = a * val_smoothing_loss
                val_smoothing_loss_lag = b * val_smoothing_loss_lag
                val_img_loss = c * val_img_loss
                val_ss_loss = d * val_ss_loss
                val_lag_img_loss = e * val_lag_img_loss

                val_loss = val_recon_cine1_loss + val_recon_cine1_lag_loss + \
                            val_smoothing_loss+ val_smoothing_loss_lag + val_img_loss + val_ss_loss + val_lag_img_loss

                val_recon_cine1_loss = val_recon_cine1_loss / val_batch_num
                val_recon_cine1_lag_loss = val_recon_cine1_lag_loss / val_batch_num
                val_smoothing_loss = val_smoothing_loss / val_batch_num
                val_smoothing_loss_lag = val_smoothing_loss_lag / val_batch_num
                val_img_loss = val_img_loss / val_batch_num
                val_ss_loss = val_ss_loss / val_batch_num
                val_lag_img_loss = val_lag_img_loss / val_batch_num
                val_loss = val_loss / val_batch_num

                
                total_val_loss += val_loss.item()
                total_val_recon_cine1_loss += val_recon_cine1_loss.item()
                total_val_recon_cine1_lag_loss += val_recon_cine1_lag_loss.item()
                total_val_smoothing_loss += val_smoothing_loss.item()
                total_val_smoothing_loss_lag += val_smoothing_loss_lag.item()
                total_val_img_loss += val_img_loss.item()
                total_val_ss_loss += val_ss_loss.item()
                total_val_img_lag_loss += val_lag_img_loss.item()
                
        val_epoch_loss = total_val_loss / len(val_loader)
        val_epoch_recon_cine1_loss= total_val_recon_cine1_loss / len(val_loader)
        val_epoch_recon_cine1_lag_loss = total_val_recon_cine1_lag_loss / len(val_loader)
        val_epoch_smoothing_loss = total_val_smoothing_loss / len(val_loader)
        val_epoch_smoothing_lag_loss = total_val_smoothing_loss_lag / len(val_loader)
        val_epoch_img_loss = total_val_img_loss / len(val_loader)
        val_epoch_ss_loss = total_val_ss_loss / len(val_loader)
        val_epoch_img_lag_loss = total_val_img_lag_loss / len(val_loader)
        print("validation loss: {:.6f}, val_rec_loss: {:.6f}, val_rec_lag_loss: {:.6f}, val_smoothing_loss: {:.6f}, val_smoothing_lag_loss: {:.6f}, val_img_loss: {:.6f}, val_ss_loss: {:.6f}, val_img_lag_loss: {:.6f}".format(val_epoch_loss, val_epoch_recon_cine1_loss, val_epoch_recon_cine1_lag_loss, val_epoch_smoothing_loss, val_epoch_smoothing_lag_loss, val_epoch_img_loss, val_epoch_ss_loss, val_epoch_img_lag_loss))
        writer.add_scalar('validation loss', val_epoch_loss, outer_epoch)

    torch.save(net.state_dict(), os.path.join(model_path, 'end_model.pth'))
    print("Training finished! It took {:.2f}s".format(time.time() - training_start_time))


if __name__ == '__main__':
    # data loader
    os.makedirs('./models/ContiMorph/', exist_ok=True)
    training_model_path = './models/ContiMorph/'
    data_path_root = './database/'
    if not os.path.exists(training_model_path):
        os.mkdir(training_model_path)
    n_epochs = 1000
    learning_rate = 1e-4
    batch_size = 1
    print("......HYPER-PARAMETERS TRAINING......")
    print("batch size = ", batch_size)
    print("learning rate = ", learning_rate)
    print("." * 30)

    # proposed model
    vol_size = (160, 160)
    nf_enc = [16, 32, 32, 32]
    nf_dec = [32, 32, 32, 32]
    net = ContiMorphNet(vol_size, nf_enc, nf_dec)

    loss_class = VM_diffeo_loss(image_sigma=0.02, prior_lambda=10, flow_vol_shape=vol_size).cuda()
    checkpoint_path = None

    train_Cardiac_Tagging_ME_net(net=net,
                         data_root=data_path_root,
                         batch_size=batch_size,
                         n_epochs=n_epochs,
                         learning_rate=learning_rate,
                         model_path=training_model_path,
                         kl_loss=loss_class.kl_loss,
                         recon_loss=loss_class.mse_loss,
                         smoothing_loss = loss_class.gradient_loss,
                         checkpoint_path = checkpoint_path
                         )
