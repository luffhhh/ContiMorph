import sys
sys.path.append(".")

import numpy as np
import torch
import os, time
from thop import profile
from thop import clever_format
from networks.ContiMorphNet import ContiMorphNet
import SimpleITK as sitk
from data_set.utils import TagDataset, ssim , psnr,compute_metrics_all_classes
import matplotlib.pyplot as plt
import random

# device configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_dec_weights(model, weights):
    print('Resuming net weights from {} ...'.format(weights))
    w_dict = torch.load(weights)
    
    model.load_state_dict(w_dict, strict=True)
    return model

def test_Cardiac_Tagging_ME_net(net, \
                                data_root, \
                                model_path, \
                                dst_root, \
                                case = 'proposed'):
    test_dataset = TagDataset(data_path=data_root,data_type='test')
    test_batch_size = 1
    test_set_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False)
    if not os.path.exists(dst_root): os.makedirs(dst_root)
    if case == 'proposed':
        model = '/pro_new_model.pth'
    else:
        model = '/end_model.pth'
    ME_model = load_dec_weights(net, model_path + model)
    ME_model = ME_model.to(device)
    ME_model.eval()
    total_psnr, total_ssim = 0.0, 0.0
    hd_dicts = {1:[], 2:[], 3:[]}
    dice_dicts = {1:[], 2:[], 3:[]}
    sample_times = []
    for i, data in enumerate(test_set_loader):
        name, cine,ed_mask,es_mask,es_value,group= data
        

        # wrap input data in a Variable object
        cine = cine.to('cuda')
        ed_mask,es_mask = ed_mask.to('cuda'),es_mask.to('cuda')
        cine,ed_mask,es_mask  = cine.float(),ed_mask.float(),es_mask.float()
        x = cine[:, 1:, ::]  # other frames except the 1st frame        
        shape = x.shape  # batch_size, seq_length, height, width
        seq_length = shape[1]
        height = shape[2]
        width = shape[3]
        y = cine[:, 0:-1, ::]  # 1st frame also is the reference frame
        z = cine[:, 0, ::]
        z = z.repeat(seq_length,1,1,1)
        z = z.contiguous()
        t = torch.ones(seq_length,1)
        t = t.to('cuda')
        x = x.contiguous()
        x = x.view(-1, 1, height, width)  # batch_size * seq_length, channels=1, height, width
        # y = y.repeat(1, seq_length, 1, 1)  # repeat the ES frame to match other frames contained in a Cine
        y = y.contiguous()
        y = y.view(-1, 1, height, width)  # batch_size * seq_length, channels=1, height, width

        ed_mask = ed_mask.repeat(seq_length, 1, 1, 1)
        ed_mask = ed_mask.contiguous()

        # forward pass
        with torch.no_grad():
            sample_start_time = time.time()
            val_registered_cine1, val_registered_cine_lag, val_deformation_matrix, val_deformation_matrix_lag, _, _, _ = net(y, x)
            sample_end_time = time.time()
            mask_cine_lag = net.spatial_transform(ed_mask, val_deformation_matrix_lag)

        sample_duration = sample_end_time - sample_start_time
        if i != 0:
            sample_times.append(sample_duration)        
        y = cine[:, -1, ::]  # the last frame
        # print(y.shape,val_registered_cine_lag[:,0,::].shape,val_deformation_matrix_lag.shape)
        val_inf_cine = val_registered_cine1[:,0,::].cpu().detach().numpy()
        val_deformation_matrix_inf2d = val_deformation_matrix.permute(0, 2, 3, 1).cpu().detach().numpy()
        val_deformation_matrix_lag2d = val_deformation_matrix_lag.permute(0, 2, 3, 1).cpu().detach().numpy()
        val_registered_cine = val_registered_cine_lag[:,0,::].cpu().detach().numpy()
        val_deformation_matrix_lag0 = torch.cat((val_deformation_matrix_lag[:,0,::], y), dim=0)
        
        val_mask_cine_lag = mask_cine_lag[:,0,::].cpu().detach().numpy().round()

        val_deformation_matrix_lag0 = val_deformation_matrix_lag0.cuda()
        val_deformation_matrix_lag0 = val_deformation_matrix_lag0.cpu().detach().numpy()

        val_deformation_matrix_lag1 = torch.cat((val_deformation_matrix_lag[:, 1, ::], y), dim=0)
        val_deformation_matrix_lag1 = val_deformation_matrix_lag1.cuda()
        val_deformation_matrix_lag1 = val_deformation_matrix_lag1.cpu().detach().numpy()


        file_path = name[0]
        file_name = os.path.basename(file_path)
        root_vec = file_path.split(os.path.sep)
        tgt_root1 = os.path.join(dst_root, root_vec[3])
        if not os.path.exists(tgt_root1): os.mkdir(tgt_root1)
        tgt_root2 = os.path.join(tgt_root1, root_vec[4])
        if not os.path.exists(tgt_root2): os.mkdir(tgt_root2)
        tgt_root3 = os.path.join(tgt_root2, os.path.splitext(os.path.splitext(root_vec[5])[0])[0])
        if not os.path.exists(tgt_root3): os.mkdir(tgt_root3)

        cine_image = sitk.ReadImage(file_path)
        spacing1 = cine_image.GetSpacing()
        origin1 = cine_image.GetOrigin()
        direction1 = cine_image.GetDirection()

        inf_cine = sitk.GetImageFromArray(val_inf_cine)
        inf_cine.SetSpacing(spacing1)
        inf_cine.SetDirection(direction1)
        inf_cine.SetOrigin(origin1)
        sitk.WriteImage(inf_cine, os.path.join(tgt_root3, 'inf_cine.nii.gz'))

        cine_img = sitk.GetImageFromArray(val_registered_cine)
        cine_img.SetSpacing(spacing1)
        cine_img.SetDirection(direction1)
        cine_img.SetOrigin(origin1)
        sitk.WriteImage(cine_img, os.path.join(tgt_root3, file_name))

        val_deformation_matrix_lag_img0 = sitk.GetImageFromArray(val_deformation_matrix_lag0)
        val_deformation_matrix_lag_img0.SetSpacing(spacing1)
        val_deformation_matrix_lag_img0.SetOrigin(origin1)
        val_deformation_matrix_lag_img0.SetDirection(direction1)
        sitk.WriteImage(val_deformation_matrix_lag_img0, os.path.join(tgt_root3, 'deformation_matrix_x.nii.gz'))

        val_deformation_matrix_lag_img1 = sitk.GetImageFromArray(val_deformation_matrix_lag1)
        val_deformation_matrix_lag_img1.SetSpacing(spacing1)
        val_deformation_matrix_lag_img1.SetOrigin(origin1)
        val_deformation_matrix_lag_img1.SetDirection(direction1)
        sitk.WriteImage(val_deformation_matrix_lag_img1, os.path.join(tgt_root3, 'deformation_matrix_y.nii.gz'))

        val_deformation_matrix_lag2d_img = sitk.GetImageFromArray(val_deformation_matrix_lag2d)
        sitk.WriteImage(val_deformation_matrix_lag2d_img, os.path.join(tgt_root3, 'deformation_matrix_2d.nii.gz'))

        val_deformation_matrix_inf2d_img = sitk.GetImageFromArray(val_deformation_matrix_inf2d)
        sitk.WriteImage(val_deformation_matrix_inf2d_img, os.path.join(tgt_root3, 'deformation_matrix_inf2d.nii.gz'))

        val_mask_cine_lag = sitk.GetImageFromArray(val_mask_cine_lag)
        val_mask_cine_lag.SetSpacing(spacing1)
        val_mask_cine_lag.SetOrigin(origin1)
        val_mask_cine_lag.SetDirection(direction1)
        sitk.WriteImage(val_mask_cine_lag, os.path.join(tgt_root3, 'mask_cine_lag.nii.gz'))

        print('finish: ' + str(i+1))

if __name__ == '__main__':
    # data loader
    data_path_root = './database/'

    # proposed model
    vol_size = (160, 160)
    nf_enc = [16, 32, 32, 32]
    nf_dec = [32, 32, 32, 32]
    net = ContiMorphNet(vol_size, nf_enc, nf_dec)
    print(net)

    test_model_path = './models/' 
    dst_root = './database/test_result/'
    if not os.path.exists(dst_root): os.mkdir(dst_root)
    test_Cardiac_Tagging_ME_net(net=net,
                             data_root=data_path_root,
                             model_path= test_model_path,
                             dst_root=dst_root,
                             case = 'choosed')










