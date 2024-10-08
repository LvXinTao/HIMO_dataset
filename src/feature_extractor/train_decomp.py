from src.utils.parser_utils import train_decomp_args
from src.utils.word_vectorizer import WordVectorizer
from src.feature_extractor.modules import MovementConvDecoder,MovementConvEncoder
from src.train.train_platforms import WandbPlatform
from src.feature_extractor.trainer import DecompTrainerV3
from src.dataset.decomp_dataset import decomp_dataset
from torch.utils.data import DataLoader
from src.utils.misc import fixseed
import torch
from os.path import join as pjoin
import os
from loguru import logger

if __name__=='__main__':
    opt=train_decomp_args()
    fixseed(opt.seed)
    opt.device = torch.device("cpu" if opt.gpu_id==-1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)
    if opt.gpu_id != -1:
        # self.opt.gpu_id = int(self.opt.gpu_id)
        torch.cuda.set_device(opt.gpu_id)

    opt.save_path = pjoin(opt.checkpoints_dir, opt.name)
    opt.model_dir = pjoin(opt.save_path, 'model')
    opt.eval_dir = pjoin(opt.save_path, 'eval')
    opt.log_dir = pjoin(opt.save_path,opt.name,'log')
    opt.meta_dir = pjoin(opt.save_path, 'meta')

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    logger.add(pjoin(opt.log_dir,'train_decomp.log'),rotation='10 MB')

    opt.data_root = '/data/xuliang/HO2_subsets_original/HO2_final/'
    opt.max_motion_length = 300
    if opt.mode=='2o':
        dim_pose = (52*3+52*6+3)+(2*(6+3))
    elif opt.mode=='3o':
        dim_pose = (52*3+52*6+3)+(3*(6+3))
    opt.is_train=True
    
    w_vectorizer = WordVectorizer(pjoin(opt.data_root,'glove'), 'himo_vab')
    movement_enc=MovementConvEncoder(dim_pose,opt.dim_movement_enc_hidden,opt.dim_movement_latent)
    movement_dec=MovementConvDecoder(opt.dim_movement_latent,opt.dim_movement_enc_hidden,dim_pose)

    all_params = 0
    pc_mov_enc = sum(param.numel() for param in movement_enc.parameters())

    logger.info("Total parameters of prior net: {}".format(pc_mov_enc))
    all_params += pc_mov_enc

    pc_mov_dec = sum(param.numel() for param in movement_dec.parameters())

    logger.info("Total parameters of posterior net: {}".format(pc_mov_dec))
    all_params += pc_mov_dec

    train_platform=WandbPlatform(opt.save_path)
    trainer=DecompTrainerV3(opt,movement_enc,movement_dec,train_platform)

    train_dataset=decomp_dataset(opt,'train')
    val_dataset=decomp_dataset(opt,'val')

    train_loader=DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                            shuffle=True, pin_memory=True)
    
    trainer.train(train_loader, val_loader)