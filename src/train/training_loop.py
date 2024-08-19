import torch
from src.diffusion.fp16_util import MixedPrecisionTrainer
from src.diffusion.resample import LossAwareSampler
from src.dataset.eval_dataset import Evaluation_Dataset
from src.dataset.tensors import gt_collate_fn
from src.eval.eval_himo_2o import evaluation
from src.feature_extractor.eval_wrapper import EvaluationWrapper
from src.dataset.eval_gen_dataset import get_eval_gen_loader
from src.diffusion import logger
import functools
from loguru import logger as log

from src.utils import dist_utils

from torch.optim import AdamW
from torch.utils.data import DataLoader
import blobfile as bf
from src.diffusion.resample import create_named_schedule_sampler
from tqdm import tqdm
import numpy as np
import os
import time


class TrainLoop:
    def __init__(self,args,train_platform,model,diffusion,data_loader):
        self.args=args
        self.train_platform=train_platform
        self.model=model
        self.diffusion=diffusion
        self.data=data_loader

        self.batch_size=args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.lr=args.lr
        self.log_interval=args.log_interval
        self.save_interval=args.save_interval
        self.resume_checkpoint=args.resume_checkpoint
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay=args.weight_decay
        self.lr_anneal_steps=args.lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size # * dist.get_world_size()
        # self.num_steps = args.num_steps
        # self.num_epochs = self.num_steps // len(self.data) + 1
        self.num_epochs=args.num_epochs

        self.sync_cuda = torch.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.save_path=args.save_path

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_utils.dev() != 'cpu':
            self.device = torch.device(dist_utils.dev())

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        self.eval_wrapper,self.eval_data,self.eval_gt_data=None,None,None
        if args.eval_during_training:
            mm_num_samples = 0  # mm is super slow hence we won't run it during training
            mm_num_repeats = 0  # mm is super slow hence we won't run it during training
            gen_dataset=Evaluation_Dataset(args,split='val',mode='eval')
            gen_loader=DataLoader(gen_dataset,batch_size=args.eval_batch_size,
                                  shuffle=True,num_workers=8,drop_last=True,collate_fn=gt_collate_fn)
            gt_dataset=Evaluation_Dataset(args,split='val',mode='gt')
            self.eval_gt_data=DataLoader(gt_dataset,batch_size=args.eval_batch_size,
                                  shuffle=True,num_workers=8,drop_last=True,collate_fn=gt_collate_fn)
            
            self.eval_wrapper=EvaluationWrapper(args)
            self.eval_data={
                'test':lambda :get_eval_gen_loader(
                    args,model,diffusion,gen_loader,gen_loader.dataset.max_motion_length,
                    args.eval_batch_size,mm_num_samples,mm_num_repeats,
                    1000,scale=1.

                )
            }
        self.use_ddp=False
        self.ddp_model=self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            log.info(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                dist_utils.load_state_dict(
                    resume_checkpoint, map_location=dist_utils.dev()
                )
            )
    
    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            log.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_utils.load_state_dict(
                opt_checkpoint, map_location=dist_utils.dev()
            )
            self.opt.load_state_dict(state_dict)
    
    def run_loop(self):
        for epoch in range(self.num_epochs):
            log.info(f'Starting epoch {epoch}/{self.num_epochs}')
            for inp,cond in tqdm(self.data):
                if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                    break
                
                inp=inp.to(self.device)
                cond['y'] ={k:v.to(self.device) if torch.is_tensor(v) else v for k,v in cond['y'].items()}

                self.run_step(inp,cond)
                if self.step % self.log_interval == 0:
                    for k,v in logger.get_current().name2val.items():
                        if k == 'loss':
                            print('step[{}]: loss[{:0.5f}]'.format(self.step+self.resume_step, v))

                        if k in ['step', 'samples'] or '_q' in k:
                            continue
                        else:
                            self.train_platform.report_scalar(name=k, value=v, iteration=self.step, group_name='Loss')
                if self.step % self.save_interval == 0:
                    self.save()
                    self.model.eval()
                    self.evaluate()
                    self.model.train()

                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                self.step += 1
            if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                break
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
            self.evaluate()
    
    def run_step(self,inp,cond):
        self.forward_backward(inp,cond)
        self.mp_trainer.optimize(self.opt)
        self._anneal_lr()
        self.log_step()

    def forward_backward(self,batch,cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            micro_cond = cond
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_utils.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,  # [bs, ...]
                t,  # [bs](int) sampled timesteps
                model_kwargs=micro_cond,
                dataset=self.args.dataset
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)
    
    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr
    
    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def ckpt_file_name(self):
        return f"model{(self.step+self.resume_step):09d}.pt"

    def save(self):
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)

            # Do not save CLIP weights
            clip_weights = [e for e in state_dict.keys() if e.startswith('clip_model.')]
            for e in clip_weights:
                del state_dict[e]

            log.info(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_path, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)

        with bf.BlobFile(
            bf.join(self.save_path, f"opt{(self.step+self.resume_step):09d}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)
    
    def evaluate(self):
        if not self.args.eval_during_training:
            return
        start_eval = time.time()
        if self.eval_wrapper is not None:
            log.info('Running evaluation loop: [Should take about 90 min]')
            log_file = os.path.join(self.save_path, f'eval_model_{(self.step + self.resume_step):09d}.log')
            diversity_times = 100 if self.args.obj=='2o' else 40# 200
            eval_rep_time=1 # 3
            mm_num_times = 0  # mm is super slow hence we won't run it during training
            eval_dict = evaluation(
                self.eval_wrapper, self.eval_gt_data, self.eval_data, log_file,
                replication_times=eval_rep_time, diversity_times=diversity_times, mm_num_times=mm_num_times, run_mm=False)
            log.info(eval_dict)
            for k, v in eval_dict.items():
                if k.startswith('R_precision'):
                    for i in range(len(v)):
                        self.train_platform.report_scalar(name=f'top{i + 1}_' + k, value=v[i],
                                                          iteration=self.step + self.resume_step,
                                                          group_name='Eval')
                else:
                    self.train_platform.report_scalar(name=k, value=v, iteration=self.step + self.resume_step,
                                                      group_name='Eval')
        end_eval=time.time()
        log.info(f'Evaluation time: {round(end_eval-start_eval)/60} seconds')
            
    
def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None

def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0
    
def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)