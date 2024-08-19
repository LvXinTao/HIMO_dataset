import os
import wandb

class TrainPlatform:
    def __init__(self, save_dir):
        pass

    def report_scalar(self, name, value, iteration, group_name=None):
        pass

    def report_args(self, args, name):
        pass

    def close(self):
        pass


class ClearmlPlatform(TrainPlatform):
    def __init__(self, save_dir):
        from clearml import Task
        path, name = os.path.split(save_dir)
        self.task = Task.init(project_name='motion_diffusion',
                              task_name=name,
                              output_uri=path)
        self.logger = self.task.get_logger()

    def report_scalar(self, name, value, iteration, group_name):
        self.logger.report_scalar(title=group_name, series=name, iteration=iteration, value=value)

    def report_args(self, args, name):
        self.task.connect(args, name=name)

    def close(self):
        self.task.close()


class TensorboardPlatform(TrainPlatform):
    def __init__(self, save_dir):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=save_dir)

    def report_scalar(self, name, value, iteration, group_name=None):
        self.writer.add_scalar(f'{group_name}/{name}', value, iteration)

    def close(self):
        self.writer.close()

class WandbPlatform(TrainPlatform):
    def __init__(self, save_dir):
        import wandb
        wandb.init(project='HIMO_eccv',
                   name=os.path.split(save_dir)[-1], dir=save_dir)

    def report_scalar(self, name, value, iteration, group_name=None):
        wandb.log({f'{group_name}/{name}': value}, step=iteration)

    def report_args(self, args, name):
        wandb.config.update(args, allow_val_change=True)
        
    def close(self):
        wandb.finish()

class NoPlatform(TrainPlatform):
    def __init__(self, save_dir):
        pass


