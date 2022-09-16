import argparse
import os
import os.path as osp
import time

import matplotlib.pyplot as plt
import torch
import uncertainty_toolbox as uct
import yaml
from attrdict import AttrDict
# from torch.utils.tensorboard import SummaryWriter
from torchsummaryX import summary
from tqdm import tqdm

from regression.data.gp import *
from regression.utils.log import get_logger, RunningAverage, CustomSummaryWriter
from regression.utils.misc import load_module, Capturing
from regression.utils.paths import results_path, evalsets_path


def main():
    parser = argparse.ArgumentParser()

    # Experiment
    parser.add_argument('--mode', default='train', choices=['train', 'eval', 'eval_all_metrics', 'plot'])
    parser.add_argument('--expid', type=str, default='default')
    parser.add_argument('--resume', type=str, default=None)

    # Data
    parser.add_argument('--max_num_pts', type=int, default=50)
    parser.add_argument('--min_num_ctx', type=int, default=3)
    parser.add_argument('--min_num_tar', type=int, default=3)

    # Model
    parser.add_argument('--model', type=str, default="tnpd",
                        choices=["np", "anp", "cnp", "canp", "bnp", "banp", "tnpd", "tnpa", "tnpnd", "ipnp", "iptnpd"])

    # Train
    parser.add_argument('--train_seed', type=int, default=0)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--train_num_samples', type=int, default=4)
    parser.add_argument('--train_num_bs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--print_freq', type=int, default=200)
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--save_freq', type=int, default=1000)

    # Eval
    parser.add_argument('--eval_seed', type=int, default=0)
    parser.add_argument('--eval_num_batches', type=int, default=3000)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--eval_num_samples', type=int, default=50)
    parser.add_argument('--eval_logfile', type=str, default=None)

    # Plot
    parser.add_argument('--plot_seed', type=int, default=0)
    parser.add_argument('--plot_batch_size', type=int, default=16)
    parser.add_argument('--plot_num_samples', type=int, default=30)
    parser.add_argument('--plot_num_ctx', type=int, default=30)
    parser.add_argument('--plot_num_tar', type=int, default=10)
    parser.add_argument('--start_time', type=str, default=None)

    # OOD settings
    parser.add_argument('--eval_kernel', type=str, default='rbf', choices=['all', 'matern', 'periodic', 'rbf'])
    parser.add_argument('--t_noise', type=float, default=None)

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args.exp_title = f'gp_maxpts-{args.max_num_pts}_minctx-{args.min_num_ctx}_mintar-{args.min_num_tar}'
    if args.expid is not None:
        args.root = osp.join(results_path, args.exp_title, args.model, args.expid)
    else:
        args.root = osp.join(results_path, args.exp_title, args.model)
    os.makedirs(args.root, exist_ok=True)
    tb = CustomSummaryWriter(log_dir=args.root)

    model_cls = getattr(load_module(f'models/{args.model}.py'), args.model.upper())
    with open(f'configs/gp/{args.model}.yaml', 'r') as f:
        config = yaml.safe_load(f)
        args.__dict__.update(config)

    model = model_cls(**config).to(device)

    if args.mode == 'train':
        train(args, model, device=device, tb=tb)
    elif args.mode == 'eval':
        eval(args, model, device=device)
    elif args.mode == 'eval_all_metrics':
        eval_all_metrics(args, model, device=device)
    elif args.mode == 'plot':
        plot(args, model, device=device)


def train(args, model, device, tb=None):
    if osp.exists(args.root + '/ckpt.tar'):
        if args.resume is None:
            raise FileExistsError(args.root)

    with open(osp.join(args.root, 'args.yaml'), 'w') as f:
        yaml.dump(args.__dict__, f)

    eval_kernels = ['matern', 'rbf', 'periodic'] if args.eval_kernel == 'all' else [args.eval_kernel]
    for ek in eval_kernels:
        args.eval_kernel = ek
        path, filename = get_eval_path(args)
        if not osp.isfile(osp.join(path, filename)):
            print('generating evaluation sets...')
            gen_evalset(args, device=device)
    if len(eval_kernels) > 1:
        args.eval_kernel = 'all'

    torch.manual_seed(args.train_seed)
    torch.cuda.manual_seed(args.train_seed)

    sampler = GPSampler(RBFKernel())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_steps)

    total_train_time = 0
    if args.resume and osp.exists(osp.join(args.root, 'ckpt.tar')):
        ckpt = torch.load(os.path.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)
        optimizer.load_state_dict(ckpt.optimizer)
        scheduler.load_state_dict(ckpt.scheduler)
        logfilename = ckpt.logfilename
        start_step = ckpt.step
        total_train_time = ckpt.total_train_time

    else:
        logfilename = os.path.join(args.root, f'train_{time.strftime("%Y%m%d-%H%M")}.log')
        start_step = 1

    logger = get_logger(logfilename)
    ravg = RunningAverage()

    # if not args.resume:
    logger.info(f"Experiment: {args.model}-{args.expid}" + (" (resume)" if args.resume and start_step > 1 else ""))
    params = sum(p.numel() for p in model.parameters())
    logger.info(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}\n')

    if not args.resume and tb:
        tb.add_scalar('info/params', params, start_step-1)

    for step in range(start_step, args.num_steps+1):
        model.train()
        optimizer.zero_grad()
        batch = sampler.sample(
            batch_size=args.train_batch_size,
            max_num_pts=args.max_num_pts,
            min_num_ctx=args.min_num_ctx,
            device=device)

        # On first step write param summary
        if step == start_step == 1:
            with torch.no_grad():
                with Capturing() as output:
                    if args.model in ["np", "anp", "cnp", "canp", "bnp", "banp"]:
                        summary(model, batch, num_samples=args.train_num_samples)
                    else:
                        summary(model, batch)
            with open(os.path.join(args.root, 'param_count.txt'), 'w') as pf:
                for line in output:
                    pf.write(line + '\n')
        step_start_time = time.time()
        if args.model in ["np", "anp", "cnp", "canp", "bnp", "banp"]:
            outs = model(batch, num_samples=args.train_num_samples)
        else:
            outs = model(batch)

        outs.loss.backward()
        optimizer.step()
        scheduler.step()
        total_train_time += time.time() - step_start_time
        if tb:
            for k, v in outs.items():
                tb.add_scalar(f'train/{k}', v.item(), step)
            tb.add_scalar('train/time', total_train_time, step)

        for key, val in outs.items():
            ravg.update(key, val)

        if step % args.print_freq == 0:
            line = f'{args.model}:{args.expid} step {step} '
            line += f'lr {optimizer.param_groups[0]["lr"]:.3e} '
            line += f"[train_loss] "
            line += ravg.info()
            logger.info(line)

            if step % args.eval_freq == 0:
                lines = eval(args, model, device=device, tb=tb, train_step=step)
                for line in lines:
                    logger.info(line)
                logger.info('\n')
            ravg.reset()

        if step % args.save_freq == 0 or step == args.num_steps:
            ckpt = AttrDict()
            ckpt.model = model.state_dict()
            ckpt.optimizer = optimizer.state_dict()
            ckpt.scheduler = scheduler.state_dict()
            ckpt.logfilename = logfilename
            ckpt.step = step + 1
            ckpt.total_train_time = total_train_time
            torch.save(ckpt, os.path.join(args.root, 'ckpt.tar'))
    with open(osp.join(args.root, 'timing.log'), 'w') as f:
        f.write(f'Total train time: {total_train_time}\n')
        f.write(f'Avg epoch time: {total_train_time / args.num_steps}')
    args.mode = 'eval'
    eval(args, model, device=device)


def get_eval_path(args):
    path = osp.join(evalsets_path, args.exp_title)  # 'gp_maxpts-50_minctx-3_mintar-3')
    filename = f'{args.eval_kernel}-seed{args.eval_seed}'
    if args.t_noise is not None:
        filename += f'_{args.t_noise}'
    filename += '.tar'
    return path, filename

def gen_evalset(args, device):
    if args.eval_kernel == 'rbf':
        kernel = RBFKernel()
    elif args.eval_kernel == 'matern':
        kernel = Matern52Kernel()
    elif args.eval_kernel == 'periodic':
        kernel = PeriodicKernel()
    else:
        raise ValueError(f'Invalid kernel {args.eval_kernel}')
    print(f"Generating Evaluation Sets with {args.eval_kernel} kernel")

    sampler = GPSampler(kernel, t_noise=args.t_noise, seed=args.eval_seed)
    batches = []
    for _ in tqdm(range(args.eval_num_batches), ascii=True):
        batches.append(sampler.sample(
            batch_size=args.eval_batch_size,
            max_num_pts=args.max_num_pts,
            device=device))

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    path, filename = get_eval_path(args)
    if not osp.isdir(path):
        os.makedirs(path)
    torch.save(batches, osp.join(path, filename))

def eval(args, model, device, train_step=None, tb=None):
    # eval a trained model on log-likelihood
    if args.mode == 'eval':
        ckpt = torch.load(os.path.join(args.root, 'ckpt.tar'), map_location=device)
        model.load_state_dict(ckpt.model)
        if not args.eval_logfile:
            eval_logfile = f'eval_{args.eval_kernel}'
            if args.t_noise is not None:
                eval_logfile += f'_tn_{args.t_noise}'
            eval_logfile += '.log'
        else:
            eval_logfile = args.eval_logfile
        filename = os.path.join(args.root, eval_logfile)
        logger = get_logger(filename, mode='a')
    else:
        logger = None

    if args.mode == "eval":
        torch.manual_seed(args.eval_seed)
        torch.cuda.manual_seed(args.eval_seed)

    eval_kernels = ['matern', 'rbf', 'periodic'] if args.eval_kernel == 'all' else [args.eval_kernel]
    lines = []
    for ek in eval_kernels:
        args.eval_kernel = ek
        path, filename = get_eval_path(args)
        if not osp.isfile(osp.join(path, filename)):
            print('generating evaluation sets...')
            gen_evalset(args, device)
        eval_batches = torch.load(osp.join(path, filename), map_location=device)

        ravg = RunningAverage()
        model.eval()
        with torch.no_grad():
            for i, batch in tqdm(enumerate(eval_batches), ascii=True):
                for key, val in batch.items():
                    batch[key] = val.to(device)
                if args.model in ["np", "anp", "bnp", "banp"]:
                    outs = model(batch, args.eval_num_samples)
                else:
                    outs = model(batch)

                for key, val in outs.items():
                    ravg.update(key, val)
        metric_dict = {f'val/{ek}/{k}': None for k in ravg.keys()}
        torch.manual_seed(time.time())
        torch.cuda.manual_seed(time.time())

        line = f'{args.model}:{args.expid} {args.eval_kernel} '
        if args.t_noise is not None:
            line += f'tn {args.t_noise} '
        line += ravg.info()
        if tb:
            for k in ravg.keys():
                tb.add_scalar(f'val/{ek}/{k}', ravg.sum[k] / ravg.cnt[k], train_step)
                tb.add_hparams(metric_dict=metric_dict,
                               hparam_dict=args.__dict__,
                               run_name=os.path.dirname(os.path.realpath(__file__)) + os.sep + args.root)

        if logger is not None:
            logger.info(line)

        lines.append(line)
    if len(eval_kernels) > 1:
        args.eval_kernel = 'all'
    return lines


def eval_all_metrics(args, model, device):
    # eval a trained model on log-likelihood, rsme, calibration, and sharpness
    ckpt = torch.load(os.path.join(args.root, 'ckpt.tar'), map_location=device)
    model.load_state_dict(ckpt.model)
    if args.eval_logfile is None:
        eval_logfile = f'eval_{args.eval_kernel}'
        if args.t_noise is not None:
            eval_logfile += f'_tn_{args.t_noise}'
        eval_logfile += f'_all_metrics'
        eval_logfile += '.log'
    else:
        eval_logfile = args.eval_logfile
    filename = os.path.join(args.root, eval_logfile)
    logger = get_logger(filename, mode='w')

    path, filename = get_eval_path(args)
    if not osp.isfile(osp.join(path, filename)):
        print('generating evaluation sets...')
        gen_evalset(args)
    eval_batches = torch.load(osp.join(path, filename), map_location=device)

    if args.mode == "eval_all_metrics":
        torch.manual_seed(args.eval_seed)
        torch.cuda.manual_seed(args.eval_seed)

    model.eval()
    with torch.no_grad():
        ravgs = [RunningAverage() for _ in range(4)] # 4 types of metrics
        for batch in tqdm(eval_batches, ascii=True):
            for key, val in batch.items():
                batch[key] = val.to(device)
            if args.model in ["np", "anp", "cnp", "canp", "bnp", "banp"]:
                outs = model.predict(batch.xc, batch.yc, batch.xt, num_samples=args.eval_num_samples)
                ll = model(batch, num_samples=args.eval_num_samples)
            elif args.model in ["tnpa", "tnpnd"]:
                outs = model.predict(
                    batch.xc, batch.yc, batch.xt,
                    num_samples=args.eval_num_samples
                )
                ll = model(batch)
            else:
                outs = model.predict(batch.xc, batch.yc, batch.xt)
                ll = model(batch)

            mean, std = outs.loc, outs.scale

            # shape: (num_samples, 1, num_points, 1)
            if mean.dim() == 4:
                # variance of samples (Law of Total Variance) - var(X) = E[var(X|Y)] + var(E[X|Y])
                # E[var(X|Y)] : average variability within each samples
                # var(E[X|Y]) : variability between samples
                var = std.pow(2).mean(dim=0) + mean.pow(2).mean(dim=0) - mean.mean(dim=0).pow(2)
                std = var.sqrt().squeeze(0)
                # mean of samples (Law of Total Expectations) - E[E[X|Y]] = E[X]
                mean = mean.mean(dim=0).squeeze(0)
            
            mean, std = mean.squeeze().cpu().numpy().flatten(), std.squeeze().cpu().numpy().flatten()
            yt = batch.yt.squeeze().cpu().numpy().flatten()

            acc = uct.metrics.get_all_accuracy_metrics(mean, yt, verbose=False)
            calibration = uct.metrics.get_all_average_calibration(mean, std, yt, num_bins=100, verbose=False)
            sharpness = uct.metrics.get_all_sharpness_metrics(std, verbose=False)
            scoring_rule = {'tar_ll': ll.tar_ll.item()}

            batch_metrics = [acc, calibration, sharpness, scoring_rule]
            for i in range(len(batch_metrics)):
                ravg, batch_metric = ravgs[i], batch_metrics[i]
                for k in batch_metric.keys():
                    ravg.update(k, batch_metric[k])

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    line = f'{args.model}:{args.expid} {args.eval_kernel} '
    if args.t_noise is not None:
        line += f'tn {args.t_noise} '
    
    line += '\n'

    for ravg in ravgs:
        line += ravg.info()
        line += '\n'

    if logger is not None:
        logger.info(line)

    return line


def plot(args, model, device):
    seed = args.plot_seed
    num_smp = args.plot_num_samples

    if args.mode == "plot":
        ckpt = torch.load(os.path.join(args.root, 'ckpt.tar'), map_location=device)
        model.load_state_dict(ckpt.model)
    model = model.to(device)

    def tnp(x):
        return x.squeeze().cpu().data.numpy()

    kernel = RBFKernel()
    sampler = GPSampler(kernel, t_noise=args.t_noise, seed=args.eval_seed)

    xp = torch.linspace(-2, 2, 200).to(device)
    batch = sampler.sample(
        batch_size=args.plot_batch_size,
        num_ctx=args.plot_num_ctx,
        num_tar=args.plot_num_tar,
        device=device,
    )

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    Nc = batch.xc.size(1)
    Nt = batch.xt.size(1)

    model.eval()
    with torch.no_grad():
        if args.model in ["np", "anp", "bnp", "banp"]:
            outs = model(batch, num_smp, reduce_ll=False)
        else:
            outs = model(batch, reduce_ll=False)
        tar_loss = outs.tar_ll  # [Ns,B,Nt] ([B,Nt] for CNP)
        if args.model in ["cnp", "canp", "tnpd", "tnpa", "tnpnd"]:
            tar_loss = tar_loss.unsqueeze(0)  # [1,B,Nt]

        xt = xp[None, :, None].repeat(args.plot_batch_size, 1, 1)
        if args.model in ["np", "anp", "bnp", "banp", "tnpa", "tnpnd"]:
            pred = model.predict(batch.xc, batch.yc, xt, num_samples=num_smp)
        else:
            pred = model.predict(batch.xc, batch.yc, xt)
        
        mu, sigma = pred.mean, pred.scale

    if args.plot_batch_size > 1:
        nrows = max(args.plot_batch_size//4, 1)
        ncols = min(4, args.plot_batch_size)
        _, axes = plt.subplots(nrows, ncols,
                figsize=(5*ncols, 5*nrows))
        axes = axes.flatten()
    else:
        axes = [plt.gca()]

    # multi sample
    if mu.dim() == 4:
        for i, ax in enumerate(axes):
            for s in range(mu.shape[0]):
                ax.plot(tnp(xp), tnp(mu[s][i]), color='steelblue',
                        alpha=max(0.5/args.plot_num_samples, 0.1))
                ax.fill_between(tnp(xp), tnp(mu[s][i])-tnp(sigma[s][i]),
                        tnp(mu[s][i])+tnp(sigma[s][i]),
                        color='skyblue',
                        alpha=max(0.2/args.plot_num_samples, 0.02),
                        linewidth=0.0)
            ax.scatter(tnp(batch.xc[i]), tnp(batch.yc[i]),
                       color='k', label=f'context {Nc}', zorder=mu.shape[0] + 1)
            ax.scatter(tnp(batch.xt[i]), tnp(batch.yt[i]),
                       color='orchid', label=f'target {Nt}',
                       zorder=mu.shape[0] + 1)
            ax.legend()
            ax.set_title(f"tar_loss: {tar_loss[:, i, :].mean(): 0.4f}")
    else:
        for i, ax in enumerate(axes):
            ax.plot(tnp(xp), tnp(mu[i]), color='steelblue', alpha=0.5)
            ax.fill_between(tnp(xp), tnp(mu[i]-sigma[i]), tnp(mu[i]+sigma[i]),
                    color='skyblue', alpha=0.2, linewidth=0.0)
            ax.scatter(tnp(batch.xc[i]), tnp(batch.yc[i]),
                       color='k', label=f'context {Nc}')
            ax.scatter(tnp(batch.xt[i]), tnp(batch.yt[i]),
                       color='orchid', label=f'target {Nt}')
            ax.legend()
            ax.set_title(f"tar_loss: {tar_loss[:, i, :].mean(): 0.4f}")

    plt.suptitle(f"{args.expid}", y=0.995)
    plt.tight_layout()

    save_dir_1 = osp.join(args.root, f"plot_num{num_smp}-c{Nc}-t{Nt}-seed{seed}-{args.start_time}.jpg")
    file_name = "-".join([args.model, args.expid, f"plot_num{num_smp}",
                          f"c{Nc}", f"t{Nt}", f"seed{seed}", f"{args.start_time}.jpg"])
    if args.expid is not None:
        save_dir_2 = osp.join(results_path, "gp", "plot", args.expid, file_name)
        if not osp.exists(osp.join(results_path, "gp", "plot", args.expid)):
            os.makedirs(osp.join(results_path, "gp", "plot", args.expid))
    else:
        save_dir_2 = osp.join(results_path, "gp", "plot", file_name)
        if not osp.exists(osp.join(results_path, "gp", "plot")):
            os.makedirs(osp.join(results_path, "gp", "plot"))
    plt.savefig(save_dir_1)
    plt.savefig(save_dir_2)
    print(f"Evaluation Plot saved at {save_dir_1}\n")
    print(f"Evaluation Plot saved at {save_dir_2}\n")


if __name__ == '__main__':
    main()
