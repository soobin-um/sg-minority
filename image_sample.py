"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

import lpips

def main():
    args = create_argparser().parse_args()
    
    logger.configure(dir=args.out_dir)

    if not args.gpu_id == '':
        logger.log("using device %s" % args.gpu_id)
        th.cuda.set_device(th.device(f"cuda:{int(args.gpu_id)}"))

    if not (args.seed == '' or args.seed == 'rand'):
        logger.log("setting seed to %s" % args.seed)
        th.manual_seed(int(args.seed))
        np.random.seed(int(args.seed))

    dist_util.setup_dist()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
        
    mg_kwargs = {
        'use_ms_grad': args.use_ms_grad,
        'norm_for_mg': args.norm_for_mg,
        't_mid': args.t_mid,
        'mg_scale': args.mg_scale,
        'p_ratio': args.p_ratio,
        'num_mc_samples': args.num_mc_samples,
        'mg_scale_type': args.mg_scale_type,
        'use_normed_grad': args.use_normed_grad,
        'mg_clip_denoised': args.mg_clip_denoised,
        'use_lpips': args.use_lpips,
        'inter_rate': args.inter_rate,
    }
    print(mg_kwargs)
    
    loss_lpips = None
    if args.use_lpips:
        loss_lpips = lpips.LPIPS(
            net='alex'
        ).to(dist_util.dev())
    mg_kwargs['loss_lpips'] = loss_lpips
    
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            mg_kwargs=mg_kwargs
        )
        
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
            
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        out_dir="./outputs",
        seed='',
        gpu_id='',
        use_ms_grad=False, # arguments for mg begin!
        use_lpips=True,
        norm_for_mg=2.0,
        t_mid=-1.0,
        mg_scale=1.0,
        p_ratio=0.8,
        num_mc_samples=1,
        mg_scale_type='var',
        use_normed_grad=True,
        mg_clip_denoised=False,
        inter_rate=5,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
