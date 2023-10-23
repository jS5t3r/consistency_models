"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from cm import dist_util, logger
from cm.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from cm.random_util import get_generator
from cm.karras_diffusion import karras_sample, iterative_inpainting


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    if "consistency" in args.training_mode:
        distillation = True
    else:
        distillation = False

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
        distillation=distillation,
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("inpainting...")
    if args.sampler == "multistep":
        assert len(args.ts) > 0
        ts = tuple(int(x) for x in args.ts.split(","))
    else:
        ts = None

    print("get_generator)")
    generator = get_generator(args.generator, args.num_samples, args.seed)

    # sample = np.load("/home/lorenzp/workspace/consistency_model/run/openai-2023-10-20-18-56-56-517029/samples_10x256x256x3.npz")
    # images = th.from_numpy(sample['arr_0'].transpose(0,3,1,2)).to("cuda:0")
    # images = ((images / 255.) * 2) - 1
    model_kwargs = {}

    sample = karras_sample(
        diffusion,
        model,
        (args.batch_size, 3, args.image_size, args.image_size),
        steps=args.steps,
        model_kwargs=model_kwargs,
        device=dist_util.dev(),
        clip_denoised=args.clip_denoised,
        sampler=args.sampler,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        s_churn=args.s_churn,
        s_tmin=args.s_tmin,
        s_tmax=args.s_tmax,
        s_noise=args.s_noise,
        generator=generator,
        ts=ts,
    )

    x_out, sample = iterative_inpainting(
        distiller=model,
        images=sample,
        x=generator.randn(*sample.shape, device=dist_util.dev()), #* sigma_max
        ts=ts,
        t_min=0.002,
        t_max=80.0,
        rho=7.0,
        steps=40,
        generator=generator,
    )

    x_out = ((x_out + 1) * 127.5).clamp(0, 255).to(th.uint8)
    x_out = x_out.permute(0, 2, 3, 1)
    x_out = x_out.contiguous()

    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()

    x_out = x_out.cpu().numpy()
    sample = sample.cpu().numpy()

    shape_str = "x".join([str(x) for x in x_out.shape])
    print("shape_str", shape_str)
    out_path = os.path.join(logger.get_dir(), f"inpainting_x_out_{shape_str}.npz")
    np.savez(out_path, x_out)
    out_path = os.path.join(logger.get_dir(), f"inpainting_samples_{shape_str}.npz")
    np.savez(out_path, sample)


def create_argparser():
    defaults = dict(
        training_mode="edm",
        generator="determ",
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        sampler="heun",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        steps=40,
        model_path="",
        seed=42,
        ts="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
