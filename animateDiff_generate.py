
#patch diffusers

import diffusers.utils.torch_utils as torch_utils_module
import diffusers.utils as utils_module

# This will copy all the attributes from torch_utils to utils.
for attribute_name in dir(torch_utils_module):
    if not attribute_name.startswith('__'):
        setattr(utils_module, attribute_name, getattr(torch_utils_module, attribute_name))
        
import diffusers.utils as utils_module

# Set the attribute value
utils_module.is_safetensors_available = True

#


import glob
import logging
import os.path
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import torch
import typer
from diffusers.utils.logging import \
    set_verbosity_error as set_diffusers_verbosity_error
from rich.logging import RichHandler

from animatediff import __version__, console, get_dir
from animatediff.generate import (controlnet_preprocess, create_pipeline,
                                  create_us_pipeline, img2img_preprocess,
                                  ip_adapter_preprocess,
                                  load_controlnet_models, prompt_preprocess,
                                  region_preprocess, run_inference,
                                  run_upscale, save_output,
                                  unload_controlnet_models,
                                  wild_card_conversion)
from animatediff.pipelines import AnimationPipeline, load_text_embeddings
from animatediff.settings import (CKPT_EXTENSIONS, InferenceConfig,
                                  ModelConfig, get_infer_config,
                                  get_model_config)
from animatediff.utils.civitai2config import generate_config_from_civitai_info
from animatediff.utils.model import (checkpoint_to_pipeline,
                                     fix_checkpoint_if_needed, get_base_model)
from animatediff.utils.pipeline import get_context_params, send_to_device
from animatediff.utils.util import (extract_frames, is_v2_motion_module,
                                    path_from_cwd, save_frames, save_imgs,
                                    save_video,
                                    set_tensor_interpolation_method)
from animatediff.utils.wild_card import replace_wild_card

from diffusers import LCMScheduler

cli: typer.Typer = typer.Typer(
    context_settings=dict(help_option_names=["-h", "--help"]),
    rich_markup_mode="rich",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
data_dir = get_dir("data")
checkpoint_dir = data_dir.joinpath("models/sd")
pipeline_dir = data_dir.joinpath("models/huggingface")


try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

if IN_COLAB:
    import sys
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
else:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            RichHandler(console=console, rich_tracebacks=True),
        ],
        datefmt="%H:%M:%S",
        force=True,
    )

logger = logging.getLogger(__name__)




from animatediff.stylize import stylize




# mildly cursed globals to allow for reuse of the pipeline if we're being called as a module
g_pipeline: Optional[AnimationPipeline] = None
last_model_path: Optional[Path] = None


def version_callback(value: bool):
    if value:
        console.print(f"AnimateDiff v{__version__}")
        raise typer.Exit()

def get_random():
    import sys

    import numpy as np
    return int(np.random.randint(sys.maxsize, dtype=np.int64))

def generate(
    model_name_or_path: Annotated[
        Path,
        typer.Option(
            ...,
            "--model-path",
            "-m",
            path_type=Path,
            help="Base model to use (path or HF repo ID). You probably don't need to change this.",
        ),
    ] = Path("runwayml/stable-diffusion-v1-5"),
    config_path: Annotated[
        Path,
        typer.Option(
            "--config-path",
            "-c",
            path_type=Path,
            exists=True,
            readable=True,
            dir_okay=False,
            help="Path to a prompt configuration JSON file",
        ),
    ] = Path("config/prompts/01-ToonYou.json"),
    width: Annotated[
        int,
        typer.Option(
            "--width",
            "-W",
            min=64,
            max=3840,
            help="Width of generated frames",
            rich_help_panel="Generation",
        ),
    ] = 512,
    height: Annotated[
        int,
        typer.Option(
            "--height",
            "-H",
            min=64,
            max=2160,
            help="Height of generated frames",
            rich_help_panel="Generation",
        ),
    ] = 512,
    length: Annotated[
        int,
        typer.Option(
            "--length",
            "-L",
            min=1,
            max=9999,
            help="Number of frames to generate",
            rich_help_panel="Generation",
        ),
    ] = 16,
    context: Annotated[
        Optional[int],
        typer.Option(
            "--context",
            "-C",
            min=1,
            max=32,
            help="Number of frames to condition on (default: max of <length> or 32). max for motion module v1 is 24",
            show_default=False,
            rich_help_panel="Generation",
        ),
    ] = None,
    overlap: Annotated[
        Optional[int],
        typer.Option(
            "--overlap",
            "-O",
            min=0,
            max=12,
            help="Number of frames to overlap in context (default: context//4)",
            show_default=False,
            rich_help_panel="Generation",
        ),
    ] = None,
    stride: Annotated[
        Optional[int],
        typer.Option(
            "--stride",
            "-S",
            min=0,
            max=8,
            help="Max motion stride as a power of 2 (default: 0)",
            show_default=False,
            rich_help_panel="Generation",
        ),
    ] = None,
    repeats: Annotated[
        int,
        typer.Option(
            "--repeats",
            "-r",
            min=1,
            max=99,
            help="Number of times to repeat the prompt (default: 1)",
            show_default=False,
            rich_help_panel="Generation",
        ),
    ] = 1,
    device: Annotated[
        str,
        typer.Option(
            "--device", "-d", help="Device to run on (cpu, cuda, cuda:id)", rich_help_panel="Advanced"
        ),
    ] = "cuda",
    use_xformers: Annotated[
        bool,
        typer.Option(
            "--xformers",
            "-x",
            is_flag=True,
            help="Use XFormers instead of SDP Attention",
            rich_help_panel="Advanced",
        ),
    ] = False,
    force_half_vae: Annotated[
        bool,
        typer.Option(
            "--half-vae",
            is_flag=True,
            help="Force VAE to use fp16 (not recommended)",
            rich_help_panel="Advanced",
        ),
    ] = False,
    out_dir: Annotated[
        Path,
        typer.Option(
            "--out-dir",
            "-o",
            path_type=Path,
            file_okay=False,
            help="Directory for output folders (frames, gifs, etc)",
            rich_help_panel="Output",
        ),
    ] = Path("output/"),
    no_frames: Annotated[
        bool,
        typer.Option(
            "--no-frames",
            "-N",
            is_flag=True,
            help="Don't save frames, only the animation",
            rich_help_panel="Output",
        ),
    ] = False,
    save_merged: Annotated[
        bool,
        typer.Option(
            "--save-merged",
            "-m",
            is_flag=True,
            help="Save a merged animation of all prompts",
            rich_help_panel="Output",
        ),
    ] = False,
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            "-v",
            callback=version_callback,
            is_eager=True,
            is_flag=True,
            help="Show version",
        ),
    ] = None,
):
    """
    Do the thing. Make the animation happen. Waow.
    """

    # be quiet, diffusers. we care not for your safety checker
    set_diffusers_verbosity_error()

    config_path = config_path.absolute()
    logger.info(f"Using generation config: {path_from_cwd(config_path)}")
    model_config: ModelConfig = get_model_config(config_path)
    is_v2 = is_v2_motion_module(data_dir.joinpath(model_config.motion_module))
    infer_config: InferenceConfig = get_infer_config(is_v2)

    set_tensor_interpolation_method( model_config.tensor_interpolation_slerp )

    # set sane defaults for context, overlap, and stride if not supplied
    context, overlap, stride = get_context_params(length, context, overlap, stride)

    if (not is_v2) and (context > 24):
        logger.warning( "For motion module v1, the maximum value of context is 24. Set to 24" )
        context = 24

    # turn the device string into a torch.device
    device: torch.device = torch.device(device)

    # Get the base model if we don't have it already
    logger.info(f"Using base model: {model_name_or_path}")
    base_model_path: Path = get_base_model(model_name_or_path, local_dir=get_dir("data/models/huggingface"))

    # get a timestamp for the output directory
    time_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    # make the output directory
    save_dir = out_dir.joinpath(f"{time_str}-{model_config.save_name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Will save outputs to ./{path_from_cwd(save_dir)}")

    controlnet_image_map, controlnet_type_map, controlnet_ref_map = controlnet_preprocess(model_config.controlnet_map, width, height, length, save_dir, device)
    img2img_map = img2img_preprocess(model_config.img2img_map, width, height, length, save_dir)

    # beware the pipeline
    global g_pipeline
    global last_model_path
    if g_pipeline is None or last_model_path != model_config.path.resolve():
        g_pipeline = create_pipeline(
            base_model=base_model_path,
            model_config=model_config,
            infer_config=infer_config,
            use_xformers=use_xformers,
        )
        last_model_path = model_config.path.resolve()

        print("here",g_pipeline,dir(g_pipeline))

        g_pipeline.scheduler = LCMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )




    else:
        logger.info("Pipeline already loaded, skipping initialization")
        # reload TIs; create_pipeline does this for us, but they may have changed
        # since load time if we're being called from another package
        load_text_embeddings(g_pipeline)

    load_controlnet_models(pipe=g_pipeline, model_config=model_config)

    if g_pipeline.device == device:
        logger.info("Pipeline already on the correct device, skipping device transfer")
    else:
        g_pipeline = send_to_device(
            g_pipeline, device, freeze=True, force_half=force_half_vae, compile=model_config.compile
        )

    # save raw config to output directory
    save_config_path = save_dir.joinpath("raw_prompt.json")
    save_config_path.write_text(model_config.json(indent=4), encoding="utf-8")

    # fix seed
    for i, s in enumerate(model_config.seed):
        if s == -1:
            model_config.seed[i] = get_random()

    # wildcard conversion
    wild_card_conversion(model_config)

    is_init_img_exist = img2img_map != None
    region_condi_list, region_list, ip_adapter_config_map = region_preprocess(model_config, width, height, length, save_dir, is_init_img_exist)

    # save config to output directory
    logger.info("Saving prompt config to output directory")
    save_config_path = save_dir.joinpath("prompt.json")
    save_config_path.write_text(model_config.json(indent=4), encoding="utf-8")

    num_negatives = len(model_config.n_prompt)
    num_seeds = len(model_config.seed)
    gen_total = repeats  # total number of generations

    logger.info("Initialization complete!")
    logger.info(f"Generating {gen_total} animations")
    outputs = []

    gen_num = 0  # global generation index

    # repeat the prompts if we're doing multiple runs
    for _ in range(repeats):
        if model_config.prompt_map:
            # get the index of the prompt, negative, and seed
            idx = gen_num
            logger.info(f"Running generation {gen_num + 1} of {gen_total}")

            # allow for reusing the same negative prompt(s) and seed(s) for multiple prompts
            n_prompt = model_config.n_prompt[idx % num_negatives]
            seed = model_config.seed[idx % num_seeds]

            logger.info(f"Generation seed: {seed}")


            output = run_inference(
                pipeline=g_pipeline,
                n_prompt=n_prompt,
                seed=seed,
                steps=model_config.steps,
                guidance_scale=model_config.guidance_scale,
                unet_batch_size=model_config.unet_batch_size,
                width=width,
                height=height,
                duration=length,
                idx=gen_num,
                out_dir=save_dir,
                context_frames=context,
                context_overlap=overlap,
                context_stride=stride,
                clip_skip=model_config.clip_skip,
                controlnet_map=model_config.controlnet_map,
                controlnet_image_map=controlnet_image_map,
                controlnet_type_map=controlnet_type_map,
                controlnet_ref_map=controlnet_ref_map,
                no_frames=no_frames,
                img2img_map=img2img_map,
                ip_adapter_config_map=ip_adapter_config_map,
                region_list=region_list,
                region_condi_list=region_condi_list,
                output_map = model_config.output,
                is_single_prompt_mode=model_config.is_single_prompt_mode
            )
            outputs.append(output)
            torch.cuda.empty_cache()

            # increment the generation number
            gen_num += 1

    unload_controlnet_models(pipe=g_pipeline)


    logger.info("Generation complete!")
    if save_merged:
        logger.info("Output merged output video...")
        merged_output = torch.concat(outputs, dim=0)
        save_video(merged_output, save_dir.joinpath("final.gif"))

    logger.info("Done, exiting...")
    cli.info

    return save_dir