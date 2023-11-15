import sys
sys.path.append("D:\\img\\IP-Adapter\\")
sys.path.append("D:/img")

from rife_inference_video import interpolate_video


from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler, StableDiffusionImg2ImgPipeline, DiffusionPipeline, AutoencoderKL, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline, AutoencoderTiny, DDIMInverseScheduler, DDIMScheduler
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, AutoencoderTiny
import hashlib
import os
import uuid
from PIL import Image
#from load_llama_model import getllama, Chatbot
from ip_adapter import IPAdapterXL, IPAdapterPlus
import gc
import numpy as np
import tomesd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import random

import subprocess

import animateDiff_generate

from pathlib import Path

import json


from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
#from diffusers.utils import export_to_video

import cv2
import tempfile
from typing import List

from llama_cpp.llama import Llama, LlamaGrammar


def export_to_video(video_frames: List[np.ndarray], output_video_path: str = None,fps=8) -> str:
    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, c = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)
    return output_video_path






# from pickScore import calc_probs


# from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


# import torchaudio
from audiocraft.models import MusicGen
import soundfile as sf
# import librosa
from pydub import AudioSegment


pipe, text_generator, tokenizer, cfg = None, None, None, None

image_prompt_file = None
attack_names_template = None
descriptions_template = None
llm = None
img2img = None
ref_pipe = None
text2music = None

ip_model = None

ip_xl = False

do_save_memory = False

chatbot = None

video_pipe = None

llm = None
deciDiffusion = None

lcm_img2img = None

rifeFolder="D:\\img\\ECCV2022-RIFE"


def setup(
    _image_prompt_file="image_prompts.txt",
    _attack_names_template="attack_names.txt",
    _descriptions_template="attack_descriptions.txt",
    model_id="xyn-ai/anything-v4.0",
    textModel="EleutherAI/gpt-neo-2.7B",
    _use_llama=True,
    upscale_model=None,
    vaeModel=None,
    llamaModel="nous-llama2-7b",
    lora=None,
    # ip_adapter_base_model="D:\\img\\auto1113\\stable-diffusion-webui\\models\\Stable-diffusion\\dreamshaperXL10_alpha2Xl10.safetensors",
    # ip_image_encoder_path = "D:\\img\\IP-Adapter\\IP-Adapter\\sdxl_models\\image_encoder",
    # ip_ckpt = "D:\\img\\IP-Adapter\\IP-Adapter\\sdxl_models\\ip-adapter_sdxl.bin",
    #ip_adapter_base_model="D:\\img\\auto1113\\ciffusion-webui\\models\\Stable-diffusion\\reliberate_v20.safetensors",
    ip_adapter_base_model = "SG161222/Realistic_Vision_V4.0_noVAE",
    ip_image_encoder_path="D:\\img\\IP-Adapter\\IP-Adapter\\models\\image_encoder",
    ip_ckpt="D:\\img\\IP-Adapter\\IP-Adapter\\models\\ip-adapter-plus_sd15.bin",
    ip_vae_model_path = "stabilityai/sd-vae-ft-mse",
    #ip_adapter_base_model="waifu-diffusion/wd-1-5-beta2",
    #ip_ckpt="D:\\img\\IP-Adapter\\IP-Adapter\\models\\wd15_ip_adapter_plus.bin",
    #ip_vae_model_path = "redstonehero/kl-f8-anime2"
    llm_model="D:\lmstudio\TheBloke\Mistral-7B-OpenOrca-GGUF\mistral-7b-openorca.Q5_K_M.gguf",
    save_memory=False,
    need_txt2img=True,
    need_img2img=True,
    need_ipAdapter=True,
    need_music=True,
    need_video=False,
    need_llm=False,
    need_deciDiffusion=False,
    need_lcm_img2img=False,
    
):

    global pipe, text_generator, tokenizer, cfg, image_prompt_file, attack_names_template, descriptions_template, llm, img2img, ref_pipe, text2music
    image_prompt_file = _image_prompt_file
    attack_names_template = _attack_names_template
    descriptions_template = _descriptions_template

    global ip_model, ip_xl

    global use_llama

    global do_save_memory

    global video_pipe

    global llm

    global deciDiffusion

    if need_llm:
        llm = Llama(llm_model,
                n_ctx=4096)


    do_save_memory = save_memory

    use_llama = _use_llama

    ip_xl= "XL" in ip_adapter_base_model

    if need_txt2img:

        print("LOADING IMAGE MODEL")

        if 'xl' in model_id.lower():
            pipe = StableDiffusionXLPipeline.from_single_file(
                model_id, torch_dtype=torch.float16, use_safetensors=True
            )
            if lora is not None:
                pipe.load_lora_weights(lora)

            vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
            pipe.enable_vae_tiling()
            pipe.vae = AutoencoderTiny.from_pretrained(
                "madebyollin/taesdxl", torch_dtype=torch.float16)

            # img2img = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            #    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
            # )#todo fixme

            # check for upscale model
            if upscale_model is None:
                upscale_model = model_id

            img2img = StableDiffusionXLImg2ImgPipeline.from_single_file(
                upscale_model, torch_dtype=torch.float16, use_safetensors=True)
            # img2img.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16)
            img2img.enable_vae_tiling()


            #move to cuda if not saving memory
            if do_save_memory==False:
                pipe = pipe.to("cuda")
                img2img = img2img.to("cuda")
            

        else:
            
            # check if vae is None
            if vaeModel is not None:
                vae = AutoencoderKL.from_pretrained(
                    vaeModel, torch_dtype=torch.float16)
            else:
                vae = None

            # check if model_id is a .ckpt or .safetensors file
            if model_id.endswith(".ckpt") or model_id.endswith(".safetensors"):
                pipe = StableDiffusionPipeline.from_single_file(model_id,
                                                                torch_dtype=torch.float16)
            else:
                pipe = StableDiffusionPipeline.from_pretrained(
                    model_id, torch_dtype=torch.float16)

            pipe.scheduler = UniPCMultistepScheduler.from_config(
                pipe.scheduler.config)
            pipe.enable_attention_slicing()
            pipe.enable_xformers_memory_efficient_attention()
            pipe.safety_checker = None
            tomesd.apply_patch(pipe, ratio=0.5)

            #use taesd
            pipe.vae=AutoencoderTiny.from_pretrained(
                "madebyollin/taesd", torch_dtype=torch.float16)

            if vae is not None:
                pipe.vae = vae

            # pipe = pipe.to("cuda")

            # move pipe to CPU
            if do_save_memory:
                pipe = pipe.to("cpu")
                gc.collect()
                torch.cuda.empty_cache()
            else:
                pipe = pipe.to("cuda")


        if need_img2img:
            print("LOADING IMG2iMG MODEL")


            dummy_path = "runwayml/stable-diffusion-v1-5"

            # load upscale model
            if upscale_model is not None:
                # check if model_id is a .ckpt or .safetensors file
                if upscale_model.endswith(".ckpt") or model_id.endswith(".safetensors"):
                    uppipe = StableDiffusionPipeline.from_single_file(upscale_model,
                                                                    torch_dtype=torch.float16)
                else:
                    uppipe = StableDiffusionPipeline.from_pretrained(
                        upscale_model, torch_dtype=torch.float16)

            else:
                uppipe = pipe

            uppipe.scheduler = UniPCMultistepScheduler.from_config(
                uppipe.scheduler.config)
            uppipe.enable_attention_slicing()
            uppipe.enable_xformers_memory_efficient_attention()
            uppipe.safety_checker = None
            tomesd.apply_patch(uppipe, ratio=0.5)

            if vae is not None:
                uppipe.vae = vae

            # image to image model
            if model_id.endswith(".ckpt") or model_id.endswith(".safetensors"):

                img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
                    dummy_path,  # dummy model
                    # revision=revision,
                    scheduler=uppipe.scheduler,
                    unet=uppipe.unet,
                    vae=uppipe.vae,
                    safety_checker=uppipe.safety_checker,
                    text_encoder=uppipe.text_encoder,
                    tokenizer=uppipe.tokenizer,
                    torch_dtype=torch.float16,
                    use_auth_token=True,
                    cache_dir="./AI/StableDiffusion"
                )

            else:
                img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
                    model_id,
                    # revision=revision,
                    scheduler=uppipe.scheduler,
                    unet=uppipe.unet,
                    vae=uppipe.vae,
                    safety_checker=uppipe.safety_checker,
                    text_encoder=uppipe.text_encoder,
                    tokenizer=uppipe.tokenizer,
                    torch_dtype=torch.float16,
                    use_auth_token=True,
                    cache_dir="./AI/StableDiffusion"
                )

            del uppipe

            img2img.enable_attention_slicing()
            img2img.enable_xformers_memory_efficient_attention()
            tomesd.apply_patch(img2img, ratio=0.5)

            # move img2img to CPU
            if save_memory:
                img2img = img2img.to("cpu")
                gc.collect()
                torch.cuda.empty_cache()
            else:
                img2img = img2img.to("cuda")


    if need_ipAdapter:

        # load ip adapter
        print("LOADING IP ADAPTER")
        # load SDXL pipeline
        if "XL" in ip_adapter_base_model:
            ippipe = StableDiffusionXLPipeline.from_single_file(
                ip_adapter_base_model,
                torch_dtype=torch.float16,
                add_watermarker=False,
            )
            ippipe.vae = AutoencoderTiny.from_pretrained(
                "madebyollin/taesdxl", torch_dtype=torch.float16).to('cuda')
            
            ippipe = ippipe.to('cuda')
            ip_model = IPAdapterXL(ippipe, ip_image_encoder_path, ip_ckpt, 'cuda')
        
        
        else:
            noise_scheduler = DDIMScheduler(
                num_train_timesteps=1000,
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
                steps_offset=1,
            )
            ip_vae = AutoencoderKL.from_pretrained(ip_vae_model_path).to(dtype=torch.float16)
            ippipe = StableDiffusionPipeline.from_pretrained(
                ip_adapter_base_model,
                torch_dtype=torch.float16,
                scheduler=noise_scheduler,
                vae=ip_vae,
                feature_extractor=None,
                safety_checker=None
            )
            ippipe = ippipe.to('cuda')
            ip_model = IPAdapterPlus(ippipe, ip_image_encoder_path, ip_ckpt, 'cuda', num_tokens=16)
            
        # move to cpu
        if do_save_memory:
            ip_model.image_encoder = ip_model.image_encoder.to('cpu')
            ip_model.pipe = ip_model.pipe.to('cpu')
            gc.collect()
            torch.cuda.empty_cache()
        else:
            ip_model.image_encoder = ip_model.image_encoder.to('cuda')
            ip_model.pipe = ip_model.pipe.to('cuda')
            

        print("LOADED IP ADAPTER", ip_model)

    if need_music:

        print("LOADING MUSIC MODEL")

        # text to music model
        text2music = MusicGen.get_pretrained('small')

        cfg = {
            "genTextAmount_min": 30,
            "genTextAmount_max": 100,
            "no_repeat_ngram_size": 8,
            "repetition_penalty": 2.0,
            "MIN_ABC": 4,
            "num_beams": 2,
            "temperature": 2.0,
            "MAX_DEPTH": 5
        }

    if need_video:
        video_pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch.float16)
        print('about to die',video_pipe)
        video_pipe.scheduler = DPMSolverMultistepScheduler.from_config(video_pipe.scheduler.config)
        video_pipe.enable_model_cpu_offload()
        video_pipe.enable_vae_slicing()

        if do_save_memory:
            video_pipe = video_pipe.to('cpu')


    if need_deciDiffusion:

        cwd=Path.cwd()

        print("LOADING DECI DIFFUSION MODEL")
        deciDiffusion = StableDiffusionImg2ImgPipeline.from_pretrained('Deci/DeciDiffusion-v1-0',
                                                   custom_pipeline=cwd+'/DeciDiffusion_img2img',#todo fixme
                                                   torch_dtype=torch.float16
                                                   )

        deciDiffusion.unet = deciDiffusion.unet.from_pretrained('Deci/DeciDiffusion-v1-0',
                                                    subfolder='flexible_unet',
                                                    torch_dtype=torch.float16)
        
        #safety checker
        deciDiffusion.safety_checker = None

        # Move pipeline to device
        if do_save_memory:
            deciDiffusion = deciDiffusion.to('cpu')
        else:
            deciDiffusion = deciDiffusion.to('cuda')


    global lcm_img2img
    if need_lcm_img2img:

        print("LOADING LCM IMG2IMG MODEL")

        '''lcm_img2img = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", custom_pipeline="latent_consistency_img2img")
        # To save GPU memory, torch.float16 can be used, but it may compromise image quality.
        lcm_img2img.safety_checker = None
        '''
        tmppipe = AutoPipelineForText2Image.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
        lcm_img2img = AutoPipelineForImage2Image.from_pipe(tmppipe)
        lcm_img2img.safety_checker = None
        del tmppipe

        lcm_img2img = lcm_img2img.to('cuda', torch_dtype=torch.float16)#need it to be float16 (to match vae)

        lcm_img2img.vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesd", torch_dtype=torch.float16)
        

        if do_save_memory:
            #lcm_img2img = lcm_img2img.to('cpu', torch_dtype=torch.float32)
            lcm_img2img = lcm_img2img.to('cpu', torch_dtype=torch.float16)
        else:
            #lcm_img2img = lcm_img2img.to('cuda', torch_dtype=torch.float32)
            lcm_img2img = lcm_img2img.to('cuda', torch_dtype=torch.float16)


def generate_music(description, duration=8, save_dir="./static/samples"):

    description="beautiful music, pleasant calming melody, "+description

    text2music.set_generation_params(duration=duration)
    wav = text2music.generate([description])
    sample_rate = 32000
    # generate unique filename .mp3
    filename = str(uuid.uuid4()) + ".mp3"
    # add filename to save_dir
    filename = os.path.join(save_dir, filename)
    # save file
    wav = wav.cpu()
    normalized_audio_tensor = wav / torch.max(torch.abs(wav))
    # convert tensor to numpy array
    single_audio = normalized_audio_tensor[0, 0, :].numpy()
    sf.write('temp.wav', single_audio, sample_rate)
    AudioSegment.from_wav('temp.wav').export(filename, format='mp3')
    return filename


def generate_attributes(level):
    attributes = ["Strength", "Dexterity", "Wisdom",
                  "Intelligence", "Constitution", "Charisma"]
    total_points = level * 10

    # Generate random partitions of total_points
    partitions = sorted(random.sample(
        range(1, total_points), len(attributes) - 1))
    partitions = [0] + partitions + [total_points]

    # Calculate the differences between adjacent partitions
    attribute_values = {
        attributes[i]: partitions[i + 1] - partitions[i]
        for i in range(len(attributes))
    }

    return attribute_values




def generate_level_and_rarity(level=None):
    # Adjust these probabilities as desired
    if level is None:
        level_probabilities = [0.5, 0.25, 0.15, 0.07, 0.03]
        level = random.choices(range(1, 6), weights=level_probabilities)[0]

    rarity_mapping = {1: "Bronze", 2: "Bronze",
                      3: "Silver", 4: "Silver", 5: "Platinum"}
    rarity = rarity_mapping[level]

    return level, rarity


def generate_image(prompt, prompt_suffix="", width=512, height=512,
                   n_prompt="cropped, collage, composite, (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation",
                   num_inference_steps=15, batch_size=1,
                   ref_image=None,
                   style_fidelity=1.0,
                   attention_auto_machine_weight=1.0,
                   gn_auto_machine_weight=1.0,
                   ref_image_scale=0.6):

    global pipe, ref_pipe
    global ip_pipe

    # add prompt suffix
    prompt += prompt_suffix

    if ref_image is not None:
        '''
        #move pipe to cuda
        ref_pipe = ref_pipe.to("cuda")

        images = ref_pipe([prompt]*batch_size, negative_prompt=[n_prompt]*batch_size,
                          width=width, height=height, num_inference_steps=num_inference_steps, ref_image=ref_image,
                          style_fidelity=style_fidelity,
                          attention_auto_machine_weight=attention_auto_machine_weight,
                          gn_auto_machine_weight=gn_auto_machine_weight
                          ).images


        #move pipe to cpu and clear cache
        ref_pipe = ref_pipe.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        '''
        # use ip adapter
        # move ip_model to cuda
        if do_save_memory:
            ip_model.image_encoder = ip_model.image_encoder.to('cuda')
            ip_model.pipe = ip_model.pipe.to('cuda')

        print("GOT REerence image, scale", ref_image_scale)


        if ip_xl:
            images = ip_model.generate(pil_image=ref_image, num_samples=1, num_inference_steps=30, seed=420,
                                    prompt=prompt+prompt_suffix, scale=ref_image_scale)
        else:
            images = ip_model.generate(pil_image=ref_image, num_samples=1, num_inference_steps=30, seed=420,
                                    prompt=prompt+prompt_suffix, scale=ref_image_scale)

        # move ip_model to cpu
        if do_save_memory:
            ip_model.image_encoder = ip_model.image_encoder.to('cpu')
            ip_model.pipe = ip_model.pipe.to('cpu')
            gc.collect()
            torch.cuda.empty_cache()

    else:
        # move pipe to cuda
        if do_save_memory:
            pipe = pipe.to("cuda")

        images = pipe([prompt]*batch_size, negative_prompt=[n_prompt]*batch_size,
                      width=width, height=height, num_inference_steps=num_inference_steps).images

        # move pipe to cpu and clear cache
        if do_save_memory:
            pipe = pipe.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
        else:
            pipe = pipe.to("cuda")

    # choose top scoring image
    image = images[0]

    return image


def upscale_image(image, prompt,
                  n_prompt="(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation",
                  width=1024, height=1024,
                  num_inference_steps=15,
                  strength=0.25):

    global img2img

    # move img2img to cuda
    if do_save_memory:
        img2img = img2img.to("cuda")
        gc.collect()
        torch.cuda.empty_cache()

    # resize image
    image = image.resize((width, height), Image.LANCZOS)

    img2 = img2img(
        prompt=prompt,
        negative_prompt=n_prompt,
        image=image,
        strength=strength,
        guidance_scale=7.5,
        num_inference_steps=num_inference_steps,
    ).images[0]

    # move to cpu and clear cache
    if do_save_memory:
        img2img = img2img.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
    else:
        img2img = img2img.to("cuda")

    return img2



def hash(s):
    sha256_hash = hashlib.sha256(s.encode('utf-8')).hexdigest()
    return sha256_hash




def process_video(video: str, output: str,exp=2) -> None:    
    #command = f"python {rifeFolder}\\inference_video.py --exp 2 --video {video} --output {output}"
    #print("about to die",command)
    #subprocess.run(command, shell=True, cwd='D:\\img\\ECCV2022-RIFE')
    interpolate_video(video,output,exp=exp)


def generate_video(prompt,output_video_path,upscale=True,base_fps=4):

    global video_pipe

    if do_save_memory:
        video_pipe = video_pipe.to('cuda')

    output_video_path = os.path.abspath(output_video_path)
    output_video_path_up=output_video_path[:-4]+"_up.mp4"
    #create video
    video_frames = video_pipe(prompt, num_inference_steps=20, height=320, width=576, num_frames=12).frames


    if do_save_memory:
        video_pipe = video_pipe.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()

    if upscale:
        #upscale
        video_frames = upscaleFrames(video_frames,prompt,width=1024,height=576)

    #save
    video_path = export_to_video(video_frames,output_video_path,fps=base_fps)
    #upscale
    process_video(output_video_path,output_video_path_up)
    
    return output_video_path_up


def image_to_image(pipeline, image,prompt,strength=0.25,seed=-1,steps=30):

    if seed==-1:
        seed=random.randint(0,100000)

    
    # Call the pipeline function directly
    result = pipeline(prompt=[prompt],
                      image=image,
                      strength=strength,
                      generator=torch.Generator("cuda").manual_seed(seed),
                     num_inference_steps=steps)
    


    img = result.images[0]
    return img

def upscaleFrames0(video_frames,prompt,width=1024,height=576,strength=0.25):

    global deciDiffusion

    if do_save_memory:
        deciDiffusion = deciDiffusion.to('cuda')

    prompt+=", high resolution photograph, detailed, 8k, real life"

    seed=random.randint(0,100000)

    video = [Image.fromarray(frame).resize((width, height)) for frame in video_frames]
    up_frames=[image_to_image(deciDiffusion,frame,prompt,seed=seed,strength=strength) for frame in video]

    if do_save_memory:
        deciDiffusion = deciDiffusion.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()

    return [np.array(x) for x in up_frames]


def upscaleFrames1(video_frames,prompt,width=1024,height=576,strength=0.6,num_inference_steps=3):

    global lcm_img2img

    if do_save_memory:
        lcm_img2img = lcm_img2img.to('cuda')

    seed=random.randint(0,100000)

    prompt+=", high resolution, detailed, 8k"

    video = [Image.fromarray(frame).resize((width, height)) for frame in video_frames]
    up_frames=[lcm_img2img(
        prompt=prompt,
        image=frame,
        strength=strength,
        num_inference_steps=num_inference_steps
        ).images[0] for frame in video]

    if do_save_memory:
        lcm_img2img = lcm_img2img.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()

    return [np.array(x) for x in up_frames]


def upscaleFrames(video_frames,prompt,width=1024,height=576,strength=0.6,num_inference_steps=10):

    global img2img

    if do_save_memory:
        img2img = img2img.to('cuda')

    seed=random.randint(0,100000)

    prompt=prompt+", high resolution, detailed, 8k"

    n_prompt="low resolution, blurry"

    video = [Image.fromarray(frame).resize((width, height)) for frame in video_frames]
    up_frames=[img2img(
        prompt=prompt,
        #negative_prompt=n_prompt,
        image=frame,
        strength=strength,
        num_inference_steps=num_inference_steps
        ).images[0] for frame in video]

    if do_save_memory:
        img2img = img2img.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()

    return [np.array(x) for x in up_frames]



def text_completion0(prompt,max_tokens=60):
    response=llm(
            prompt,
            repeat_penalty=1.2,
            stop=["\n"],
            max_tokens=max_tokens
        )
    
    outputText = response['choices'][0]['text']

    return outputText


import requests

def text_completion(prompt,max_tokens=60,stop_tokens=["\n"]):
    # Define the URL and payload
    url = 'http://localhost:8000/generate'
    payload = {
        "prompt": prompt,
        "use_beam_search": True,
        "n": 4,
        "temperature": 0,
        "stop":stop_tokens,
        "max_tokens": max_tokens,
    }

    # Send POST request
    response = requests.post(url, json=payload)

    data = response.json()
    # Check if the request was successful
    if response.ok:
        #print('Response:', data)
        pass
    else:
        print('Failed to get response. Status code:', response.status_code)

    outputTexts =[ data['text'][i][len(prompt):] for i in range(len(data['text']))]
    #print(outputTexts)

    #for some reason the last one is always the best (which explains why choosing the first one was so bad)
    outputText = outputTexts[-1]

    print("\n\nOUTPUT TEXT:",outputText,"\n\n")

    return outputText


def create_prompt(prompts,max_tokens=60,prompt_hint="",prompt_suppliment=""):

    s=prompt_hint+"\n"
    for prompt in prompts:
        s+="DESCRIPTION:\n"+prompt+"\n"
    
    
    s+="\n> Write a 1-2 sentence description using the following words: "+prompt_suppliment+"\n"

    s+="DESCRIPTION:\n"

    

    print("PROMPT",s)

    outputText = text_completion(s,max_tokens=max_tokens)

    #add prompt suppliment
    outputText= outputText

    print("OUTPUT",outputText)

    return outputText




import json
import subprocess
import shutil
import os
import glob
import hashlib

#animDiffDir = "D:\\img\\animatediff-cli-prompt-travel"
animDiffDir = "D:\\img\\id1"

promptFileName = "noloop_prompt_travel_multi_controlnet.json"



def camera_transform(image_path, frames=8, zoom_percent=0.33, motion=None):
    # Read the image using OpenCV and convert to RGB
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_height, original_width = image.shape[:2]
    mid_x, mid_y = original_width // 2, original_height // 2

    # Default motion to a random choice if None
    if motion is None:
        motions = ["zoom_in", "zoom_out", "pan_left", "pan_right", "pan_up", "pan_down"]
        motion = random.choice(motions)

    #generate a random point (that is not too close to the edge, this depends on zoom_percent)
    random_x0 = (random.random()*(1-zoom_percent)+zoom_percent/2)*original_width
    random_y0 = (random.random()*(1-zoom_percent)+zoom_percent/2)*original_height
    random_x1 = (random.random()*(1-zoom_percent)+zoom_percent/2)*original_width
    random_y1 = (random.random()*(1-zoom_percent)+zoom_percent/2)*original_height

    # Define initial and final states for different motions
    motion_states = {
        "zoom_in": ((mid_x, mid_y, 1), (mid_x, mid_y, zoom_percent)),
        "zoom_in_random": ((mid_x, mid_y, 1), (random_x0, random_y0, zoom_percent)),
        "zoom_out": ((mid_x, mid_y, zoom_percent), (mid_x, mid_y, 1)),
        "zoom_out_random": ((random_x0, random_y0, zoom_percent), (mid_x, mid_y, 1)),
        "pan_left": ((original_width - original_width * zoom_percent / 2, mid_y, zoom_percent), 
                     (original_width * zoom_percent / 2, mid_y, zoom_percent)),
        "pan_right": ((original_width * zoom_percent / 2, mid_y, zoom_percent), 
                      (original_width - original_width * zoom_percent / 2, mid_y, zoom_percent)),
        "pan_up": ((mid_x, original_height - original_height * zoom_percent / 2, zoom_percent), 
                   (mid_x, original_height * zoom_percent / 2, zoom_percent)),
        "pan_down": ((mid_x, original_height * zoom_percent / 2, zoom_percent), 
                     (mid_x, original_height - original_height * zoom_percent / 2, zoom_percent)),
        "pan_random": ((random_x0, random_y0, zoom_percent),(random_x1, random_y1, zoom_percent))
    }

    init_state, final_state = motion_states[motion]

    transformed_images = []

    # Interpolate and create each frame
    for frame in range(frames):
        # Interpolate tx, ty, and s
        tx, ty, s = [np.interp(frame, [0, frames - 1], [init_state[i], final_state[i]]) for i in range(3)]

        # Calculate size of the cropped area
        crop_width, crop_height = int(original_width * s), int(original_height * s)

        
        # Calculate start and end positions for cropping
        start_x = int(tx - crop_width / 2)
        start_y = int(ty - crop_height / 2)
        end_x = start_x + crop_width
        end_y = start_y + crop_height

        #make sure start and end are in-bounds
        if start_x<0 or start_y<0 or end_x>original_width or end_y>original_height:
            print("THIS SHOULD NEVER HAPPEN!",motion,random_x0,random_y0,random_x1,random_y1)
            start_x = 0
            start_y = 0
            end_x = original_width
            end_y = original_height


        # Crop and resize to original size
        cropped_img = image[start_y:end_y, start_x:end_x]
        resized_img = cv2.resize(cropped_img, (original_width, original_height), interpolation=cv2.INTER_LINEAR)

        # Convert to PIL Image and add to list
        pil_img = Image.fromarray(resized_img)
        transformed_images.append(pil_img)

    return transformed_images


def generate_video_camera_transforms(prompt, image, output_video_path, prompt_suffix="", n_prompt=None, num_frames=8,do_upscale=True,width=512,height=512, upscale_size=[1024,576],base_fps=2):

    global lcm_img2img

    #save img to tmp
    image.save("static/samples/temp.png")

    motion = random.choice(["zoom_in", "zoom_out", "zoom_in_random", "zoom_out_random", "pan_random",
                            "pan_left", "pan_right", "pan_up", "pan_down"])
    if motion in ["zoom_in", "zoom_out", "zoom_in_random", "zoom_out_random"]:
        zoom_percent = 0.33
    else:
        zoom_percent = 0.66

    #frames
    frames=camera_transform("static/samples/temp.png",frames=num_frames,zoom_percent=zoom_percent,motion=motion)

    #resize frames
    frames = [frame.resize((upscale_size[0], upscale_size[1] ), Image.LANCZOS) for frame in frames]

    #upscale
    if do_upscale:
        if do_save_memory:
            lcm_img2img = lcm_img2img.to('cuda')
        
        upframes=lcm_img2img(
                            [prompt]*num_frames,
                            frames,
                            width=upscale_size[0],
                            height=upscale_size[1],
                            strength=0.5,
                            num_inference_steps=2
                            ).images

        if do_save_memory:
            lcm_img2img = lcm_img2img.to('cpu')
            gc.collect()
            torch.cuda.empty_cache()

    #interpolate
    output_video_path = os.path.abspath(output_video_path)
    output_video_path_up=output_video_path[:-4]+"_up.mp4"

    video_frames=[np.array(x) for x in upframes]

    #save
    video_path = export_to_video(video_frames,output_video_path,fps=base_fps)
    #upscale
    process_video(output_video_path,output_video_path_up,exp=3)

    return output_video_path_up




def generate_video_animdiff(prompt, image, output_video_path, prompt_suffix="", n_prompt=None, num_frames=8,do_upscale=True,width=512,height=512, upscale_size=[1024,576],base_fps=4):

    # hash prompt to get a unique filename
    sampleFilename = hashlib.sha256(prompt.encode()).hexdigest()[:32]+".webm"


    cwd = os.getcwd()

    # if the file already exists, return the filename
    if os.path.exists(f".//static//samples//{sampleFilename}"):
        return sampleFilename

    # save the image in <animDiffDir>/data/controlnet_images/controlnet_tile/0.png
    image.save(cwd+"/data/controlnet_image/ew/controlnet_tile/0.png")

    #also hsave to controlnet_ip2p
    image.save(cwd+"/data/controlnet_image/ew/controlnet_ip2p/0.png")

    # read the (json formatted)prompt file from <animDiffDir>/config/prompts/<promptFileName>
    with open(cwd+"/config/prompts/"+promptFileName, "r") as promptFile:
        promptFileContent = promptFile.read()
        data = json.loads(promptFileContent)

    # modify data['promptMap'][0]
    data['prompt_map']["0"] = prompt+prompt_suffix
    if n_prompt is None:
        n_prompt = "(worst quality, low quality:1.4),nudity,simple background,border,mouth closed,text, patreon,bed,bedroom,white background,((monochrome)),sketch,(pink body:1.4),7 arms,8 arms,4 arms"
    else:
        n_prompt = "(watermark:1.5), artifacts, (worst quality, low quality:1.4)"+n_prompt


    #for each of the following, we need to replace the intial "./" with cwd
    '''
    data["path"]
    data["motion_module"]
    data["ip_adapter_map"]["input_image_dir"]
    data["controlnet_map"]["input_image_dir"]
    data["controlnet_map"]["controlnet_ref"]["ref_image"]
    data["upscale_config"]["controlnet_ref"]["ref_image"]
    '''
    data["path"]=os.path.abspath(data["path"])
    data["motion_module"]=os.path.abspath(data["motion_module"])
    data["ip_adapter_map"]["input_image_dir"]=os.path.abspath(data["ip_adapter_map"]["input_image_dir"])
    data["controlnet_map"]["input_image_dir"]=os.path.abspath(data["controlnet_map"]["input_image_dir"])
    data["controlnet_map"]["controlnet_ref"]["ref_image"]=os.path.abspath(data["controlnet_map"]["controlnet_ref"]["ref_image"])
    data["upscale_config"]["controlnet_ref"]["ref_image"]=os.path.abspath(data["upscale_config"]["controlnet_ref"]["ref_image"])



    data['n_prompt'][0] = n_prompt

    #save to promptFileName_modified.json
    modified_name=promptFileName[:-5]+"_modified.json"

    # write to file
    with open(cwd+"/config/prompts/"+modified_name, "w") as promptFile:
        json.dump(data, promptFile, indent=4)

    # call animdiff
    # python -m animatediff generate -c <animdiffdir>\config\prompts\<promptFileName> -W 512 -H 512 -L 16 -C 16
    # using popen

    '''

    cmd = ["python", "-m", "animatediff", "generate", "-c",
           f"{animDiffDir}\\config\\prompts\\{promptFileName}", "-W", str(width), "-H", str(height), "-L", str(num_frames), "-C", "16"]

    print(" ".join(cmd))

    result = subprocess.run(cmd, capture_output=True,
                            text=True, cwd=animDiffDir)

    outputFileName = None

    mode = None

    # lines = result.stderr.split('\n')
    lines = result.stdout.split('\n')

    for i, line in enumerate(lines):
        print(line)
        if "Saved sample to" in line and i + 1 < len(lines):

            # we need to concat all lines until we see "Saving frames to"
            outputFileName = ""
            mode = "OutputFileName"
        elif mode == "OutputFileName":
            if "Saving frames to" in line:
                break
            else:
                outputFileName += line.strip()


    '''

    cwd=os.getcwd()

    model_name=Path(cwd+"/runwayml/stable-diffusion-v1-5/")
    config_path=Path(cwd+"/config/prompts/"+modified_name)
    length=num_frames
    context=min(num_frames,16)
    overlap=4
    stride = 0
    repeats=1
    device='cuda'
    use_xformers=True
    force_half_vae=True
    out_dir=Path(cwd+"/output/")
    no_frames=False
    save_merged=False
    version=False

    outputFileName=animateDiff_generate.generate(model_name,config_path,width,height,length,context,overlap,stride,repeats,device,use_xformers,
         force_half_vae,out_dir,no_frames,save_merged,version)
    
    #convert to string
    outputFileName=str(outputFileName)

    print("outputFIleName", outputFileName)

    # remove filename from outputFileName
    #outputFolderName = "\\".join(outputFileName.split('\\')[:-1])
    outputFolderName = outputFileName


    # first find the folder in the same directory as the output file that starts with "00-"
    #s = f"{animDiffDir}//{outputFolderName}//00-*"
    s=f"{outputFolderName}//00-*"
    print(s)
    imagesFolder = glob.glob(s)[0]

    imageFiles = glob.glob(imagesFolder+"//*.png")
    video_frames=[np.array(Image.open(x)) for x in imageFiles]

    output_video_path_orig=output_video_path[:-4]+"_orig.mp4"
    orig_path = export_to_video(video_frames,output_video_path_orig,fps=base_fps)


    if do_upscale:
        #upscale
        video_frames = upscaleFrames1(video_frames,prompt,width=upscale_size[0],height=upscale_size[1])

    output_video_path = os.path.abspath(output_video_path)
    output_video_path_up=output_video_path[:-4]+"_up.mp4"

    #save
    video_path = export_to_video(video_frames,output_video_path,fps=base_fps)
    #upscale
    process_video(output_video_path,output_video_path_up)

    return output_video_path_up




if __name__ == "__main__":
    setup()
    card = generate_card()
    print(card)
