

To run:

clone this repo

install all the requirements [todo: make a requirements.txt]

install https://github.com/s9roll7/animatediff-cli-prompt-travel
and follow the setup steps

this will give you a folder called data which you must copy into your interdimensinalCable folder

to run, you will need to run a server and a helper that generates the actual media

to run the server:

python flaskApp.py

to run the helper:

python .\flaskApp.py --video_chance 0.95 --use_animdiff --start_media_generation --suffix ", high resolution" --model_id D:\img\auto1113\stable-diffusion-webui\models\Stable-diffusion\reliberate_v20.safetensors --image_size 768 424 --num_frames 16 --video_upscale_size 1024 576

I will try and get this running in a colab notebook at some point so I can make sure the entire install process works




