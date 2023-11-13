from flask import Flask, request, jsonify, send_from_directory
import threading
import queue
import random
import os
import dataset
import re

from datetime import datetime

import argparse

import subprocess


from generation_functions import setup, create_prompt, generate_video, generate_music, generate_image, generate_video_animdiff, generate_video_camera_transforms

from threading import Lock

import numpy as np

import json

import shutil

generate_image_lock = Lock()
text_generation_lock = Lock()



app = Flask(__name__)

db = dataset.connect('sqlite:///media.db')
table = db['media']

#prompt_queue = queue.Queue()
sample_prompts = ["high resolution painting of a beautiful sunset over a gentle lake",
                  "a photograph of a calming forest filled with trees and a river",
                  "a painting of a beautiful beach with waves crashing on the shore",
                  ]

# check if database is empty
if not db['media'].count(type='prompt'):
    print("Database is empty, adding sample prompts")
    # write sample prompts to database
    for prompt in sample_prompts:
        table.insert(dict(type='prompt', prompt=prompt))


lastPrompt=None
lastPromptCount=0

def media_generator():
    global lastPrompt
    #random is getting seeded somewhere ?where? so we need to seed it here
    random.seed()
    while True:
        #check if there's any music or video in the table
        n_music=table.count(type='audio')
        n_video=table.count(type='video')


        #count how many new videos we already have stored
        new_videos_stored=table.count(type='video',status='new')
        print("NEW VIDEOS STORED",new_videos_stored)

        # chance of generating a video
        if new_videos_stored<10 and ((random.random() < args.video_chance and n_music>0) or n_video==0):


            if lastPrompt is not None and lastPromptCount<args.prompt_persistence:
                print("USING LAST PROMPT",lastPrompt)
                prompt = lastPrompt
                lastPromptCount+=1
            else:
                lastPromptCount=1

                #find all prompts with status=new
                newPrompt=table.find_one(type='prompt',status='new')

                #if not prompt_queue.empty():
                #    prompt = prompt_queue.get()
                if newPrompt:
                    prompt=newPrompt['prompt']
                    print("Got prompt:", prompt)
                    # add to database
                    #table.insert(dict(type='prompt', prompt=prompt))
                    #update status to old
                    table.update(dict(type='prompt', prompt=prompt,status='old'),['type','prompt'])

                    print("found new prompt",prompt)

                else:
                    # random_prompt = random.choice(sample_prompts)
                    # choose 5 random prompts from database
                    statement = """
                    SELECT * FROM media
                    WHERE user_created=1
                    ORDER BY RANDOM()
                    LIMIT 5;        
                    """
                    gotPrompts = [x['prompt'] for x in db.query(statement)]
                    #fallback if there are no user created prompts
                    if len(gotPrompts)<5:
                        statement = """
                    SELECT * FROM media
                    ORDER BY RANDOM()
                    LIMIT 5;        
                    """
                        gotPrompts += [x['prompt'] for x in db.query(statement)] 


                    print("Got prompts:", gotPrompts)

                    with text_generation_lock:
                        random.seed()#apparently dataset is doing it?
                        #use prompt suppliment
                        prompt_suppliment=""
                        for prompt_suppliments in all_prompt_suppliments:
                            #prompt_suppliment += random.choice(list(prompt_suppliments.keys()))
                            l=list(prompt_suppliments.keys())
                            index=random.randint(0,len(l)-1)
                            print("index",index)
                            to_add= l[index]
                            print("to_add",to_add)
                            prompt_suppliment += to_add
                            prompt_suppliment += ", "

                        print("prompt_suppliment",prompt_suppliment)

                        #remove last space
                        prompt_suppliment=prompt_suppliment.strip()

                        prompt = create_prompt(gotPrompts,prompt_hint=args.prompt_hint,prompt_suppliment=prompt_suppliment)

                    print("Generated prompt:", prompt)

                #set last prompt
                lastPrompt=prompt


            # let's create a unique filename based off of datetime and the prompt
            # replace non alphanumeric in prompt with _ and trim to 100 chars
            prompt_filename = re.sub(r'\W+', '_', prompt)[:100]
            # prepend timestamp
            prompt_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{prompt_filename}"

            temp_video_path = f"./static/samples/temp_video.mp4"
            video_path = f"./static/samples/{prompt_filename}_video.mp4"

                


            if args.use_animdiff:
                with generate_image_lock:
                    #image=generate_image(prompt, prompt_suffix="", width=576, height=320)
                    image=generate_image(prompt, 
                                         num_inference_steps=args.num_inference_steps,
                                         prompt_suffix=args.suffix, 
                                         width=args.image_size[0], 
                                         height=args.image_size[1])
                    
                    #resize image to video size
                    image=image.resize(args.movie_size)

                    #save image for inspection
                    image.save(f"./static/samples/temp.png")
                    #filename = generateGif(prompt, image, prompt_suffix="", n_prompt=None, num_frames=16,do_upscale=True,width=576,height=320)
                    #video_path = f"./static/samples/{filename}"

                    #temp_video_path_up = generate_video_animdiff(
                    temp_video_path_up = generate_video_camera_transforms(
                                                                 prompt, 
                                                                 image, 
                                                                 temp_video_path, 
                                                                 prompt_suffix="", 
                                                                 n_prompt=args.n_prompt,
                                                                 num_frames=args.num_frames,
                                                                 do_upscale=True,
                                                                 width=args.movie_size[0],
                                                                 height=args.movie_size[1],
                                                                 upscale_size=args.video_upscale_size)

                    if args.mustReencode:
                        # Re-encode the video using FFmpeg
                        cmd = [
                            'ffmpeg', 
                            '-i', temp_video_path_up, 
                            '-c:v', 'libx264', 
                            '-c:a', 'aac', 
                            '-strict', 'experimental',
                            video_path
                        ]
                        subprocess.run(cmd)
                    else:
                        #just copy
                        shutil.copy(temp_video_path_up,video_path)
                
            else:
                print("Generating video")
                with generate_image_lock:
                    temp_video_path_up=generate_video(prompt, temp_video_path)

                # Re-encode the video using FFmpeg
                cmd = [
                    'ffmpeg', 
                    '-i', temp_video_path_up, 
                    '-c:v', 'libx264', 
                    '-c:a', 'aac', 
                    '-strict', 'experimental',
                    video_path
                ]
                subprocess.run(cmd)



            table.insert(dict(type='video', path=video_path, prompt=prompt,status='new'))

        else:
            print("Generating audio")

            # random_prompt = random.choice(sample_prompts)
            # choose 5 random prompts from database
            statement = """
            SELECT * FROM media
            WHERE type='prompt'
            ORDER BY RANDOM()
            LIMIT 5;        
            """
            gotPrompts = [x['prompt'] for x in db.query(statement)]
            print("Got prompts:", gotPrompts)

            with text_generation_lock:
                prompt = create_prompt(gotPrompts,prompt_hint=args.prompt_hint)
            
            print("Generated prompt:", prompt)


            with generate_image_lock:
                audio_path = generate_music(prompt, args.music_duration, "./static/samples")
            table.insert(dict(type='audio', path=audio_path, prompt=prompt,status='new'))



@app.route('/add_prompt', methods=['POST'])
def add_prompt():
    data = request.json
    user_prompt = data['prompt']
    #prompt_queue.put(user_prompt)
    print("Added prompt to queue:", user_prompt)

    #insert prompt into table with status new
    table.insert(dict(type='prompt', prompt=user_prompt, status='new',user_created=True))

    # Generate a new prompt using create_prompt()
    statement = """
    SELECT * FROM media
    ORDER BY RANDOM()
    LIMIT 5;
    """
    gotPrompts = [x['prompt'] for x in db.query(statement)]
    #with text_generation_lock:
    #    generated_prompt = create_prompt(gotPrompts)

    generated_prompt=gotPrompts[0]
    
    return jsonify({'status': 'success', 'random_prompt': generated_prompt})



shown_clips = {}
shown_clips['video'] = []
shown_clips['audio'] = []

@app.route('/get_media', methods=['GET'])
def get_media():
    global shown_clips
    media_type = request.args.get('type')

    #first check if there are any new clips (make sure we are consuming them in order)
    new_clip=table.find_one(type=media_type,status='new',order_by='id')
    if new_clip:
        item = new_clip
        #update status
        table.update(dict(type=media_type, path=item['path'], prompt=item['prompt'],status='old'),['type','path','prompt'])
        return jsonify({'path': item['path'], 'prompt': item['prompt']}), 200

    #get most recent if args.always_get_recent_video
    if media_type=='video' and args.always_get_recent_video:
        item = table.find_one(type=media_type,order_by='-id')
        if item:
            return jsonify({'path': item['path'], 'prompt': item['prompt']}), 200    

    if True:
        #seed random (ugh)
        random.seed()
        #choose between geometric distribution and shuffle
        if random.random()<args.choose_recent_clip_chance:
            #geometric distribution
            k = np.random.geometric(1/args.g_mean)

            #print("getting recent video",k)

            #k cannot be more than count -1
            k=min(k,db['media'].count(type=media_type)-1)

            statement = f"""
SELECT * FROM media
WHERE type='{media_type}'
ORDER BY id DESC
LIMIT 1 OFFSET {k - 1};
"""
            item = list(db.query(statement))[0]

            print("getting recent",k,item['prompt'])

            #make sure we actually got something
            if not item:
                item=db.find_one(type=media_type)

        else:
            
            if media_type not in shown_clips or not shown_clips[media_type]:
                statement = f"""
                SELECT * FROM media
                WHERE type='{media_type}';
                """
                all_clips = list(db.query(statement))
                #seed random (ugh)
                random.seed()
                random.shuffle(all_clips)
                shown_clips[media_type] = all_clips
                #print("Got new clips:", shown_clips[media_type])
                print("shuffling")

            item = shown_clips[media_type].pop(0)


    #verify that the file exists
    if item:
        if os.path.isfile(item['path']):
            return jsonify({'path': item['path'], 'prompt': item['prompt']}), 200
        #if not, try again
        else:
            return get_media()
    return '', 404


@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')


@app.route('/viewer')
def viewer():
    return send_from_directory('templates', 'viewer.html')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_chance', type=float, default=0.5)

    parser.add_argument('--music_duration', type=int, default=30)

    #whether or not to use animatediff
    parser.add_argument('--use_animdiff', action='store_true')

    #model_id
    #default D:\img\auto1113\stable-diffusion-webui\models\Stable-diffusion\dreamlike-photoreal-2.0.safetensors
    parser.add_argument('--model_id', type=str, default="D:/img/auto1113/stable-diffusion-webui/models/Stable-diffusion/dreamlike-photoreal-2.0.safetensors")


    #whether or not to start media generation
    parser.add_argument('--start_media_generation', action='store_true')

    #number of frames to generate
    parser.add_argument('--num_frames', type=int, default=24)

    #gmean, default = 10
    parser.add_argument('--g_mean', type=float, default=10)

    #suffix to add to prompt
    parser.add_argument('--suffix', type=str, default="")

    #prompt hint
    parser.add_argument('--prompt_hint', type=str, default="")

    #image size (default [1024,576])
    parser.add_argument('--image_size', type=int, nargs=2, default=[1024,576])

    #movie size (default [576,320])
    parser.add_argument('--movie_size', type=int, nargs=2, default=[576,320])

    #video upscale size
    parser.add_argument('--video_upscale_size', type=int, nargs=2, default=[1920,1080])

    #negative prompt
    parser.add_argument('--n_prompt', type=str, default="(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation")

    #number of inference steps
    parser.add_argument('--num_inference_steps', type=int, default=20)

    #prompt hint
    parser.add_argument('--prompt_suppliment_files', type=str, nargs='+', default=[])

    #choose recent clip chance
    parser.add_argument('--choose_recent_clip_chance', type=float, default=0.25)

    #always get recent video
    parser.add_argument('--always_get_recent_video', action='store_true')

    #prompt persistence (how many times to repeat the same prompt)
    parser.add_argument('--prompt_persistence', type=int, default=1)

    #args.mustReencode (let's have a flat to set this false)
    parser.add_argument('--donotreencode', dest='mustReencode', action='store_false')
    


    args = parser.parse_args()

    #read prompt suppliment from file (json)
    #this is a k,v file with prompt suppliment as key and weight as value
    all_prompt_suppliments=[]

    for prompt_suppliment_file in args.prompt_suppliment_files:
        with open(prompt_suppliment_file) as f:
            prompt_suppliments=json.load(f)
            all_prompt_suppliments.append(prompt_suppliments)

    if args.start_media_generation:

        if args.use_animdiff:
            print("Using animdiff")
            need_txt2img=True
            need_video=False
            need_deciDiffusion=True
        else:
            print("Using zeroscope")
            need_txt2img=False
            need_video=True
            need_deciDiffusion=True

        setup(
            model_id=args.model_id,
            need_txt2img=need_txt2img,
            need_img2img=False,
            need_ipAdapter=False,
            need_music=True,
            need_video=need_video,
            need_llm=False,#we'll use vllm
            need_deciDiffusion=False,
            need_lcm_img2img=True
        )

        media_thread = threading.Thread(target=media_generator)
        media_thread.start()

    else:
        app.run(port=5000)
