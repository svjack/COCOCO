https://github.com/svjack/COCOCO

sudo apt-get update
sudo apt-get install git-lfs ffmpeg cbm
git clone https://github.com/svjack/COCOCO && cd COCOCO
pip install -r requirements.txt
pip install -e .

git clone https://huggingface.co/svjack/CoCoCo

huggingface-cli login

from huggingface_hub import snapshot_download
import os

def download_model(repo_id_with_username):
    # 拆分 repo_id_with_username 为 username 和 repo_id
    username, repo_id = repo_id_with_username.split('/')

    # 构建完整的 repo_id
    full_repo_id = f"{username}/{repo_id}"

    # 构建本地目录路径
    local_dir = os.path.join("./", repo_id)

    # 确保本地目录存在，如果不存在则创建
    os.makedirs(local_dir, exist_ok=True)

    # 下载模型文件夹到指定目录，不使用缓存
    snapshot_download(repo_id=full_repo_id, local_dir=local_dir, cache_dir=None)

    print(f"Model downloaded to: {local_dir}")

download_model("benjamin-paine/stable-diffusion-v1-5-inpainting")
import os
os.remove("stable-diffusion-v1-5-inpainting/config.json")

'''
rename model file in stable-diffusion-v1-5-inpainting to stable-diffusion-v1-5-inpainting-f16
safetensors use float16 replace ori
'''

download_model("facebook/sam2-hiera-large")

from huggingface_hub import upload_folder, create_repo
import os

def upload_model(local_dir, repo_id_with_username, repo_type="model"):
    # 拆分 repo_id_with_username 为 username 和 repo_id
    username, repo_id = repo_id_with_username.split('/')
    # 构建完整的 repo_id
    full_repo_id = f"{username}/{repo_id}"
    # 确保本地目录存在
    if not os.path.exists(local_dir):
        raise FileNotFoundError(f"Local directory {local_dir} does not exist.")
    # 检查仓库是否存在，如果不存在则创建
    try:
        create_repo(repo_id=full_repo_id, repo_type=repo_type, exist_ok=True)
    except Exception as e:
        print(f"Failed to create repository: {e}")
        # 不使用 return，继续执行上传操作
    # 上传本地文件夹到指定的仓库
    try:
        upload_folder(
            folder_path=local_dir,
            repo_id=full_repo_id,
            repo_type=repo_type,
            create_pr=False,  # 不创建PR
            allow_patterns="*",  # 上传所有文件
            ignore_patterns=None,  # 不忽略任何文件
        )
        print(f"Folder uploaded to: {full_repo_id}")
    except Exception as e:
        print(f"Failed to upload folder: {e}")

# 示例调用
upload_model("stable-diffusion-v1-5-inpainting-f16", "svjack/stable-diffusion-v1-5-inpainting-f16", "model")

### sam2 may require high version gpu

CUDA_VISIBLE_DEVICES=0,1 python app.py \
--config ./configs/code_release.yaml \
--model_path CoCoCo \
--pretrain_model_path stable-diffusion-v1-5-inpainting \
--sub_folder unet

推荐使用 export port 的方法 在公网上调用
而且app share为True

python app.py \
--config ./configs/code_release.yaml \
--model_path CoCoCo \
--pretrain_model_path stable-diffusion-v1-5-inpainting-f16 \
--sub_folder unet

meet core dumpd error
https://github.com/zibojia/COCOCO/issues/11

may sam predictor not works

import PIL
import requests
import torch
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline

def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

init_image = download_image(img_url).resize((512, 512))
mask_image = download_image(mask_url).resize((512, 512))

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
