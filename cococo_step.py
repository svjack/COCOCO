https://github.com/svjack/COCOCO

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

CUDA_VISIBLE_DEVICES=0,1 python app.py \
--config ./configs/code_release.yaml \
--model_path CoCoCo \
--pretrain_model_path stable-diffusion-v1-5-inpainting \
--sub_folder unet

python app.py \
--config ./configs/code_release.yaml \
--model_path CoCoCo \
--pretrain_model_path stable-diffusion-v1-5-inpainting-f16 \
--sub_folder unet
