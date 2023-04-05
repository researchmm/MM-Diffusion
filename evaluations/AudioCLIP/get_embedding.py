
import  os, sys
sys.path.append(os.path.dirname (os.path.abspath (__file__)))
from einops import rearrange
import torch.distributed as dist
from model import AudioCLIP
from utils.transforms import ToTensor1D
import torch 
import torchvision as tv
from torchvision.transforms import InterpolationMode

IMAGE_SIZE = 224
IMAGE_MEAN = 0.48145466, 0.4578275, 0.40821073
IMAGE_STD = 0.26862954, 0.26130258, 0.27577711

AUDIO_TRANSFORM = ToTensor1D()
IMAGE_TRANSFORM = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Resize(IMAGE_SIZE, interpolation=InterpolationMode.BICUBIC),
    tv.transforms.CenterCrop(IMAGE_SIZE),
    tv.transforms.Normalize(IMAGE_MEAN, IMAGE_STD)
])
torch.set_grad_enabled(False)
#ROOT_PATHES=[os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../copy/assets'), "/mnt/external/code/guided-diffusion/models/"]
ROOT=os.path.expanduser("~/.cache/mmdiffusion")
def download(fname):
    destination = os.path.join(ROOT, fname)
    if os.path.exists(destination):
        return destination
        
    os.makedirs(ROOT, exist_ok=True)
    download_cammand = f"wget -P {ROOT} https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/{fname}"
    
    os.system(download_cammand)
    return destination



def preprocess_video(videos):
    # videos in {0, ..., 255} as np.uint8 array
    b, f, c, h, w = videos.shape
    # TODO: 
    # videos = videos.float() / 255.
    
    images = rearrange(videos, "b f c h w -> (b f) h w c").to("cpu").numpy()
    images = torch.stack([IMAGE_TRANSFORM(image) for image in images])
 
    videos = rearrange(images, "(b f) c h w -> b f c h w", b=b)
    return videos # [-0.5, 0.5] -> [-1, 1]

def preprocess_audio(audios):
    b,c,l = audios.shape
    # audios = torch.stack([AUDIO_TRANSFORM(track.reshape(1, -1)) for track in audios])
    
    return audios
    




def load_audioclip_pretrained(device=torch.device('cpu')):
    if dist.get_rank()==0:
        filepath = download('AudioCLIP-Full-Training.pt')
    dist.barrier()
    filepath = download('AudioCLIP-Full-Training.pt')
    audioclip = AudioCLIP(pretrained=filepath).to(device)

    return audioclip

def get_audioclip_embeddings_scores(aclp, videos, audios):
    
    videos = preprocess_video(videos).to(aclp.device)
    audios = preprocess_audio(audios).to(aclp.device)

    with torch.no_grad():
        ((audio_features, video_features, _), (logits_audio_video,_ ,_)), _ = aclp(audio=audios, video=videos)
    
    scores_audio_video = torch.diag(logits_audio_video)
    return video_features, audio_features, scores_audio_video


def get_audioclip_a_embeddings(aclp, audios):
    
    
    audios = preprocess_audio(audios).to(aclp.device)

    with torch.no_grad():
        ((audio_features, _, _), _), _ = aclp(audio=audios)
    
   
    return audio_features

def get_audioclip_v_embeddings(aclp, videos):
    videos = preprocess_video(videos).to(aclp.device)

    with torch.no_grad():
        ((_, video_features, _), _), _ = aclp(video=videos)
    
   
    return video_features
