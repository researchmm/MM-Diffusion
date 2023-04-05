from email.policy import strict
import requests
from tqdm import tqdm
import os
import torch
import torch.distributed as dist

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 8192

    pbar = tqdm(total=0, unit='iB', unit_scale=True)
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    pbar.close()

ROOT=os.path.expanduser("~/.cache/mmdiffusion")
def download(id, fname):
    
    destination = os.path.join(ROOT, fname)
    if os.path.exists(destination):
        return destination 

    os.makedirs(ROOT, exist_ok=True)
    destination = os.path.join(ROOT, fname)
    
    URL = 'https://drive.google.com/uc?export=download'
    session = requests.Session()
  
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)
    return destination


_I3D_PRETRAINED_ID = '1mQK8KD8G6UWRa5t87SRMm5PVXtlpneJT'
def load_i3d_pretrained(device=torch.device('cpu')):
    from .pytorch_i3d import InceptionI3d
    i3d = InceptionI3d(400, in_channels=3).to(device)

    if dist.get_rank()==0:
        filepath = download(_I3D_PRETRAINED_ID, 'i3d_pretrained_400.pt')
    dist.barrier()
    filepath = download(_I3D_PRETRAINED_ID, 'i3d_pretrained_400.pt')
    is_strict=True
    state_dict=torch.load(filepath, map_location=device)

    i3d.load_state_dict(state_dict, strict=is_strict)
    i3d.eval()
    return i3d

def load_i3d_pretrained_classifier(device=torch.device('cpu'), num_class=400):
    from .pytorch_i3d import InceptionI3d_Classifier
    i3d = InceptionI3d_Classifier(num_class, in_channels=3).to(device)
    filepath = download(_I3D_PRETRAINED_ID, 'i3d_pretrained_400.pt')
    is_strict=True
    state_dict=torch.load(filepath, map_location=device)
    if num_class!=400:
        state_dict.pop("logits.conv3d.weight")
        state_dict.pop("logits.conv3d.bias")
        is_strict=False
    i3d.load_state_dict(state_dict, strict=is_strict)

    return i3d

