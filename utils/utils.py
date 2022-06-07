import torch 
import numpy as np 


def Load_state_dict(ckpt, net):
    state_dict = torch.load(ckpt, map_location="cpu")['state_dict']
    model_state_dict = net.state_dict()
    new_state_dict = {}

    count = 0
    total = 0
    for k, v in model_state_dict.items():
        total += 1
        if k in state_dict and v.shape == state_dict[k].shape:
            new_state_dict[k] = state_dict[k]
            count += 1
    
    print(f"{count}/{total}kernel have load the pretrian!!!")

    return new_state_dict


def check_tensor(vector, name=None):
    if isinstance(vector, torch.Tensor):
        if torch.any(torch.isnan(vector)) and torch.any(torch.isinf(vector)):
            return True
    elif isinstance(vector, np.ndarray):
        if np.any(np.isnan(vector)):
            return True 
    elif isinstance(vector, list):
        vector = np.asarray(vector, dtype=np.float32)
        if np.any(np.isnan(vector)):
            return True


if __name__ == '__main__':
    inputs = torch.zeros(1,1,4,4)
    inputs[:,:,0,0] = torch.inf
    
    inputs[:,:,0,1] = torch.nan
    print(inputs)
    print(check_tensor(inputs))

    