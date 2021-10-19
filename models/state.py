
import copy
import torch
import warnings


def get_state_dict(model_path, device= "cuda" if torch.cuda.is_available() else "cpu", jit=True):
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = model.state_dict()
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")
    return state_dict


def merge_fine_tune_CLIP_into_CLIP_ViL(clip_vil_model_path, clip_model, save_path):
    clip_vil_state_dict = get_state_dict(clip_vil_model_path)
    ft_clip_state_dict = clip_model.state_dict()
    clip_vil_keys = list(clip_vil_state_dict.keys())
    ft_clip_keys = list(ft_clip_state_dict.keys())
    new_clip_vil_state_dict = copy.deepcopy(clip_vil_state_dict)
    # Manipulate these keys and replace the old CLIP segment of the clip vil model with this one.
    for ft_clip_key in ft_clip_keys:
        common_keys = [k for k in clip_vil_keys if ft_clip_key in k]
        if len(common_keys) == 1:
            print(common_keys)
            new_clip_vil_state_dict[common_keys[0]] = ft_clip_state_dict[ft_clip_key]
        elif len(common_keys) > 1:
            if ft_clip_key == 'positional_embedding':
                # Use the visual_model.positional_embedding
                new_clip_vil_state_dict[common_keys[0]] = ft_clip_state_dict[ft_clip_key]
            else:
                raise ValueError('Ambiguous, multiple replacement candidates')
    print('Saving...')
    torch.save(new_clip_vil_state_dict, save_path)
    print('DONE')


def extract_CLIP_from_CLIP_ViL(clip_vil_model_path, clip_model):
    clip_vil_state_dict = get_state_dict(clip_vil_model_path)
    ft_clip_state_dict = clip_model.state_dict()
    clip_vil_keys = list(clip_vil_state_dict.keys())
    ft_clip_keys = list(ft_clip_state_dict.keys())
    new_clip_state_dict = copy.deepcopy(ft_clip_state_dict)
    # Manipulate these keys and copy the CLIP-ViL CLIP into this CLIP.
    for ft_clip_key in ft_clip_keys:
        common_keys = [k for k in clip_vil_keys if ft_clip_key in k]
        if len(common_keys) == 1:
            print(common_keys)
            new_clip_state_dict[ft_clip_key] = clip_vil_state_dict[common_keys[0]]
        elif len(common_keys) > 1:
            if ft_clip_key == 'positional_embedding':
                # Use the visual_model.positional_embedding
                new_clip_state_dict[ft_clip_key] = clip_vil_state_dict[common_keys[0]]
            else:
                raise ValueError('Ambiguous, multiple replacement candidates')
    return new_clip_state_dict
    #shapes = {k : w.shape for k, w in clip_model.state_dict().items()}
    #clip_model.load_state_dict(new_clip_state_dict)
    #return clip_model