import torch

model_path = '/exports/eddie/scratch/s2017377/code/ff/FaceForensics/classification/network/faceforensics++_models_subset/full/xception/full_c23.p'
cuda = True

def get_model():
    # Load model
    if model_path is not None:
        if not cuda:
            model = torch.load(model_path, map_location = "cpu")
        else:
            model = torch.load(model_path)
        print('Model found in {}'.format(model_path))
    else:
        print('No model found, initializing random model.')
    if cuda:
        print("Converting mode to cuda")
        model = model.cuda()
        for param in model.parameters():
            param.requires_grad = True
        print("Converted to cuda")
    return model