import torch
import torchvision.transforms as transforms

def get_bbox_image(img, bbox):
    '''
    Extract image from bounding box the and resize it to 28x28.
    '''
    x = int(bbox[0])
    y = int(bbox[1])
    if x < 0: x = 0
    if y < 0: y = 0
    
    w = int(bbox[2]-x)
    h = int(bbox[3]-y)
    if w == 0 or h == 0:
        w = 127
        h = 127
    
    img = img.numpy()
    
    new_img = img[:, y:y+h, x:x+w]
    new_img = torch.tensor(new_img, dtype=torch.float32)
    new_img = transforms.Resize(size=(28, 28))(new_img) 
    
    return new_img

def extract_image_from_bbox(x_train, w_train):
    out = []
    for x, w in zip(x_train, w_train):
        x_star = get_bbox_image(x, w)
        out.append(x_star)
    return out

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        super().__init__()
        if isinstance(X, list):
            self.X = extract_image_from_bbox(*X)
            self.y = y
        else:
            self.X = X
            self.y = y
    
    def describe(self, *args, **kwargs):
        pass

    def decide_transform(self, *args, **kwargs):
        pass
 
    @property
    def class_labels(self):
        return ['0', '1', '2', '3', '4']
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        Xi = self.X[i]
        yi = self.y[i] if self.y is not None else torch.Tensor([float('nan')])
        return Xi, yi
