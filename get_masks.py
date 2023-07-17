import torch
import torchvision
from PIL import Image
import numpy as np

def get_masks(img):
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    mask_rcnn=torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights='DEFAULT').to(device)
    mask_rcnn.eval()
    # img=Image.open(img_path).convert('RGB')
    transforms=torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1.transforms()
    img_tensor=transforms(img).to(device)
    with torch.no_grad():
        pred=mask_rcnn([img_tensor])[0]
        human=pred['labels']==1
        if human.sum()==0:
            return None
        else:
            idx=torch.argmax(pred['scores'][human])
            mask=pred['masks'][human][idx]
            print(mask.max(),mask.min())
            mask[mask>0.5]=1
            mask[mask<0.5]=0
            mask=mask.squeeze(0).cpu().numpy()*255
    return mask

if __name__=='__main__':
    # img_path='./dataset/000134.jpg'
    img_path='./data/tmp/angelina.jpg'
    img=Image.open(img_path).convert('RGB')
    mask=get_masks(img)
    print(mask)
    mask=mask[...,np.newaxis]
    mask=(mask!=0)*img
    mask_im=Image.fromarray(mask.astype('uint8'))
    print(mask.shape,mask.max(),mask.min())
    if mask_im.mode !='RGB':
        mask_im=mask_im.convert('RGB')
    mask_im.show()
    mask_im.save(fp='./data/tmp/pranav_mask.jpg')