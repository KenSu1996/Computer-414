import os
import random
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim
import torch.utils.data
from torch.optim.lr_scheduler import StepLR
from model import Resnet50_scratch_dag,resnet50_scratch_dag
import numpy as np
import pandas as pd
import swifter
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, rbf_kernel
from collections import Counter





def generate_image_embedding(model, image, transform):
    image = transform(image)
    image = torch.unsqueeze(image, 0).cuda()
    model.cuda().eval()
    embedding = model(image)
    return embedding[1].detach().cpu().numpy()
 


def euclidean_similarity():
    pass



def generate_preds(model, file, root_path, transform, output_path):
    """Function to generate consine similarity of images for face verification"""
    df_table = pd.read_csv(file)
    df_table = df_table.drop(columns=['vgg16', 'resnet50', 'senet50'])
    df_table['img_p1'] = df_table['p1'].swifter.apply(lambda x: cv2.imread(root_path + x))
    df_table['emb_p1'] = df_table['img_p1'].swifter.apply(lambda x: generate_image_embedding(model, x, transform))
    df_table['norm_emb_p1'] = df_table['emb_p1'].swifter.apply(lambda x: x/np.linalg.norm(x))
    df_table['img_p2'] = df_table['p2'].swifter.apply(lambda x: cv2.imread(root_path + x))
    df_table['emb_p2'] = df_table['img_p2'].swifter.apply(lambda x: generate_image_embedding(model, x, transform))
    df_table['norm_emb_p2'] = df_table['emb_p2'].swifter.apply(lambda x: x/np.linalg.norm(x))
    df_table['shape_norm_p1'] = df_table['norm_emb_p1'].swifter.apply(lambda x: x.shape)
    df_table['shape_norm_p2'] = df_table['norm_emb_p2'].swifter.apply(lambda x: x.shape)
    df_table.drop('img_p1', axis=1, inplace=True)
    df_table.drop('img_p2', axis=1, inplace=True)

    df_table.to_csv(path_or_buf=output_path, index=False)
    df_table['VGGFace2'] = df_table[['norm_emb_p1', 'norm_emb_p2']].swifter.apply(lambda x: (cosine_similarity(x["norm_emb_p1"], x["norm_emb_p2"])[0][0]), axis=1)
    df_table.to_csv(path_or_buf=output_path, index=False)
    print(max(df_table['VGGFace2']), flush=True)
    return True


def main():

    torch.backends.cudnn.benchmark = True
  
    model = resnet50_scratch_dag(weights_path = None)
    model.load_state_dict(torch.load("./models/resnet50_scratch_dag.pth", map_location='cpu'))        
    

    transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize((131.0912/255, 103.8827/255, 91.4953/255), (1/255, 1/255, 1/255))])

    
    model.eval()
    generate_preds(model=model, file="./filtered_data.csv", root_path="./Users/jrobby/bfw/bfw-cropped-aligned/", transform=transform, output_path="results/pred.csv")


if __name__ == '__main__':
    main()

