import argparse
import datetime
import os
import pickle 
import uuid

import torch
import matplotlib.pyplot as plt
import numpy as np 
from PIL import Image
from moviepy.editor import VideoFileClip

from torch.autograd import Variable 
from torchvision import transforms

from .model import EncoderCNN, DecoderRNN


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def transform_image(image, transform=None):
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image
    

def load_model(vocab_path, embed_size, hidden_size, num_layers, encoder_path, decoder_path):
    
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build Models
    encoder = EncoderCNN(embed_size)
    encoder.eval()  # evaluation mode (BN uses moving mean/variance)
    decoder = DecoderRNN(embed_size, hidden_size, 
                         len(vocab), num_layers)
    
    # Load the trained model parameters
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))

    return encoder, decoder, vocab, transform


def caption_video(encoder, decoder, vocab, transform, video, fps=0.1, show=False):
    # Image preprocessing
    report = []
    for i, frame in enumerate(video.iter_frames(fps=fps)):
        time_stamp = datetime.timedelta(seconds=i / fps)

        image = Image.fromarray(frame)
        image = transform_image(image, transform)
        image_tensor = to_var(image, volatile=True)
        
        # If use gpu
        if torch.cuda.is_available():
            encoder.cuda()
            decoder.cuda()
        
        # Generate caption from image
        feature = encoder(image_tensor)
        sampled_ids = decoder.sample(feature)
        sampled_ids = sampled_ids.cpu().data.numpy()
        
        # Decode word_ids to words
        sampled_caption = []
        for word_id in sampled_ids:
            word = vocab.idx2word[word_id]
            if word != '<start>' and word != '<end>':
                sampled_caption.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(sampled_caption)
        
        report.append((str(time_stamp), sentence))

        print(time_stamp, sentence)

        # Print out image and generated caption
        if show:
            plt.axis('off')
            plt.imshow(frame)
            plt.title(sentence)
            # plt.savefig(str(uuid.uuid4()))
            plt.show()

    return report
