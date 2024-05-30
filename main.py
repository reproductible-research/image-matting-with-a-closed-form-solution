#!/usr/bin/env python 
'''
Copyright (C) 2024 Aissa Abdelaziz, Mahdi Ranjbar, and Mohammad Ali Jauhar.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
'''
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import spdiags, csr_matrix
from scipy.ndimage import binary_erosion
import scipy.sparse
import scipy.sparse.linalg
import logging
import sys




#Calcualte Matting Laplacian 
def getLaplacian(I, consts, epsilon=0.00001, win_size=1):
    '''
    This function is used to calculate the laplacian matting as described in the paper
    and return the sparse matrix of L  (Eq. 21)
    '''
    logging.info('Computing Matting Laplacian ...')
    neb_size = (win_size * 2 + 1) ** 2
    h, w, c = I.shape
    img_size = w * h
    consts = binary_erosion(consts, structure=np.ones((win_size * 2 + 1, win_size * 2 + 1)))
    indsM = np.arange(img_size).reshape(h, w)
    tlen = int(((1 - consts[win_size:-win_size, win_size:-win_size]).sum()) * (neb_size ** 2))
    row_inds = np.zeros(tlen)
    col_inds = np.zeros(tlen)
    vals = np.zeros(tlen)
    len_ = 0
    for j in range(win_size, w - win_size):
        for i in range(win_size, h - win_size):
            if consts[i, j]:
               continue
            win_inds = indsM[i - win_size:i + win_size + 1, j - win_size:j + win_size + 1]
            win_inds = win_inds.flatten()
            winI = I[i - win_size:i + win_size + 1, j - win_size:j + win_size + 1, :]
            winI = winI.reshape(neb_size, c)
            win_mu = np.mean(winI, axis=0)
            win_var = np.linalg.inv(winI.T @ winI / neb_size - np.outer(win_mu, win_mu) + epsilon / neb_size * np.eye(c))
            winI = winI - win_mu
            tvals = (1 + winI @ win_var @ winI.T) / neb_size
            
            row_inds[len_:len_ + neb_size ** 2] = np.repeat(win_inds, neb_size)
            col_inds[len_:len_ + neb_size ** 2] = np.tile(win_inds, neb_size)
            vals[len_:len_ + neb_size ** 2] = tvals.flatten()
            len_ += neb_size ** 2
    vals = vals[:len_]
    row_inds = row_inds[:len_]
    col_inds = col_inds[:len_]
    A = csr_matrix((vals, (row_inds, col_inds)), shape=(img_size, img_size))
    sumA = A.sum(axis=1)
    A = spdiags(sumA.flatten(), 0, img_size, img_size) - A
    return A


def main():
    import argparse
    logging.basicConfig(level=logging.INFO)
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument('image', type=str, help='input image')
    arg_parser.add_argument('-s', '--scribbles', type=str, help='input scribbles')
    #arg_parser.add_argument('-o', '--output', type=str, required=True, help='output image')    
    args = arg_parser.parse_args()
    image_input = cv2.imread(args.image, cv2.IMREAD_COLOR)  #read the input image
    #Normalization
    image=image_input/ 255.0

    scribbles_input = cv2.imread(args.scribbles, cv2.IMREAD_COLOR) #read the scribbles image
    scribbles= scribbles_input / 255.0
    #print(image_input.shape)
    #print(scribbles_input.shape)
    if image_input.shape != scribbles_input.shape:
        print("Error: There was a problem with the user input.")
        sys.exit()
    prior = np.sign(np.sum(image - scribbles, axis=2)) / 2 + 0.5 
    #Constant Map 
    consts_map = prior != 0.5 
    #Calculating Matting Laplacian 
    laplacian = getLaplacian(image, consts_map)
    scribbles_confidence=100 #the scribbles confidences
    prior_confidence = scribbles_confidence * consts_map
    confidence = scipy.sparse.diags(prior_confidence.flatten())
    logging.info('Solving the Linear System ...')
    solution = scipy.sparse.linalg.spsolve(laplacian + confidence,prior.flatten() * prior_confidence.flatten()) #Solving the linear system
    #Ensure that the result lies within the range [0, 1]
    alpha = np.clip(solution.reshape(prior.shape), 0, 1)
    #Save the images
    cv2.imwrite("output.png", (1-alpha) * 255.0)




if __name__ == "__main__":
    main()
