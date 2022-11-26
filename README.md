# testing feature tolerance to transformation
- fMRI data from xu et al. 2022 (https://osf.io/tsz47/wiki/home/) showed increasing tolerance along the hierarchy; highly consistent object representational structure across feature changes towards the end of ventral processing
- These characteristics of tolerance, however, were absent in eight CNNs pretrained with ImageNet images with varying network architecture, depth, the presence/absence of recurrent processing, or whether a network was pretrained with the original or stylized ImageNet images that encouraged shape processing. (either decreasing or inverted u-shape) 
- the original study used stimuli from fMRI data to test feature tolerance (qualitatively different from the imagenet dataset the models trained on)
- We created the imagenet version of the same dataset (to test the tolerarnce to image position, size, stats, sf transformation)
- We tested and compared two models (one resnet, our--resnet with object slot and a decoder to reconstruct objects). We test whether learning to reconstruct objects improves the tolerance to transformation in visual representations and yieding more similar neural response.

# TODOs
- ablation study: Our (recon+classification), Our (recon only), Our (classification)