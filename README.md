# Counterfactual Attacks

Code, data, and figures accompanying the paper, *Unifying Image Counterfactuals and Feature Attributions with Latent-Space Adversarial Attacks*. The function ```generate_counterfactuals``` is our Counterfactual Attacks algorithm. 

## CelebA Details

Neural networks predicting CelebA attributes can be trained by running

```
python fit_attribute_predictors.py <<<ATTRIBUTE>>>
```

Our analyses ran this with `<<<ATTRIBUTE>>>` as Attractive, Young, Male, and Smiling.

This repository requires training Nvidia's Pytorch implementation of [StyleGAN3](https://github.com/NVlabs/stylegan3) on the CelebA dataset. To do so, it is necessary to create a virtual environment following its setup instructions. After training the StyleGAN model, ```project_images.py``` projects images into StyleSpace; its functions are from ```projector.py``` in [this repository](https://github.com/ouhenio/stylegan3-projector).
