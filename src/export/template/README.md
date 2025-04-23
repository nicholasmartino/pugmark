---
title: Metro Vancouver Building Footprints
emoji: 🏙️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 3.50.2
app_file: app.py
pinned: false
---

# Metro Vancouver Building Footprints

This is a pix2pix model for generating building footprints from satellite imagery. The model was trained on Metro Vancouver data and can be used to predict building footprints from aerial images.

## Usage

Upload a satellite image to get the predicted building footprint. The model works best with 256x256 RGB images of urban areas.

## Model

- Architecture: pix2pix GAN
- Training Dataset: Metro Vancouver satellite imagery and building footprints
- Input: RGB aerial/satellite image
- Output: Building footprint mask

## Example Results

The model can identify building footprints from aerial imagery with reasonable accuracy.

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference 