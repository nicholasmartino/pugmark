# Building footprint generator

Building footprint generator based on the [pix2pix](https://www.tensorflow.org/tutorials/generative/pix2pix) model using conditional Generative Adversarial Networks. Training data was sourced from Statistics Canada's [Open Database of Buildings]('https://www.statcan.gc.ca/eng/lode/databases/odb').

1. Connect GitHub repository on Google Cloud console

2. Run the following setup scripts

```bash
# Authenticate with your Google account
gcloud auth login

chmod +x gcloud_setup.sh
chmod +x gcloud_submit_job.sh

./setup_gcloud.sh
```

## Results

![](https://raw.githubusercontent.com/nicholasmartino/pugmar/master/footprints_gen/150epochs/fg3.png)
![](https://raw.githubusercontent.com/nicholasmartino/pugmar/master/footprints_gen/150epochs/fg1.png)
![](https://raw.githubusercontent.com/nicholasmartino/pugmark/master/footprints_gen/150epochs/fg4.png)

## License

[![cc-by-image](https://i.creativecommons.org/l/by/4.0/88x31.png)](https://creativecommons.org/licenses/by/4.0/)
