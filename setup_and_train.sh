#!/bin/bash
set -e  # Exit on any error

# Configuration
DATASET_PATH="path/to/your/local/dataset"
HF_USERNAME="your-username"
DATASET_NAME="your-dataset-name"
MODEL_NAME="your-model-name"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Step 1: Installing required packages...${NC}"
pip install --upgrade pip
pip install transformers datasets tensorflow wandb huggingface-hub

echo -e "${BLUE}Step 2: Logging into Hugging Face...${NC}"
if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "Please enter your Hugging Face token:"
    read HUGGINGFACE_TOKEN
fi
huggingface-cli login --token $HUGGINGFACE_TOKEN

echo -e "${BLUE}Step 3: Creating dataset repository...${NC}"
huggingface-cli repo create $DATASET_NAME --type dataset || true

echo -e "${BLUE}Step 4: Creating model repository...${NC}"
huggingface-cli repo create $MODEL_NAME --type model || true

echo -e "${BLUE}Step 5: Preparing and uploading dataset...${NC}"
python prepare_dataset.py

echo -e "${BLUE}Step 6: Creating training configuration...${NC}"
cat > training_config.yaml << EOL
compute:
  instance_type: t4-medium
  region: us-east-1

environment:
  huggingface_hub_token: \${HUGGINGFACE_TOKEN}
  env_vars:
    DATASET_PATH: "${HF_USERNAME}/${DATASET_NAME}"
    EPOCHS: "10"
    BATCH_SIZE: "1"
    WANDB_API_KEY: \${WANDB_API_KEY}

training:
  command: python train.py
  repository: ${HF_USERNAME}/${MODEL_NAME}
  package:
    - train.py
    - pix2pix.py
    - requirements.txt
EOL

echo -e "${BLUE}Step 7: Creating requirements.txt...${NC}"
cat > requirements.txt << EOL
transformers>=4.30.0
datasets>=2.12.0
tensorflow>=2.12.0
wandb
numpy
EOL

echo -e "${BLUE}Step 8: Starting training on Hugging Face...${NC}"
huggingface-cli run-training training_config.yaml

echo -e "${GREEN}Setup and training launch complete!${NC}" 