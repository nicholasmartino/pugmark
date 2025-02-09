import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--data-dir", type=str, required=True)
    # ... other args ...
    args = parser.parse_args()

    # Your training code here
    # Make sure to save outputs to $AIP_MODEL_DIR if using AI Platform


if __name__ == "__main__":
    main()
