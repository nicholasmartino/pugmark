from Trainer import train

if __name__ == "___main___":
    try:
        train()
    except Exception as e:
        print(f"An error occurred: {e}")
