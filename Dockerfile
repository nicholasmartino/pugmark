FROM huggingface/autotrain-advanced:latest
CMD autotrain setup && autotrain app --port 7860
