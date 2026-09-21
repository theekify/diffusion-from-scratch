import os

checkpoint_dir = "../outputs/checkpoints"

for filename in os.listdir(checkpoint_dir):
    if filename.startswith("model_epoch"):
        new_name = filename.replace("model_epoch", "cosine_epoch")
        old_path = os.path.join(checkpoint_dir, filename)
        new_path = os.path.join(checkpoint_dir, new_name)
        os.rename(old_path, new_path)
        print(f"renamed {filename} -> {new_name}")