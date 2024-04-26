import os
from pathlib import Path
import argparse
import hashlib


def compute_sha256(file_path):
    """
    Compute the SHA-256 hash of the file specified by file_path.
    
    Args:
    file_path (str): The path to the file.
    
    Returns:
    str: The SHA-256 hash of the file in hexadecimal format.
    """
    # Create a sha256 hash object
    hash_sha256 = hashlib.sha256()
    
    # Open the file in binary mode and read chunks
    with open(file_path, 'rb') as file:
        for chunk in iter(lambda: file.read(4096), b""):
            hash_sha256.update(chunk)
    
    # Return the hexadecimal representation of the digest
    return hash_sha256.hexdigest()


def download_jamendo_dataset(output_dir):

    hash_url = "https://essentia.upf.edu/datasets/mtg-jamendo/autotagging_moodtheme/audio/checksums_sha256.txt"
    cmd = f"""wget -O {os.path.join(output_dir, "checksums_sha256.txt")} {hash_url}"""
    os.system(cmd)

    with open(os.path.join(output_dir, "checksums_sha256.txt"), "r") as fp:
        hashes = fp.readlines()

    hash_dict = {}
    for sha256_hash in hashes:
        value = sha256_hash.split(" ")[0]
        fname = sha256_hash.split(" ")[1].strip("\n")
        hash_dict[fname] = value

    # Mood/theme subset is divided into 100 files
    for n in range(100):
        base_url = (
            "https://essentia.upf.edu/datasets/mtg-jamendo/autotagging_moodtheme/audio/"
        )
        fname = f"autotagging_moodtheme_audio-{n:02}.tar"
        url = base_url + fname
        
        # Check if file has been downloaded
        if os.path.isfile(os.path.join(output_dir, fname)):

            # Compute hash for downloaded file
            sha256_hash = compute_sha256(os.path.join(output_dir, fname))

            # Check against dictionary
            if sha256_hash == hash_dict[fname]:
                print(f"Checksum PASSED. Skipping {fname}...")
                continue
            else:
                print("Checksum FAILED. Re-downloading...")

        cmd = f"wget -O {os.path.join(output_dir, fname)} {url}"
        os.system(cmd)

    for n in range(100):
        fname = f"autotagging_moodtheme_audio-{n:02}.tar"
        # Untar
        print(f"Extracting {fname}...")
        cmd = f"tar -xvf {os.path.join(output_dir, fname)} -C {output_dir}"
        os.system(cmd)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Path to data directory')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    mtg_dir = data_dir / "mtg_jamendo"

    if not mtg_dir.exists():
        mtg_dir.mkdir(parents=True, exist_ok=True)
 
    download_jamendo_dataset(mtg_dir)
  

if __name__ == '__main__':
    main()