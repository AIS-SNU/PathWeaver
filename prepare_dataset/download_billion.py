import argparse
import requests
import os

def download_file(url, save_dir):
    try:
        file_name = os.path.basename(url)
        save_path = os.path.join(save_dir, file_name)
        
        if os.path.exists(save_path):
            print(f"File already exists at {save_path}. Skipping download.")
            return

        response = requests.get(url, stream=True)
        response.raise_for_status()  # 요청이 성공했는지 확인

        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk: 
                    file.write(chunk)
        
        print(f"File downloaded successfully and saved to {save_path}")
    
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while downloading file: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=True, help="The URL of the file to download")
    parser.add_argument("--save_dir", type=str, required=True, help="The directory where the file will be saved")

    args = parser.parse_args()

    download_file(args.url, args.save_dir)

if __name__ == "__main__":
    main()
