import os
import zipfile

def package_project():
    zip_filename = "AI_Project_GroupE_LorentzianClassification.zip"
    print(f"Starting packaging of semester project into: {zip_filename}...")
    
    # List of directories to include (exactly per submission guidelines)
    dirs_to_include = [
        "report",
        "slides",
        "code",
        "results",
        "dataset",
        "model",
        "contribution"
    ]

    
    # Ensure all directories exist
    for d in dirs_to_include:
        if not os.path.exists(d):
            print(f"Warning: Directory {d} does not exist. Creating it.")
            os.makedirs(d, exist_ok=True)
            
    # Create the zip archive
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for folder in dirs_to_include:
            print(f"Adding folder: {folder}/")
            for root, dirs, files in os.walk(folder):
                # Exclude __pycache__ folders
                if "__pycache__" in root:
                    continue
                for file in files:
                    file_path = os.path.join(root, file)
                    # Keep relative path starting with the directory name itself
                    arcname = os.path.relpath(file_path, start=os.path.dirname(folder))
                    zipf.write(file_path, arcname)
                    print(f"  + {arcname}")
                    
    print(f"\nSuccessfully created and packaged zip archive: {zip_filename}")
    size_mb = os.path.getsize(zip_filename) / (1024 * 1024)
    print(f"Zip file size: {size_mb:.2f} MB")

if __name__ == "__main__":
    package_project()
