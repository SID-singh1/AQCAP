import os
import zipfile

def package_code_zip():
    zip_filename = "capstone_kaggle_code.zip"
    include_dirs = ["aqcap", "configs", "scripts", "tests", "data/splits"]
    include_files = [
        "README.md", "ROADMAP.md", "PROMPT_FROZEN.md", "requirements.txt",
        "data/juliet_paired.jsonl"
    ]

    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as z:
        for f in include_files:
            if os.path.exists(f):
                z.write(f, arcname=f)
        for d in include_dirs:
            for root, _, files in os.walk(d):
                if "__pycache__" in root or ".pytest_cache" in root:
                    continue
                for f in files:
                    if f.endswith((".pyc", ".tmp", ".pyo")):
                        continue
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, ".")
                    z.write(full_path, arcname=rel_path)

    print(f"[PACKAGE] Successfully updated {zip_filename} ({os.path.getsize(zip_filename) / (1024*1024):.2f} MB)")


def package_upload_zip():
    zip_filename = "upload.zip"
    include_dirs = ["aqcap", "configs", "scripts", "tests", "data"]
    include_files = [
        "README.md", "ROADMAP.md", "PROMPT_FROZEN.md", "requirements.txt"
    ]

    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as z:
        for f in include_files:
            if os.path.exists(f):
                z.write(f, arcname=f)
        for d in include_dirs:
            for root, _, files in os.walk(d):
                if "__pycache__" in root or ".pytest_cache" in root:
                    continue
                for f in files:
                    if f.endswith((".pyc", ".tmp", ".pyo")):
                        continue
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, ".")
                    z.write(full_path, arcname=rel_path)

    print(f"[PACKAGE] Successfully updated {zip_filename} ({os.path.getsize(zip_filename) / (1024*1024):.2f} MB)")


def main():
    package_code_zip()
    package_upload_zip()

if __name__ == "__main__":
    main()

