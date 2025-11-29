# data_loader.py - Fixed folder browser
import pandas as pd
import os
import requests
import zipfile
import tempfile
from urllib.parse import urlparse
from utils import ensure_dir, safe_filename
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

def load_file_or_url(path_or_url, out_dir, progress_callback=None):
    """
    Accepts:
      - local path to CSV/Excel
      - http(s) URL to CSV/Excel (will download to out_dir)
      - kaggle://dataset/owner/slug
      - gdrive://file-id or gdrive-folder://folder-id
    Returns (df, saved_path)
    """
    ensure_dir(out_dir)
    
    if progress_callback:
        progress_callback(0, "Starting download...")
    
    # Kaggle dataset
    if path_or_url.startswith("kaggle://"):
        return _download_kaggle_dataset(path_or_url, out_dir, progress_callback)
    
    # Google Drive file
    elif path_or_url.startswith("gdrive://"):
        return _download_gdrive_file(path_or_url, out_dir, progress_callback)
    
    # Google Drive folder
    elif path_or_url.startswith("gdrive-folder://"):
        return _download_gdrive_folder(path_or_url, out_dir, progress_callback)
    
    # Regular HTTP URL
    elif path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return _download_http_file(path_or_url, out_dir, progress_callback)
    
    # Local file
    else:
        if not os.path.exists(path_or_url):
            raise FileNotFoundError(path_or_url)
        if progress_callback:
            progress_callback(100, "Loading local file...")
        return _read_file(path_or_url), os.path.abspath(path_or_url)

def _download_http_file(url, out_dir, progress_callback=None):
    """Download regular HTTP/HTTPS file"""
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise ValueError(f"Download failed: {r.status_code}")
    
    total_size = int(r.headers.get('content-length', 0))
    parsed = urlparse(url)
    fname = os.path.basename(parsed.path) or "download.csv"
    fname = safe_filename(fname)
    saved = os.path.join(out_dir, fname)
    
    downloaded = 0
    with open(saved, "wb") as f:
        for chunk in r.iter_content(32768):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if progress_callback and total_size > 0:
                    progress = int((downloaded / total_size) * 100)
                    progress_callback(progress, f"Downloading... {progress}%")
    
    if progress_callback:
        progress_callback(100, "Download complete!")
    return _read_file(saved), saved

def _download_kaggle_dataset(kaggle_url, out_dir, progress_callback=None):
    """Download Kaggle dataset"""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise ImportError("Kaggle API not installed. Run: pip install kaggle")
    
    # Parse kaggle://dataset/owner/slug
    parts = kaggle_url.replace("kaggle://", "").split("/")
    if len(parts) < 3:
        raise ValueError("Invalid Kaggle URL format. Use: kaggle://dataset/owner/slug")
    
    dataset_type, owner, slug = parts[0], parts[1], parts[2]
    
    if progress_callback:
        progress_callback(10, "Authenticating with Kaggle...")
    
    api = KaggleApi()
    api.authenticate()
    
    if progress_callback:
        progress_callback(30, f"Downloading {owner}/{slug}...")
    
    # Download to temp directory first
    temp_dir = tempfile.mkdtemp()
    api.dataset_download_files(f"{owner}/{slug}", path=temp_dir, quiet=False)
    
    if progress_callback:
        progress_callback(80, "Extracting files...")
    
    # Find and extract zip file
    zip_files = [f for f in os.listdir(temp_dir) if f.endswith('.zip')]
    if not zip_files:
        raise ValueError("No zip file found in downloaded dataset")
    
    zip_path = os.path.join(temp_dir, zip_files[0])
    extract_dir = os.path.join(out_dir, f"kaggle_{owner}_{slug}")
    ensure_dir(extract_dir)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    if progress_callback:
        progress_callback(100, "Kaggle download complete!")
    
    # Return first CSV file found
    csv_files = [f for f in os.listdir(extract_dir) if f.endswith('.csv')]
    if csv_files:
        first_csv = os.path.join(extract_dir, csv_files[0])
        return _read_file(first_csv), first_csv
    else:
        # Return the directory path if no CSV found
        return None, extract_dir

def _download_gdrive_file(gdrive_url, out_dir, progress_callback=None):
    """Download Google Drive file"""
    file_id = gdrive_url.replace("gdrive://", "").strip()
    return _download_gdrive_file_by_id(file_id, out_dir, progress_callback)

def _download_gdrive_folder(gdrive_url, out_dir, progress_callback=None):
    """Download Google Drive folder"""
    folder_id = gdrive_url.replace("gdrive-folder://", "").strip()
    
    if progress_callback:
        progress_callback(10, "Getting folder contents...")
    
    # We'll just download the first file in the folder for now
    # In a real implementation, you'd list all files and let user choose
    try:
        import gdown
    except ImportError:
        raise ImportError("gdown not installed. Run: pip install gdown")
    
    # This is a simplified implementation - in practice you'd need to use Google Drive API
    # to list folder contents. Using gdown with folder ID directly:
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    output = os.path.join(out_dir, f"gdrive_folder_{folder_id}")
    ensure_dir(output)
    
    if progress_callback:
        progress_callback(50, "Downloading folder (this may take a while)...")
    
    gdown.download_folder(url, output=output, quiet=True)
    
    if progress_callback:
        progress_callback(100, "Folder download complete!")
    
    # Find first CSV file
    csv_files = []
    for root, dirs, files in os.walk(output):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    if csv_files:
        return _read_file(csv_files[0]), csv_files[0]
    else:
        return None, output

def _download_gdrive_file_by_id(file_id, out_dir, progress_callback=None):
    """Download Google Drive file by ID"""
    try:
        import gdown
    except ImportError:
        raise ImportError("gdown not installed. Run: pip install gdown")
    
    url = f"https://drive.google.com/uc?id={file_id}"
    output = os.path.join(out_dir, f"gdrive_{file_id}.csv")
    
    if progress_callback:
        progress_callback(50, "Downloading from Google Drive...")
    
    gdown.download(url, output, quiet=False)
    
    if progress_callback:
        progress_callback(100, "Google Drive download complete!")
    
    return _read_file(output), output

def _read_file(path):
    if path.lower().endswith((".xls",".xlsx")):
        xls = pd.ExcelFile(path)
        df = pd.read_excel(xls, sheet_name=0)
        df.__excel_sheets__ = xls.sheet_names
        return df
    elif path.lower().endswith('.csv'):
        df = pd.read_csv(path)
        return df
    else:
        # For non-CSV/Excel files, return None but keep the path
        return None

def browse_and_preview_folder(master, preview_callback, info_callback):
    """Open folder browser and set up file list preview"""
    folder_path = filedialog.askdirectory(title="Select Folder")
    if not folder_path:
        return
    
    # Create a popup window to show folder contents
    folder_window = tk.Toplevel(master)
    folder_window.title(f"Folder: {os.path.basename(folder_path)}")
    folder_window.geometry("600x400")
    
    # File list with scrollbar
    frame = ttk.Frame(folder_window)
    frame.pack(fill="both", expand=True, padx=5, pady=5)
    
    listbox = tk.Listbox(frame)
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=listbox.yview)
    listbox.configure(yscrollcommand=scrollbar.set)
    
    listbox.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    # Load files
    supported_extensions = ('.csv', '.xlsx', '.xls', '.txt', '.pdf', '.docx')
    files = []
    for file in sorted(os.listdir(folder_path)):
        if file.lower().endswith(supported_extensions):
            files.append(file)
    
    for file in files:
        listbox.insert("end", file)
    
    def on_file_select(event):
        selection = listbox.curselection()
        if selection:
            filename = listbox.get(selection[0])
            full_path = os.path.join(folder_path, filename)
            
            # Update info callback
            file_info = f"File: {filename}\nPath: {full_path}\nSize: {os.path.getsize(full_path)} bytes\nType: {os.path.splitext(filename)[1]}"
            info_callback(file_info)
            
            # Try to preview if it's a data file
            if filename.lower().endswith(('.csv', '.xlsx', '.xls')):
                try:
                    df = _read_file(full_path)
                    if df is not None:
                        preview_callback(df.head(20))
                    else:
                        preview_callback(f"Preview not available for {filename}")
                except Exception as e:
                    preview_callback(f"Error reading {filename}: {str(e)}")
            else:
                preview_callback(f"Preview not available for {filename}\n\n{file_info}")
    
    listbox.bind("<<ListboxSelect>>", on_file_select)
    
    # Double-click to load
    def on_double_click(event):
        selection = listbox.curselection()
        if selection:
            filename = listbox.get(selection[0])
            full_path = os.path.join(folder_path, filename)
            if filename.lower().endswith(('.csv', '.xlsx', '.xls')):
                try:
                    df = _read_file(full_path)
                    if df is not None:
                        preview_callback(df)
                        folder_window.destroy()
                except Exception as e:
                    messagebox.showerror("Error", f"Could not load {filename}: {str(e)}")
    
    listbox.bind("<Double-1>", on_double_click)