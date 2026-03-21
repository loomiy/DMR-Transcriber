import os
import whisperx
import gc
import re
from datetime import datetime
from dotenv import load_dotenv
import sqlite3
import shutil
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pathlib import Path

def start_watchdog_with_existing(input_folder="input", output_folder="output", model=None):
    """
    Processes existing files and then starts Watchdog for new files.
    """
    # Process all existing Files
    process_existing_files(input_folder, model, output_folder)

    # Start Watchdog
    event_handler = AudioHandler(model, output_folder)
    observer = Observer()
    observer.schedule(event_handler, path=input_folder, recursive=False)
    observer.start()

    print(f"Watching folder '{input_folder}' for new audio files...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

def process_existing_files(input_folder, model, output_folder=None):
    """
    Processes all existing audio files in the input folder.
    """
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith((".mp3", ".wav", ".m4a")):
            continue
        filepath = os.path.join(input_folder, filename)
        print(f"Processing existing file: {filepath}")
        process_audio_file(filepath, model, output_folder=output_folder)

def process_audio_file(filepath: str, model, db_file="transcriptions.db", output_folder=None):
    """
    Processes a single audio file:
        - Parses the filename
        - Transcribes the audio
        - Writes metadata and transcription to the DB
        - Prints summary info
        - Optionally moves the file to an output folder after processing

    :param filepath: Path to the audio file
    :param model: Initialized Whisper model
    :param db_file: SQLite File Name (default: "transcriptions.db")
    :param output_folder: Optional folder to move the file after processing
    """
    filename = os.path.basename(filepath)

    # Parse Filename
    file_info = parse_filename(filename)

    # Transcribe Audio
    transcription = transcribe_audio(model, filepath)

    # Write to DB (Thread-safe)
    conn = sqlite3.connect(DB_FOLDER + db_file)
    cursor = conn.cursor()

    # Expected that Table exists
    insert_transcription(cursor, filename, file_info, transcription)

    conn.commit()
    conn.close()

    # Print Infos
    print_file_info(file_info, transcription)

    # Move File
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        shutil.move(filepath, os.path.join(output_folder, filename))

class AudioHandler(FileSystemEventHandler):
    """
    Watchdog handler to process new audio files.
    """
    def __init__(self, model, output_folder=None):
        self.model = model
        self.output_folder = output_folder

    def on_created(self, event):
        if event.is_directory:
            return
        if not event.src_path.lower().endswith((".mp3", ".wav", ".m4a")):
            return
        print(f"New audio file detected: {event.src_path}")
        time.sleep(5)
        process_audio_file(event.src_path, self.model, output_folder=self.output_folder)

def print_file_info(file_info: dict, transcription: str):
    """
    Prints structured file info and transcription summary.

    :param file_info: Dictionary with parsed filename info
    :param transcription: String with Transcribed Audio
    """
    print("\n==============================")
    print(f"Filename: {file_info['raw']}")
    print(f"Date: {file_info['date']}")
    print(f"Time: {file_info['time']}")
    print(f"Speaker: {file_info['speaker']}")
    print(f"Channel: {file_info['channel']}")
    print("Transcription:")
    print(transcription)

def insert_transcription(cursor, filename: str, file_info: dict, transcription: str):
    """
    Inserts a transcription entry into the SQLite database.

    :param cursor: SQLite cursor object
    :param filename: Name of the audio file
    :param file_info: Dictionary with parsed filename info
    :param transcription: String with Transcribed Audio
    """
    cursor.execute("""
    INSERT INTO transcriptions (
        filename, date, time, speaker, channel, text
    ) VALUES (?, ?, ?, ?, ?, ?)
    """, (
        filename,
        file_info['date'],
        file_info['time'],
        file_info['speaker'],
        file_info['channel'],
        transcription
    ))

def transcribe_audio(model, filepath: str) -> dict:
    """
    Function to transcribe a single Audio File
    
    :param model: Initialized faster whisper model
    :param filepath: Path to Audio File
    :return: string with transcribed text
    """
    combined = ""

    try:
        audio = whisperx.load_audio(filepath)
        result = model.transcribe(audio, batch_size=BATCH_SIZE)

        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        aligned = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        combined = " ".join(segment["text"] for segment in aligned["segments"])

    except Exception as e:
        print(f"Error at {filepath}: {e}")

    return combined


def parse_filename(filename: str) -> dict:
    """
    Parses an audio filename to extract structured metadata.

    The function expects filenames in the format:
        DATE_TIME_<ChannelName>_TO_<Receiver>_FROM_<Speaker>

    Example:
        20260317_183128XXXX_DMR_Digital_01__02__TO_1_FROM_67469

    Extracted fields:
        - raw (str): Original filename without extension
        - date (str): Date in ISO format (YYYY-MM-DD)
        - time (str): Time in HH:MM:SS
        - datetime (str): Combined ISO datetime (YYYY-MM-DDTHH:MM:SS)
        - channel (str): Name of the channel (e.g., 'XXXX DMR Digital 01 02')
        - channel_receiver (str): Receiver number (from TO_X)
        - speaker (str): Speaker ID (from FROM_X)

    :param filename: The filename to parse
    :return: dict containing the extracted metadata
    """

    name = os.path.splitext(filename)[0]

    data = {
        "raw": name,
        "date": None,
        "time": None,
        "datetime": None,
        "channel": None,
        "channel": None,
        "speaker": None
    }

    # Date + Time
    match_datetime = re.match(r"(\d{8})_(\d{6})", name)

    if match_datetime:
        raw_date = match_datetime.group(1)
        raw_time = match_datetime.group(2)

        try:
            dt = datetime.strptime(raw_date + raw_time, "%Y%m%d%H%M%S")

            data["date"] = dt.strftime("%Y-%m-%d")
            data["time"] = dt.strftime("%H:%M:%S")
            data["datetime"] = dt.isoformat()

        except ValueError:
            print(f"Error while parsing Date/Time in {filename}")

    # Speaker (FROM_XXXX)
    match_user = re.search(r"FROM_(\d+)", name)
    if match_user:
        data["speaker"] = match_user.group(1)

    # Channel Info
    channel_info = name

    channel_info = re.sub(r"^\d{8}_\d{6}", "", channel_info)  # Remove Date from filename
    channel_info = re.sub(r"_?FROM_\d+$", "", channel_info)   # Remove Reciever from filename
    channel_info = channel_info.strip("_")

    # Extract reciever Channel (TO_X)
    cannel_reciever = re.search(r"TO_(\d+)", channel_info)
    if cannel_reciever:
        data["channel_receiver"] = cannel_reciever.group(1)

    # Extract Channel Name
    # Remove TO_X
    channel = re.sub(r"_?TO_\d+", "", channel_info)

    # "__" > " "
    channel = channel.replace("__", " ")

    # "_" > " "
    channel = channel.replace("_", " ")

    # Remove multiple Spaces
    channel = re.sub(r"\s+", " ", channel).strip()

    data["channel"] = channel

    return data

if __name__ == "__main__":

    # Internal Folder Paths
    INPUT_FOLDER = "input"
    OUTPUT_FOLDER = "output"
    DB_FOLDER = "db/"
    MODEL_FOLDER = "models"

    # Load .env File 
    load_dotenv()

    DB_FILE = os.getenv("DB_FILE", "transcriptions.db")

    # Whisper Config
    MODEL_SIZE = os.getenv("MODEL_SIZE", "medium")
    DEVICE = os.getenv("DEVICE", "cuda")
    COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")
    BATCH_SIZE = os.getenv("BATCH_SIZE", "16")

    # Huggingface Token (optional)
    HF_TOKEN =  os.getenv("HF_TOKEN", None)

    # Create Folders if they don't exist
    Path(INPUT_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(DB_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(MODEL_FOLDER).mkdir(parents=True, exist_ok=True)

    print("Environment variables loaded:")
    print(f"INPUT_FOLDER={INPUT_FOLDER}")
    print(f"OUTPUT_FOLDER={OUTPUT_FOLDER}")
    print(f"DB_FILE={DB_FILE}")
    print(f"MODEL_SIZE={MODEL_SIZE}")
    print(f"DEVICE={DEVICE}")
    print(f"COMPUTE_TYPE={COMPUTE_TYPE}")


    # Create SQLite Table if it doesn't exist
    print("Building Database:")
    print(DB_FOLDER + DB_FILE)
    conn = sqlite3.connect(DB_FOLDER + DB_FILE)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transcriptions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        date TEXT,
        time TEXT,
        speaker TEXT,
        channel TEXT,
        text TEXT
    )
    """)

    conn.commit()

    # Load Model
    print("Loading WhisperX:")

    model = whisperx.load_model("large-v2", DEVICE, compute_type=COMPUTE_TYPE, download_root=MODEL_FOLDER)

    # Watchdog loop
    start_watchdog_with_existing(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        model=model
    )