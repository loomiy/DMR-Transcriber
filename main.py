import os
import re
import time
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from faster_whisper import WhisperModel
from dotenv import load_dotenv
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a")

# Common Whisper hallucinations on silence / noise (lowercased, no trailing dots).
# Files whose entire output matches one of these are treated as "no speech".
HALLUCINATION_PHRASES = {
    "vielen dank",
    "danke",
    "tschüss",
    "das war's",
    "bis zum nächsten mal",
    "danke fürs zuschauen",
    "untertitel",
    "untertitelung des zdf",
    "amara.org",
    "untertitel von",
    "copyright",
    "you",
    "thank you",
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Runtime configuration, mostly populated from environment variables."""
    input_folder: str = "input"
    output_folder: str = "output"
    db_folder: str = "db"
    model_folder: str = "models"

    db_file: str = "transcriptions.db"
    model_size: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str = "auto"            # "auto" or a code like "de", "en", "fr"
    beam_size: int = 10
    initial_prompt: str | None = None
    hotwords: str | None = None

    # Thresholds for the "nothing understood" detection
    no_speech_threshold: float = 0.6   # segment counts as silence above this
    logprob_threshold: float = -1.0    # below this the result is "low_confidence"

    @property
    def db_path(self) -> str:
        return os.path.join(self.db_folder, self.db_file)

    @property
    def whisper_language(self) -> str | None:
        """faster-whisper expects None for automatic detection."""
        return None if self.language == "auto" else self.language


def load_config() -> Config:
    """Reads the .env file and builds a validated Config object."""
    load_dotenv()

    try:
        beam_size = int(os.getenv("BEAM_SIZE", "10"))
    except ValueError:
        beam_size = 10

    device = os.getenv("DEVICE", "cuda").strip().lower()
    compute_type = os.getenv("COMPUTE_TYPE", "float16").strip().lower()

    # float16 is not supported on CPU -> fall back to int8 instead of crashing
    if device == "cpu" and compute_type == "float16":
        print("Warning: float16 is not supported on CPU - falling back to int8.")
        compute_type = "int8"

    def _opt(name: str) -> str | None:
        value = os.getenv(name, "").strip()
        return value or None

    def _float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except ValueError:
            return default

    return Config(
        db_file=os.getenv("DB_FILE", "transcriptions.db"),
        model_size=os.getenv("MODEL_SIZE", "large-v3"),
        device=device,
        compute_type=compute_type,
        language=os.getenv("LANGUAGE", "auto").strip().lower(),
        beam_size=beam_size,
        initial_prompt=_opt("INITIAL_PROMPT"),
        hotwords=_opt("HOTWORDS"),
        no_speech_threshold=_float("NO_SPEECH_THRESHOLD", 0.6),
        logprob_threshold=_float("LOGPROB_THRESHOLD", -1.0),
    )


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def parse_filename(filename: str) -> dict:
    """
    Parses an audio filename to extract structured metadata.

    Expected format:
        DATE_TIME_<ChannelName>_TO_<Receiver>_FROM_<Speaker>

    Example:
        20260317_183128XXXX_DMR_Digital_01__02__TO_1_FROM_67469
    """
    name = os.path.splitext(filename)[0]

    data = {
        "raw": name,
        "date": None,
        "time": None,
        "datetime": None,
        "channel": None,
        "channel_receiver": None,
        "speaker": None,
    }

    # Date + Time (YYYYMMDD_HHMMSS at the start)
    match_datetime = re.match(r"(\d{8})_(\d{6})", name)
    if match_datetime:
        raw_date, raw_time = match_datetime.group(1), match_datetime.group(2)
        try:
            dt = datetime.strptime(raw_date + raw_time, "%Y%m%d%H%M%S")
            data["date"] = dt.strftime("%Y-%m-%d")
            data["time"] = dt.strftime("%H:%M:%S")
            data["datetime"] = dt.isoformat()
        except ValueError:
            print(f"Error while parsing date/time in {filename}")

    # Speaker (FROM_XXXX)
    match_speaker = re.search(r"FROM_(\d+)", name)
    if match_speaker:
        data["speaker"] = match_speaker.group(1)

    # Channel info: strip the leading date/time and the trailing FROM_ part
    channel_info = re.sub(r"^\d{8}_\d{6}", "", name)         # remove date/time
    channel_info = re.sub(r"_?FROM_\d+$", "", channel_info)  # remove speaker
    channel_info = channel_info.strip("_")

    # Receiver channel (TO_X)
    match_receiver = re.search(r"TO_(\d+)", channel_info)
    if match_receiver:
        data["channel_receiver"] = match_receiver.group(1)

    # Channel name: drop TO_X and normalise separators to spaces
    channel = re.sub(r"_?TO_\d+", "", channel_info)
    channel = channel.replace("__", " ").replace("_", " ")
    channel = re.sub(r"\s+", " ", channel).strip()
    data["channel"] = channel

    return data


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

@dataclass
class TranscriptionResult:
    text: str
    status: str            # "ok" | "low_confidence" | "no_speech"
    avg_logprob: float | None
    language: str | None


def _looks_like_hallucination(text: str) -> bool:
    """True if the whole output is just a known silence-hallucination phrase."""
    cleaned = text.lower().strip(" .!?,")
    if cleaned in HALLUCINATION_PHRASES:
        return True
    return any(cleaned.startswith(phrase) for phrase in ("untertitel", "amara.org"))


def transcribe_audio(model: WhisperModel, filepath: str, config: Config) -> TranscriptionResult | None:
    """
    Transcribes a single audio file and classifies the result.
    Returns None only on a real error (so the file can be retried later).
    """
    try:
        segments_gen, info = model.transcribe(
            filepath,
            language=config.whisper_language,
            beam_size=config.beam_size,
            vad_filter=True,                       # drop silence / noise up front
            initial_prompt=config.initial_prompt,
            hotwords=config.hotwords,
        )
        segments = list(segments_gen)
    except Exception as e:
        print(f"Error while transcribing {filepath}: {e}")
        return None

    text = " ".join(seg.text.strip() for seg in segments).strip()
    avg_logprob = (
        sum(seg.avg_logprob for seg in segments) / len(segments) if segments else None
    )

    # --- classify ---------------------------------------------------------
    if not text:
        status = "no_speech"
    elif segments and all(seg.no_speech_prob > config.no_speech_threshold for seg in segments):
        status = "no_speech"
    elif _looks_like_hallucination(text):
        status = "no_speech"
    elif avg_logprob is not None and avg_logprob < config.logprob_threshold:
        status = "low_confidence"
    else:
        status = "ok"

    return TranscriptionResult(
        text=text,
        status=status,
        avg_logprob=avg_logprob,
        language=info.language,
    )


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def build_database(config: Config) -> None:
    """Creates the SQLite table if it does not exist yet."""
    print(f"Initializing database at {config.db_path}")
    conn = sqlite3.connect(config.db_path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transcriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                date TEXT,
                time TEXT,
                speaker TEXT,
                channel TEXT,
                text TEXT,
                status TEXT,
                avg_logprob REAL,
                language TEXT,
                created_at TEXT
            )
        """)
        conn.commit()
    finally:
        conn.close()


def insert_transcription(conn, filename: str, file_info: dict, result: TranscriptionResult) -> None:
    """Inserts one transcription row."""
    conn.execute(
        """
        INSERT INTO transcriptions
            (filename, date, time, speaker, channel, text, status, avg_logprob, language, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            filename,
            file_info["date"],
            file_info["time"],
            file_info["speaker"],
            file_info["channel"],
            result.text,
            result.status,
            result.avg_logprob,
            result.language,
            datetime.now().isoformat(timespec="seconds"),
        ),
    )


# ---------------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------------

def print_file_info(file_info: dict, result: TranscriptionResult) -> None:
    print("\n==============================")
    print(f"Filename: {file_info['raw']}")
    print(f"Date:     {file_info['date']}")
    print(f"Time:     {file_info['time']}")
    print(f"Speaker:  {file_info['speaker']}")
    print(f"Channel:  {file_info['channel']}")
    print(f"Status:   {result.status} (avg_logprob={result.avg_logprob})")
    print("Transcription:")
    print(result.text if result.text else "<nothing understood>")


def process_audio_file(filepath: str, model: WhisperModel, config: Config) -> None:
    """Parses, transcribes, stores and (on success) moves a single audio file."""
    filename = os.path.basename(filepath)
    file_info = parse_filename(filename)

    result = transcribe_audio(model, filepath, config)
    if result is None:
        print(f"Skipping {filename} (transcription error, file kept in input).")
        return

    conn = sqlite3.connect(config.db_path)
    try:
        insert_transcription(conn, filename, file_info, result)
        conn.commit()
    finally:
        conn.close()

    print_file_info(file_info, result)

    # A "no_speech"/"low_confidence" result is still a valid outcome -> move the
    # file out of input. Only real errors (result is None) keep it for a retry.
    if config.output_folder:
        os.makedirs(config.output_folder, exist_ok=True)
        shutil.move(filepath, os.path.join(config.output_folder, filename))


def process_existing_files(config: Config, model: WhisperModel) -> None:
    """Processes every audio file already present in the input folder."""
    for filename in sorted(os.listdir(config.input_folder)):
        if not filename.lower().endswith(AUDIO_EXTENSIONS):
            continue
        filepath = os.path.join(config.input_folder, filename)
        print(f"Processing existing file: {filepath}")
        process_audio_file(filepath, model, config)


def wait_for_file_ready(filepath: str, timeout: int = 60, interval: float = 1.0) -> bool:
    """
    Waits until a file stops growing (i.e. the copy/recording has finished).
    Returns True once the size is stable, or after the timeout.
    """
    previous_size = -1
    waited = 0.0
    while waited < timeout:
        try:
            size = os.path.getsize(filepath)
        except OSError:
            return False
        if size > 0 and size == previous_size:
            return True
        previous_size = size
        time.sleep(interval)
        waited += interval
    return True


class AudioHandler(FileSystemEventHandler):
    """Watchdog handler that processes newly created audio files."""

    def __init__(self, model: WhisperModel, config: Config):
        self.model = model
        self.config = config

    def on_created(self, event):
        if event.is_directory:
            return
        if not event.src_path.lower().endswith(AUDIO_EXTENSIONS):
            return

        print(f"New audio file detected: {event.src_path}")
        if not wait_for_file_ready(event.src_path):
            print(f"File vanished before it was ready: {event.src_path}")
            return
        process_audio_file(event.src_path, self.model, self.config)


def start_watcher(config: Config, model: WhisperModel) -> None:
    """Processes existing files, then watches the input folder for new ones."""
    process_existing_files(config, model)

    handler = AudioHandler(model, config)
    observer = Observer()
    observer.schedule(handler, path=config.input_folder, recursive=False)
    observer.start()

    print(f"Watching folder '{config.input_folder}' for new audio files...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    config = load_config()

    for folder in (config.input_folder, config.output_folder,
                   config.db_folder, config.model_folder):
        Path(folder).mkdir(parents=True, exist_ok=True)

    print("Configuration:")
    print(f"  DB_FILE       = {config.db_file}")
    print(f"  MODEL_SIZE    = {config.model_size}")
    print(f"  DEVICE        = {config.device}")
    print(f"  COMPUTE_TYPE  = {config.compute_type}")
    print(f"  LANGUAGE      = {config.language}")
    print(f"  BEAM_SIZE     = {config.beam_size}")

    build_database(config)

    print("Loading faster-whisper model...")
    model = WhisperModel(
        config.model_size,
        device=config.device,
        compute_type=config.compute_type,
        download_root=config.model_folder,
    )

    start_watcher(config, model)


if __name__ == "__main__":
    main()