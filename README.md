# DMR-Transcriber

Super WIP atm... and so far i only got CPU stuff working in the Docker container.

This thing is using [WhisperX](https://github.com/m-bain/whisperX) to transcribe DMR Audiologs. 

Intended functionality is, that once the container is started, it watches the input folder and transcribes audio files that appear there.
All completed audio files will be moved to the output folder. And every transcribtion will be documented in an SQLite Database.

It is expected that the audiofiles have the name structure `YYYYMMDD_HHMMss<Channel ID>_DMR_Digital_01__02__TO_X_FROM_<Speaker ID>.mp3` 
Example: `20260317_183128CATS_DMR_Digital_01__02__TO_1_FROM_67420.mp3`.

The SQLite Database consits of an `transcriptions` table:
example:

|id|filename|date|time|speaker|channel|text|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|1|20260317_183128CATS_DMR_Digital_01__02__TO_1_FROM_67420.mp3|2026-03-17|18\:38\:28|67420|CATS DMR Digital 01 02|Hello, this is a Test!|

# Usage
First of all you need Docker and Docker compose.

I don't provide build images, because WhisperX images are a bit to large. 

Build command is in the compose.yaml


## CPU
```bash
git clone https://github.com/loomiy/DMR-Transcriber
cd DMR-Transcriber

# build and run
docker compose -f compose.cpu.yaml up --build

# just build
docker compose -f compose.cpu.yaml build

# run in background
docker compose -f compose.cpu.yaml up -d
```

It should have created folders for input, output, db and cache (For Whisper Model Downloads).

## CUDA
You need an Nvidia GPU and need to install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

After it is installed you can run

``` bash
git clone https://github.com/loomiy/DMR-Transcriber
cd DMR-Transcriber

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# build and run
docker compose -f compose.cuda.yaml up --build

# just build
docker compose -f compose.cuda.yaml build

# run in background
docker compose -f compose.cuda.yaml up -d
```

# Developement

For local testing
```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r rquirements.txt
```