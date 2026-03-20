# DMR-Transcriber

Super WIP atm... and so far i only got CPU stuff working in the Docker container.

This thing is using [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) to transcribe DMR Audiologs. 

Intended functionality is, that once the container is started, it watches the input folder and transcribes audio files that appear there.
All completed audio files will be moved to the output folder. And every transcribtion will be documented in an SQLite Database.

It is expected that the audiofiles have the name structure 'YYYYMMDD_HHMMss<Channel ID>_DMR_Digital_01__02__TO_X_FROM_<Speaker ID>.mp3' example '20260317_183128CATS_DMR_Digital_01__02__TO_1_FROM_67420.mp3'.

The SQLite Database consits of an 'transcriptions' table:
example:

|id|filename|date|time|speaker|channel|text|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|1|20260317_183128CATS_DMR_Digital_01__02__TO_1_FROM_67420.mp3|2026-03-17|18\:38\:28|67420|CATS DMR Digital 01 02|Hello, this is a Test!|

# Installing
First of all you need Docker and Docker compose.

```bash
git clone https://github.com/loomiy/DMR-Transcriber
cd DMR-Transcriber
mv example-cpu.env .env
docker compose up -d
```

It should have created folders for input, output, db and cache (For Whisper Model Downloads).

Now you can stop the whole thing with `docker compose down` and see the logs with `docker compose logs`.

# Building
`docker build . -t dmr-transcriber:latest`
