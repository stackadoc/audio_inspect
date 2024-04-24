import os
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from pydub.utils import mediainfo
from diskcache import Cache
import pyloudnorm as pyln


def compute_bpm(fpath):
    seg = AudioSegment.from_file(fpath)

    # reduce loudness of sounds over 120Hz (focus on bass drum, etc)
    seg = seg.low_pass_filter(120.0)

    # we'll call a beat: anything above average loudness
    beat_loudness = seg.dBFS

    # the fastest tempo we'll allow is 240 bpm (60000ms / 240beats)
    minimum_silence = int(60000 / 240.0)

    nonsilent_times = detect_nonsilent(seg, minimum_silence, beat_loudness)

    # Handling "empty audio" files detection
    if not nonsilent_times or not nonsilent_times[0]:
        return 0

    spaces_between_beats = []
    last_t = nonsilent_times[0][0]

    for peak_start, _ in nonsilent_times[1:]:
        spaces_between_beats.append(peak_start - last_t)
        last_t = peak_start

    # We'll base our guess on the median space between beats
    spaces_between_beats = sorted(spaces_between_beats)
    if not spaces_between_beats:
        return 0

    # Finally compute bpm
    space = spaces_between_beats[int(len(spaces_between_beats) / 2)]
    bpm = 60000 / space
    return bpm


def get_audio_properties(file_path, cache_folder:Path=None, get_bpm=False, skip_cache=False):
    """
    :param file_path:
    :param get_bpm: Set to default to False, because it is computationally slow
    :return:
    """

    cache = Cache((cache_folder or Path('/tmp')).as_posix())
    file_key = f"audio_properties-{os.path.basename(file_path)}"

    # If file in cache
    if file_key in cache and not skip_cache:
        # If the embedding for this file is cached, retrieve it
        return cache[file_key]

    # Load the audio file
    y, sr = librosa.load(file_path, sr=None)  # Load with its original sampling rate

    # Getting the duration in seconds
    duration_secs = int(librosa.get_duration(y=y, sr=sr))

    # Get BPM (tempo)
    # tempo, _ = librosa.beat.beat_track(y=y, sr=sr) if get_bpm else None, None
    tempo = compute_bpm(file_path) if get_bpm else 0

    # Get media info for additional properties
    info = mediainfo(file_path)

    # Loudness -- be careful when using this.
    data, rate = sf.read(file_path)  # load audio (with shape (samples, channels))
    meter = pyln.Meter(rate)  # create BS.1770 meter
    loudness = meter.integrated_loudness(data)  # measure loudness
    ref = 1
    loudness = 20 * np.log10(abs(loudness) / ref) if not np.isinf(loudness) else 0

    # Get other properties
    bitrate_bps = int(info['bit_rate'])
    nb_channels = int(info['channels'])
    try:
        bit_depth = int(sf.info(file_path).subtype.split('_')[1])
    except ValueError:
        bit_depth = None  # This is the case when the file is wav
    rates_hz = sr

    # Caching and return
    audio_properties = {
        "bpm": int(tempo),
        "sample_rate_hz": rates_hz,
        "bitrate_bps": bitrate_bps,
        "nb_channels": nb_channels,
        "bit_depth": bit_depth,
        "loudness": loudness,
        "duration_secs": duration_secs
    }
    cache[file_key] = audio_properties
    return audio_properties


if __name__ == '__main__':
    fpath = '/home/arthur/data/musicdb/library/7fbb0c4de0e4bdf5e1dec8f3e803174b.mp3'
    print(get_audio_properties(fpath, get_bpm=True))

    fpath = '/home/arthur/code/musicsearch/audios/empty_audio_5sec_no_sound.wav'
    print(get_audio_properties(fpath, get_bpm=True))

    fpath = '/home/arthur/code/musicsearch/audios/quasi_empty_sound.wav'
    print(get_audio_properties(fpath, get_bpm=True))

    fpath = '/home/arthur/data/musicdb/split_demucses/474499222_456577880 --__-- Polo & Pan - Ani Kuni_mp3_drums/chunk_1.wav'
    print(get_audio_properties(fpath, get_bpm=False, skip_cache=True))

    fpath = '/home/arthur/data/music/originals/372554152_456240890 --__-- Paul Kalkbrenner - No Goodbye.mp3'
    print(get_audio_properties(fpath, get_bpm=True))

    fpath = '/home/arthur/data/mididb/SM101 - MIDI Elements 80s Beats (WAV-MIDI)/80b_kit_707raw/drum patches/707raw_kit Samples/80b_kit_707raw_perc2.wav'
    print(get_audio_properties(fpath, get_bpm=False, skip_cache=True))

    fpath = '/home/arthur/data/music/training_db/Paul Kalkbrenner - Revolte - intro - 1.wav'
    print(get_audio_properties(fpath, get_bpm=False, skip_cache=True))

    fpath = '/home/arthur/data/music/training_db/Paul Kalkbrenner - Revolte -  - 34.wav'
    print(get_audio_properties(fpath, get_bpm=False, skip_cache=True))
