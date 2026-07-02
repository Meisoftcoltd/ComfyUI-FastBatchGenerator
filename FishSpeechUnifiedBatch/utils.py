
import torch
import torchaudio
import os
import re
from pydub import AudioSegment

def split_text_by_segments(text):
    """
    Splits text by newlines and identifies segments that may contain emotional tags.
    Returns a list of strings, where each string is a chunk of text.
    """
    lines = [line.strip()
             for line in text.split('\n') 
             if line.strip()]
    return lines

def extract_audio_metadata_tags(text):
    """
    Searches for tags like [long pause], [emphasis], etc., within a segment.
    Returns (cleaned_text, detected_tags).
    """
    # Detects any content inside square brackets []
    tags = re.findall(r'\[(.*?)\]', text)
    # Removes the tags from the text so the model doesn't 'read' them aloud
    cleaned_text = re.sub(r'\[.*?\]', '', text).strip()
    return cleaned_text, tags

def concatenate_audios_with_silence(audio_tensors, sample_rate, silence_duration=0.3, extra_pauses_seconds=0.0):
    """
    Concatenates audio tensors with standard silence + optional extra pauses.
    """
    if not audio_tensors:
        return None

    # Calculate total silence to add (standard gap + any identified long pauses)
    total_gap = silence_duration + extra_pauses_seconds
    silence_samples = int(sample_rate * total_gap)
    silence = torch.zeros((1, silence_samples), dtype=audio_tensors[0].dtype, device=audio_tensors[0].device)

    combined_parts = []
    for tensor in audio_tensors:
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        
        tensor = tensor.to(audio_tensors[0].device).to(audio_tensors[0].dtype)
        combined_parts.append(tensor)
        combined_parts.append(silence)

    output = torch.cat(combined_parts, dim=1)
    return output

def save_as_mp3(audio_tensor, sample_rate, output_path):
    """Saves tensor as MP3."""
    audio_cpu = audio_tensor.squeeze().cpu()
    temp_wav = output_path.replace(".mp3", "_temp.wav")
    torchaudio.save(temp_wav, audio_cpu.unsqueeze(0), sample_rate)
    
    audio_seg = AudioSegment.from_wav(temp_wav)
    audio_segment_mp3 = audio_seg.export(output_path, format="mp3", bitrate="192k")
    
    if os.path.exists(temp_wav):
        os.remove(temp_wav)
    return output_path
