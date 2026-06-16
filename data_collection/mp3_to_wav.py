import os
from pydub import AudioSegment

def convert_mp3_folder(input_dir, output_dir=None, target_sr=16000):
    if output_dir is None:
        output_dir = input_dir  # overwrite in-place

    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith(".mp3"):
                mp3_path = os.path.join(root, filename)
                rel_path = os.path.relpath(mp3_path, input_dir)
                wav_name = os.path.splitext(rel_path)[0] + ".wav"
                wav_path = os.path.join(output_dir, wav_name)

                os.makedirs(os.path.dirname(wav_path), exist_ok=True)

                try:
                    print(f"🎵 Converting {mp3_path} → {wav_path}")
                    audio = AudioSegment.from_mp3(mp3_path)
                    audio = audio.set_frame_rate(target_sr).set_channels(1)
                    audio.export(wav_path, format="wav")
                except Exception as e:
                    print(f"❌ Failed to convert {mp3_path}: {e}")

    print("✅ Conversion complete.")

# Example usage
if __name__ == "__main__":
    convert_mp3_folder(
        input_dir="/home/ai/AI-DataPool/Datasets/audio/Schweiz/swissdials/other",
        output_dir="/home/ai/AI-DataPool/Datasets/audio/Schweiz/swissdials/other_wav",
        target_sr=16000
    )
