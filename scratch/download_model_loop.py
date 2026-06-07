import time
from speechbrain.inference.speaker import EncoderClassifier

def loop_download():
    attempt = 1
    while True:
        try:
            print(f"Attempt {attempt}: Trying to download/cache SpeechBrain ECAPA-TDNN model from Hugging Face...")
            # This triggers the download of the model weights and saves them locally.
            # Once downloaded, it will immediately succeed on subsequent runs.
            classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb"
            )
            print("\n✅ SUCCESS! The model has been downloaded and cached locally.")
            print("You can now safely cancel this script (Ctrl+C) and run the benchmark again.")
            break
        except Exception as e:
            print(f"❌ Attempt {attempt} failed: {e}")
            print("Retrying in 5 seconds...\n")
            time.sleep(5)
            attempt += 1

if __name__ == "__main__":
    loop_download()
