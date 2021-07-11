from mel2wav import MelVocoder

from pathlib import Path
from tqdm import tqdm
import argparse
import librosa
import torch
import soundfile as sf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text2wav", type=Path, required=False)
    parser.add_argument("--load_path", type=Path, required=True)
    parser.add_argument("--save_path", type=Path, required=True)
    parser.add_argument("--folder", type=Path, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    vocoder = MelVocoder(args.load_path, github=True)

    args.save_path.mkdir(exist_ok=True, parents=True)

    if (args.text2wav is not None and args.text2wav):
        for i, fname in tqdm(enumerate(args.folder.glob("*.pt"))):
            melname = fname.name + ".wav"
            mel = torch.load(fname, map_location=torch.device('cpu'))

            print(mel)
            print(mel.shape)

            recons = vocoder.inverse(mel).squeeze().cpu().numpy()
            sf.write(args.save_path / melname, recons, 22050, 'PCM_24')
    else:
        for i, fname in tqdm(enumerate(args.folder.glob("*.wav"))):
            wavname = fname.name
            wav, sr = librosa.core.load(fname)

            mel = vocoder(torch.from_numpy(wav)[None])
            print(mel)
            print(mel.shape)

            recons = vocoder.inverse(mel).squeeze().cpu().numpy()
            sf.write(args.save_path / wavname, recons, 22050, 'PCM_24')


if __name__ == "__main__":
    main()
