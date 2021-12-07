import torch
from transformers import AutoTokenizer
from pathlib import Path
import json
from nemo_text_processing.text_normalization.normalize import Normalizer
from nltk.tokenize import sent_tokenize, word_tokenize
from g2p_en import G2p
import soundfile as sf
import numpy as np
import gdown

def download_torchscript():
    out_dir = Path(".") / "torchscript"
    url = 'https://drive.google.com/uc?id=0B9P1L--7Wd2vNm9zMTJWOGxobkU'
    print("Downloading acoustic model ...")
    gdown.download("https://drive.google.com/file/d/1-145-USEQpM3pTU_Hzle7fmESu-xztEw/view?usp=sharing", out_dir / "acoustic_model.pt", quiet=False)
    print("Downloading style predictor ...")
    gdown.download("https://drive.google.com/file/d/1Vmm76LkPYxhG6Erjh5UV_FtaOesW54l0/view?usp=sharing", out_dir / "style_predictor.pt", quiet=False)
    print("Downloading vocoder ...")
    gdown.download("https://drive.google.com/file/d/1hNDVeCdumQsBIFCW48LjmjJxyF4YZHCy/view?usp=sharing", out_dir / "vocoder.pt", quiet=False)


class DeepPoniesTTS():
    def __init__(self):
        self.g2p = G2p()
        self.acoustic_model = torch.jit.load(Path(".") / "torchscript" / "acoustic_model.pt")
        self.style_predictor = torch.jit.load(Path(".") / "torchscript" / "style_predictor.pt")        
        self.vocoder = torch.jit.load(Path(".") / "torchscript" / "vocoder.pt")
        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        self.normalizer = Normalizer(input_case='cased', lang='en')
        self.speaker2id = self.get_speaker2id()
        self.symbol2id = self.get_symbol2id()
        self.acoustic_model.eval()
        self.style_predictor.eval()
        self.vocoder.eval()

    def get_speaker2id(self):
        speaker2id = {}
        with open(Path(".") / "assets" / "speakerCategories.json", "r") as json_file:
            data = json.load(json_file)
        for category in data.keys():
            for item in data[category]["items"]:
                if not item["activated"]:
                    continue
                speaker2id[item["speaker"]] = item["speaker_id"]
        return speaker2id

    def get_symbol2id(self):
        with open(Path(".") / "assets" / "symbol2id.json", "r") as json_file:
            symbol2id = json.load(json_file)
        return symbol2id

    def synthesize(self, text: str, speaker_name: str, duration_control: float=1.0) -> np.ndarray:
        waves = []
        text = self.normalizer.normalize(text, verbose=False)
        text = text.strip()
        speaker_ids = torch.LongTensor([self.speaker2id[speaker_name]]) 
        if text[-1] not in [".", "?", "!"]:
            text = text + "."
        for sentence in sent_tokenize(text):
            encoding = self.tokenizer(
                sentence,
                add_special_tokens=True,
                padding=True, 
                return_tensors="pt"
            )
            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]
            phone_ids = []
            for word in word_tokenize(sentence):
                if word in [".", "?", "!"]:
                    phone_ids.append(self.symbol2id[word])
                elif word in [",", ";"]:
                    phone_ids.append(self.symbol2id["@SILENCE"])
                else:
                    for phone in self.g2p(word):
                        phone_ids.append(self.symbol2id["@" + phone])
            phone_ids = torch.LongTensor([phone_ids])
            with torch.no_grad():
                style = self.style_predictor(input_ids, attention_mask)
                mels = self.acoustic_model(
                    phone_ids,
                    speaker_ids,
                    style,
                    1.0,
                    duration_control
                )[0]
                wave = self.vocoder(mels, speaker_ids)
                waves.append(wave.view(-1))
        full_wave = torch.cat(waves, dim=0).cpu().numpy()
        return full_wave
        
if __name__ == "__main__":
    tts = DeepPoniesTTS()
    wave = tts.synthesize("Show me the way!", "Twilight Sparkle")
    sf.write("out.wav", wave, 44100)