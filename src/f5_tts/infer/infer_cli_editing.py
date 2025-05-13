import argparse
import codecs
import os
import re
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from glob import glob
from tqdm import tqdm
import json
import numpy as np
import soundfile as sf
import tomli

# from cached_path import cached_path
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download

from f5_tts.infer.utils_infer import (
    mel_spec_type,
    target_rms,
    cross_fade_duration,
    nfe_step,
    cfg_strength,
    sway_sampling_coef,
    speed,
    fix_duration,
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)
from f5_tts.model import DiT, UNetT
import torch
import torchaudio
import torch.nn.functional as F
from f5_tts.model.utils import convert_char_to_pinyin
from f5_tts.infer.utils_infer import save_spectrogram

parser = argparse.ArgumentParser(
    prog="python3 infer-cli.py",
    description="Commandline interface for E2/F5 TTS with Advanced Batch Processing.",
    epilog="Specify options above to override one or more settings from config.",
)
parser.add_argument(
    "-c",
    "--config",
    type=str,
    # default=os.path.join(
    #     files("f5_tts").joinpath("infer/examples/basic"), "basic.toml"
    # ),
    default="/storage/zhangxueyao/workspace/F5-TTS/src/f5_tts/infer/examples/basic/basic.toml",
    help="The configuration file, default see infer/examples/basic/basic.toml",
)


# Note. Not to provide default value here in order to read default from config file

parser.add_argument(
    "-m",
    "--model",
    type=str,
    help="The model name: F5-TTS | E2-TTS",
)
parser.add_argument(
    "-mc",
    "--model_cfg",
    type=str,
    help="The path to F5-TTS model config file .yaml",
)
parser.add_argument(
    "-p",
    "--ckpt_file",
    type=str,
    help="The path to model checkpoint .pt, leave blank to use default",
)
parser.add_argument(
    "-v",
    "--vocab_file",
    type=str,
    help="The path to vocab file .txt, leave blank to use default",
)
parser.add_argument(
    "-r",
    "--ref_audio",
    type=str,
    help="The reference audio file.",
    default="/storage/zhangxueyao/workspace/F5-TTS/00000309-00000300.wav",  # Fake ref audio, just a placeholder
)
parser.add_argument(
    "-s",
    "--ref_text",
    type=str,
    help="The transcript/subtitle for the reference audio",
    default="fake_ref_text",  # Fake ref text, just a placeholder
)
parser.add_argument(
    "-t",
    "--gen_text",
    type=str,
    help="The text to make model synthesize a speech",
    default="fake_gen_text",  # Fake gen text, just a placeholder
)
parser.add_argument(
    "-f",
    "--gen_file",
    type=str,
    help="The file with text to generate, will ignore --gen_text",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="The path to output folder",
)
parser.add_argument(
    "-w",
    "--output_file",
    type=str,
    help="The name of output file",
)
parser.add_argument(
    "--save_chunk",
    action="store_true",
    help="To save each audio chunks during inference",
)
parser.add_argument(
    "--remove_silence",
    action="store_true",
    help="To remove long silence found in ouput",
)
parser.add_argument(
    "--load_vocoder_from_local",
    action="store_true",
    help="To load vocoder from local dir, default to ../checkpoints/vocos-mel-24khz",
)
parser.add_argument(
    "--vocoder_name",
    type=str,
    choices=["vocos", "bigvgan"],
    help=f"Used vocoder name: vocos | bigvgan, default {mel_spec_type}",
)
parser.add_argument(
    "--target_rms",
    type=float,
    help=f"Target output speech loudness normalization value, default {target_rms}",
)
parser.add_argument(
    "--cross_fade_duration",
    type=float,
    help=f"Duration of cross-fade between audio segments in seconds, default {cross_fade_duration}",
)
parser.add_argument(
    "--nfe_step",
    type=int,
    help=f"The number of function evaluation (denoising steps), default {nfe_step}",
)
parser.add_argument(
    "--cfg_strength",
    type=float,
    help=f"Classifier-free guidance strength, default {cfg_strength}",
)
parser.add_argument(
    "--sway_sampling_coef",
    type=float,
    help=f"Sway Sampling coefficient, default {sway_sampling_coef}",
)
parser.add_argument(
    "--speed",
    type=float,
    help=f"The speed of the generated audio, default {speed}",
)
parser.add_argument(
    "--fix_duration",
    type=float,
    help=f"Fix the total duration (ref and gen audios) in seconds, default {fix_duration}",
)
parser.add_argument("--save_root", type=str, required=True)
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--evalset_root", type=str, required=True)
parser.add_argument("--eval_setting", type=str, required=True)
parser.add_argument("--task_name", type=str, required=True)

args = parser.parse_args()


# config file

config = tomli.load(open(args.config, "rb"))


# command-line interface parameters

model = args.model or config.get("model", "F5-TTS")
model_cfg = args.model_cfg or config.get(
    "model_cfg",
    "/storage/zhangxueyao/workspace/F5-TTS/src/f5_tts/configs/F5TTS_Base_train.yaml",
)
ckpt_file = args.ckpt_file or config.get("ckpt_file", "")
vocab_file = args.vocab_file or config.get("vocab_file", "")

ref_audio = args.ref_audio or config.get(
    "ref_audio", "infer/examples/basic/basic_ref_en.wav"
)
ref_text = (
    args.ref_text
    if args.ref_text is not None
    else config.get("ref_text", "Some call me nature, others call me mother nature.")
)
gen_text = args.gen_text or config.get(
    "gen_text", "Here we generate something just for test."
)
gen_file = args.gen_file or config.get("gen_file", "")

output_dir = args.output_dir or config.get("output_dir", "tests")
output_file = args.output_file or config.get(
    "output_file", f"infer_cli_{datetime.now().strftime(r'%Y%m%d_%H%M%S')}.wav"
)

save_chunk = args.save_chunk or config.get("save_chunk", False)
remove_silence = args.remove_silence or config.get("remove_silence", False)
load_vocoder_from_local = args.load_vocoder_from_local or config.get(
    "load_vocoder_from_local", False
)

vocoder_name = args.vocoder_name or config.get("vocoder_name", mel_spec_type)
target_rms = args.target_rms or config.get("target_rms", target_rms)
cross_fade_duration = args.cross_fade_duration or config.get(
    "cross_fade_duration", cross_fade_duration
)
nfe_step = args.nfe_step or config.get("nfe_step", nfe_step)
cfg_strength = args.cfg_strength or config.get("cfg_strength", cfg_strength)
sway_sampling_coef = args.sway_sampling_coef or config.get(
    "sway_sampling_coef", sway_sampling_coef
)
speed = args.speed or config.get("speed", speed)
fix_duration = args.fix_duration or config.get("fix_duration", fix_duration)


# patches for pip pkg user
if "infer/examples/" in ref_audio:
    ref_audio = str(files("f5_tts").joinpath(f"{ref_audio}"))
if "infer/examples/" in gen_file:
    gen_file = str(files("f5_tts").joinpath(f"{gen_file}"))
if "voices" in config:
    for voice in config["voices"]:
        voice_ref_audio = config["voices"][voice]["ref_audio"]
        if "infer/examples/" in voice_ref_audio:
            config["voices"][voice]["ref_audio"] = str(
                files("f5_tts").joinpath(f"{voice_ref_audio}")
            )


# ignore gen_text if gen_file provided

if gen_file:
    gen_text = codecs.open(gen_file, "r", "utf-8").read()


# output path

wave_path = Path(output_dir) / output_file
# spectrogram_path = Path(output_dir) / "infer_cli_out.png"
if save_chunk:
    output_chunk_dir = os.path.join(output_dir, f"{Path(output_file).stem}_chunks")
    if not os.path.exists(output_chunk_dir):
        os.makedirs(output_chunk_dir)


# load vocoder

if vocoder_name == "vocos":
    vocoder_local_path = "../checkpoints/vocos-mel-24khz"
elif vocoder_name == "bigvgan":
    vocoder_local_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"

vocoder = load_vocoder(
    vocoder_name=vocoder_name,
    is_local=load_vocoder_from_local,
    local_path=vocoder_local_path,
)


# load TTS model

if model == "F5-TTS":
    model_cls = DiT
    model_cfg = OmegaConf.load(model_cfg).model.arch
    if not ckpt_file:  # path not specified, download from repo
        if vocoder_name == "vocos":
            repo_name = "F5-TTS"
            exp_name = "F5TTS_Base"
            ckpt_step = 1200000
            ckpt_file = hf_hub_download(
                repo_id=f"SWivid/{repo_name}",
                filename=f"{exp_name}/model_{ckpt_step}.safetensors",
            )
            # ckpt_file = f"ckpts/{exp_name}/model_{ckpt_step}.pt"  # .pt | .safetensors; local path
        elif vocoder_name == "bigvgan":
            repo_name = "F5-TTS"
            exp_name = "F5TTS_Base_bigvgan"
            ckpt_step = 1250000
            ckpt_file = str(
                cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.pt")
            )

elif model == "E2-TTS":
    assert args.model_cfg is None, "E2-TTS does not support custom model_cfg yet"
    assert vocoder_name == "vocos", "E2-TTS only supports vocoder vocos yet"
    model_cls = UNetT
    model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
    if not ckpt_file:  # path not specified, download from repo
        repo_name = "E2-TTS"
        exp_name = "E2TTS_Base"
        ckpt_step = 1200000
        ckpt_file = str(
            cached_path(
                f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"
            )
        )
        # ckpt_file = f"ckpts/{exp_name}/model_{ckpt_step}.pt"  # .pt | .safetensors; local path

print(f"Using {model}...")
ema_model = load_model(
    model_cls, model_cfg, ckpt_file, mel_spec_type=vocoder_name, vocab_file=vocab_file
)


# inference process


def main():
    main_voice = {"ref_audio": ref_audio, "ref_text": ref_text}
    if "voices" not in config:
        voices = {"main": main_voice}
    else:
        voices = config["voices"]
        voices["main"] = main_voice
    for voice in voices:
        print("Voice:", voice)
        print("ref_audio ", voices[voice]["ref_audio"])
        voices[voice]["ref_audio"], voices[voice]["ref_text"] = (
            preprocess_ref_audio_text(
                voices[voice]["ref_audio"], voices[voice]["ref_text"]
            )
        )
        print("ref_audio_", voices[voice]["ref_audio"], "\n\n")

    generated_audio_segments = []
    reg1 = r"(?=\[\w+\])"
    chunks = re.split(reg1, gen_text)
    reg2 = r"\[(\w+)\]"
    for text in chunks:
        if not text.strip():
            continue
        match = re.match(reg2, text)
        if match:
            voice = match[1]
        else:
            print("No voice tag found, using main.")
            voice = "main"
        if voice not in voices:
            print(f"Voice {voice} not found, using main.")
            voice = "main"
        text = re.sub(reg2, "", text)
        ref_audio_ = voices[voice]["ref_audio"]
        ref_text_ = voices[voice]["ref_text"]
        gen_text_ = text.strip()
        print(f"Voice: {voice}")
        audio_segment, final_sample_rate, spectragram = infer_process(
            ref_audio_,
            ref_text_,
            gen_text_,
            ema_model,
            vocoder,
            mel_spec_type=vocoder_name,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
        )
        generated_audio_segments.append(audio_segment)

        if save_chunk:
            if len(gen_text_) > 200:
                gen_text_ = gen_text_[:200] + " ... "
            sf.write(
                os.path.join(
                    output_chunk_dir,
                    f"{len(generated_audio_segments)-1}_{gen_text_}.wav",
                ),
                audio_segment,
                final_sample_rate,
            )

    if generated_audio_segments:
        final_wave = np.concatenate(generated_audio_segments)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(wave_path, "wb") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            # Remove silence
            if remove_silence:
                remove_silence_for_generated_wav(f.name)
            print(f.name)


def infer_one_sample(
    ref_audio,
    ref_text,
    gen_text,
    output_path,
    ref_audio_duration,
    gen_audio_duration=None,
):
    main_voice = {"ref_audio": ref_audio, "ref_text": ref_text}
    if "voices" not in config:
        voices = {"main": main_voice}
    else:
        voices = config["voices"]
        voices["main"] = main_voice
    for voice in voices:
        print("Voice:", voice)
        print("ref_audio ", voices[voice]["ref_audio"])
        voices[voice]["ref_audio"], voices[voice]["ref_text"] = (
            preprocess_ref_audio_text(
                voices[voice]["ref_audio"], voices[voice]["ref_text"]
            )
        )
        print("ref_audio_", voices[voice]["ref_audio"], "\n\n")

    generated_audio_segments = []
    reg1 = r"(?=\[\w+\])"
    chunks = re.split(reg1, gen_text)
    reg2 = r"\[(\w+)\]"
    for text in chunks:
        if not text.strip():
            continue
        match = re.match(reg2, text)
        if match:
            voice = match[1]
        else:
            print("No voice tag found, using main.")
            voice = "main"
        if voice not in voices:
            print(f"Voice {voice} not found, using main.")
            voice = "main"
        text = re.sub(reg2, "", text)
        ref_audio_ = voices[voice]["ref_audio"]
        ref_text_ = voices[voice]["ref_text"]
        gen_text_ = text.strip()
        print(f"Voice: {voice}")

        if gen_audio_duration is not None:
            target_duration = ref_audio_duration + gen_audio_duration
        else:
            target_duration = fix_duration

        audio_segment, final_sample_rate, spectragram = infer_process(
            ref_audio_,
            ref_text_,
            gen_text_,
            ema_model,
            vocoder,
            mel_spec_type=vocoder_name,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=target_duration,
        )
        generated_audio_segments.append(audio_segment)

        if save_chunk:
            if len(gen_text_) > 200:
                gen_text_ = gen_text_[:200] + " ... "
            sf.write(
                os.path.join(
                    output_chunk_dir,
                    f"{len(generated_audio_segments)-1}_{gen_text_}.wav",
                ),
                audio_segment,
                final_sample_rate,
            )

    if generated_audio_segments:
        final_wave = np.concatenate(generated_audio_segments)

        with open(output_path, "wb") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            # Remove silence
            # if remove_silence:
            #     remove_silence_for_generated_wav(f.name)


def load_evalset(evalset_name):
    evalset_file = os.path.join(evalset_root, evalset_name, "evalset_with_span.json")

    with open(evalset_file, "r") as f:
        evalset = json.load(f)
    return evalset


def run_editing(
    audio_to_edit,
    target_text,
    parts_to_edit,
    fix_duration=None,
    output_path=None,
):
    # audio_to_edit = "/storage/zhangxueyao/workspace/F5-TTS/src/f5_tts/infer/examples/basic/basic_ref_en.wav"
    # origin_text = "Some call me nature, others call me mother nature."
    # target_text = "Some call me optimist, others call me realist."
    # parts_to_edit = [
    #     [1.42, 2.44],
    #     [4.04, 4.9],
    # ]  # stard_ends of "nature" & "mother nature", in seconds

    target_sample_rate = 24000
    hop_length = 256
    tokenizer = "pinyin"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Audio
    audio, sr = torchaudio.load(audio_to_edit)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)
    offset = 0
    audio_ = torch.zeros(1, 0)
    edit_mask = torch.zeros(1, 0, dtype=torch.bool)
    for part in parts_to_edit:
        start, end = part
        part_dur = end - start if fix_duration is None else fix_duration.pop(0)
        part_dur = part_dur * target_sample_rate
        start = start * target_sample_rate
        audio_ = torch.cat(
            (
                audio_,
                audio[:, round(offset) : round(start)],
                torch.zeros(1, round(part_dur)),
            ),
            dim=-1,
        )
        edit_mask = torch.cat(
            (
                edit_mask,
                torch.ones(1, round((start - offset) / hop_length), dtype=torch.bool),
                torch.zeros(1, round(part_dur / hop_length), dtype=torch.bool),
            ),
            dim=-1,
        )
        offset = end * target_sample_rate
    # audio = torch.cat((audio_, audio[:, round(offset):]), dim = -1)
    edit_mask = F.pad(
        edit_mask,
        (0, audio.shape[-1] // hop_length - edit_mask.shape[-1] + 1),
        value=True,
    )
    audio = audio.to(device)
    edit_mask = edit_mask.to(device)

    # Text
    text_list = [target_text]
    if tokenizer == "pinyin":
        final_text_list = convert_char_to_pinyin(text_list)
    else:
        final_text_list = [text_list]
    print(f"text  : {text_list}")
    print(f"pinyin: {final_text_list}")

    # Duration
    ref_audio_len = 0
    duration = audio.shape[-1] // hop_length

    # Inference
    with torch.inference_mode():
        generated, trajectory = ema_model.sample(
            cond=audio,
            text=final_text_list,
            duration=duration,
            steps=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            seed=None,
            edit_mask=edit_mask,
        )
        print(f"Generated mel: {generated.shape}")

        # Final result
        generated = generated.to(torch.float32)
        generated = generated[:, ref_audio_len:, :]
        gen_mel_spec = generated.permute(0, 2, 1)
        if mel_spec_type == "vocos":
            generated_wave = vocoder.decode(gen_mel_spec).cpu()
        elif mel_spec_type == "bigvgan":
            generated_wave = vocoder(gen_mel_spec).squeeze(0).cpu()

        if rms < target_rms:
            generated_wave = generated_wave * rms / target_rms

        # output_dir = (
        #     "/storage/zhangxueyao/workspace/F5-TTS/src/f5_tts/infer/examples/basic"
        # )
        # save_spectrogram(
        #     gen_mel_spec[0].cpu().numpy(), f"{output_dir}/speech_edit_out.png"
        # )
        # torchaudio.save(
        #     f"{output_dir}/speech_edit_out.wav", generated_wave, target_sample_rate
        # )
        # print(f"Generated wav: {generated_wave.shape}")

        torchaudio.save(output_path, generated_wave, target_sample_rate)


if __name__ == "__main__":
    evalset_root = args.evalset_root
    evalset_name = args.eval_setting
    evalset = load_evalset(evalset_name)

    print("\nFor {}...".format(evalset_name))
    save_dir = os.path.join(args.save_root, args.task_name, evalset_name)
    os.makedirs(save_dir, exist_ok=True)

    for item in tqdm(evalset):
        output_filename = "{}-{}-{}".format(
            args.model_name, evalset_name, item["output_path"]
        )
        output_path = os.path.join(save_dir, output_filename)
        if os.path.exists(output_path):
            continue

        raw_text = item["prompt"]["text"]
        raw_audio = os.path.join(
            evalset_root,
            evalset_name,
            "wav",
            "{}.wav".format(item["prompt"]["uid"]),
        )
        target_text = item["input"]["text"]
        parts_to_edit = item["morphed_span"]

        run_editing(
            audio_to_edit=raw_audio,
            target_text=target_text,
            parts_to_edit=parts_to_edit,
            output_path=output_path,
        )

    ### Eval ###
    eval_log_dir = os.path.join(args.save_root, "log_eval")
    os.makedirs(eval_log_dir, exist_ok=True)

    eval_log_file = os.path.join(
        eval_log_dir, f"{args.task_name}_{args.eval_setting}.log"
    )

    os.system(
        f"python /storage/zhangxueyao/workspace/SpeechGenerationYC/models/svc/llm/evaluation/eval_wer_sim_fpc.py \
    --save_root {args.save_root} \
    --model_name {args.model_name} \
    --evalset_root {args.evalset_root} \
    --eval_setting {args.eval_setting} \
    --task_name {args.task_name} > {eval_log_file} 2>&1 &"
    )
