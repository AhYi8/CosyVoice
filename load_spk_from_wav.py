import torch, os, torchaudio, argparse, random, librosa
from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.utils.common import set_all_random_seed
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.speaker_utils import save_speaker_features
from pathlib import Path


max_val = 0.8

def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }

def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    """对生成的语音进行后处理
    
    Args:
        speech: 输入的语音张量
        top_db: 音频修剪的分贝阈值,默认60
        hop_length: 帧移,默认220
        win_length: 窗长,默认440
        
    Returns:
        处理后的语音张量
    """
    # 对语音进行静音修剪
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    # 音量归一化
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    # 在语音末尾添加0.2秒静音
    speech = torch.concat([speech, torch.zeros(1, int(cosyvoice.sample_rate * 0.2))], dim=1)
    return speech


def load_spk_from_wav(spk_id, wav_file, prompt_text, text_frontend=True):
    prompt_speech_16k = postprocess(load_wav(wav_file, prompt_sr))
    # 设置随机种子
    set_all_random_seed(random.randint(1, 100000000))
    # 对参考文本进行标准化处理
    prompt_text = cosyvoice.frontend.text_normalize(prompt_text, split=False, text_frontend=text_frontend)
    model_input = cosyvoice.frontend.frontend_zero_shot("这是一段测试文本。", prompt_text, prompt_speech_16k, cosyvoice.sample_rate)

    save_speaker_features(
                features=model_input,
                speaker_id=spk_id,
                save_dir=Path("./speakers")
            )

if __name__ == "__main__":
    prompt_sr = 16000
    parser = argparse.ArgumentParser()
    parser.add_argument('--promt_wav',
                        type=str,
                        help='Requires advance timbre characteristics recording file')
    parser.add_argument('--promt_text',
                        type=str,
                        help='Recording Files For Prompt Text')
    parser.add_argument('--spk_id',
                        type=str,
                        help='Saved speaker_id')
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    try:
        cosyvoice = CosyVoice(args.model_dir)
    except Exception:
        try:
            cosyvoice = CosyVoice2(args.model_dir)
        except Exception:
            raise TypeError('no valid model_type!')
    load_spk_from_wav(args.spk_id, args.promt_wav, args.promt_text)