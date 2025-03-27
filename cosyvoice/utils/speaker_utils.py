import torch, os, logging, datetime
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

def load_spk_info_from_pt(spk_pt_filename, spk_dir = "./speakers"):
    """从PT文件加载说话人特征
    
    Args:
        spk_id: 要加载的说话人ID
        spk_dir: 保存说话人特征的目录路径，默认为"./speakers"
        
    Returns:
        Dict[str, Any]: 成功时返回加载的说话人特征字典
        None: 文件不存在时返回None
    """
    spk_pt = os.path.join(spk_dir, spk_pt_filename)

    if os.path.exists(spk_pt) and os.path.isfile(spk_pt):
        return torch.load(spk_pt)

    return None


def save_speaker_features(
    features: Dict[str, Any],
    speaker_id: str,
    save_dir: Path = Path("speakers"),
    overwrite: bool = False
) -> Path:
    """安全保存说话人特征到PT文件
    
    Args:
        features: 包含说话人特征的字典
        speaker_id: 字母数字组成的唯一标识符
        save_dir: 保存目录路径 (默认当前目录/speakers)
        overwrite: True：覆盖已存在文件，False：追加文件
    
    Returns:
        生成的完整文件路径
    
    Raises:
        ValueError: speaker_id包含非法字符
        FileExistsError: 文件已存在且不允许覆盖
    """
    # 验证speaker_id合法性
    if not speaker_id.isalnum():
        raise ValueError(f"Invalid speaker_id: {speaker_id} (只允许字母和数字)")
    
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(save_dir, "spk2info.pt")
    try:
        if not overwrite:
            spk_info_dict = load_spk_info_from_pt("spk2info.pt", spk_dir=save_dir)
            if speaker_id in spk_info_dict:
                raise Exception("speaker_id 已存在")
            else:
                spk_info_dict[speaker_id] = features
        else:
            spk_info_dict = {speaker_id: features}

        torch.save(spk_info_dict, save_path)
        logger.info(f"成功保存说话人特征到: {save_path}")
    except Exception as e:
        logger.error(f"保存失败: {str(e)}")
        raise
    
    return save_path