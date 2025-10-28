#!/usr/bin/env python3
"""
Wrapper for using audeering's pretrained wav2vec2 age prediction model.
This is the RECOMMENDED model for age prediction with MAE ~7-10 years.

Based on: https://huggingface.co/audeering/wav2vec2-large-robust-24-ft-age-gender
Paper: https://arxiv.org/abs/2306.16962
"""

import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
import soundfile as sf
import librosa
from pathlib import Path
from typing import Union, Tuple, Optional

import warnings

# Suppress UserWarning and FutureWarning globally
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class ModelHead(nn.Module):
    """Classification head for age/gender prediction."""
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class AgeGenderModel(Wav2Vec2PreTrainedModel):
    """Speech age and gender classifier based on wav2vec2."""
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)
        return hidden_states, logits_age, logits_gender


class AudeeringAgePredictor:
    """
    Easy-to-use wrapper for audeering's age prediction model.
    
    Performance:
        - Age MAE: 7.1 - 10.8 years (depending on dataset)
        - Gender accuracy: >91%
    
    Usage:
        predictor = AudeeringAgePredictor()
        age, gender, confidence = predictor.predict('audio.wav')
        print(f"Predicted age: {age:.1f} years")
        print(f"Gender: {gender} (confidence: {confidence:.2f})")
    """
    
    def __init__(
        self,
        model_name: str = 'audeering/wav2vec2-large-robust-24-ft-age-gender',
        device: Optional[str] = None
    ):
        """
        Initialize the age predictor.
        
        Args:
            model_name: HuggingFace model name. Options:
                - 'audeering/wav2vec2-large-robust-24-ft-age-gender' (best accuracy, slower)
                - 'audeering/wav2vec2-large-robust-6-ft-age-gender' (faster, slightly lower accuracy)
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Loading model {model_name} on {self.device}...")
        
        # Load processor and model
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = AgeGenderModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self.gender_labels = ['female', 'male', 'child'] #['child', 'female', 'male']
        
        print("✓ Model loaded successfully!")
    
    def predict_from_audio(
        self,
        audio: np.ndarray,
        sampling_rate: int,
        return_embeddings: bool = False
    ) -> Union[Tuple[float, str, float], Tuple[float, str, float, np.ndarray]]:
        """
        Predict age and gender from audio array.
        
        Args:
            audio: Audio signal (1D numpy array)
            sampling_rate: Sample rate in Hz (model expects 16000)
            return_embeddings: If True, also return wav2vec2 embeddings
            
        Returns:
            age: Predicted age in years (0-100)
            gender: Predicted gender ('female', 'male', 'child')
            confidence: Confidence for gender prediction (0-1)
            embeddings: (optional) wav2vec2 embeddings if return_embeddings=True
        """
        # Resample if necessary
        if sampling_rate != 16000:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
            except ImportError:
                print("Warning: librosa not installed, cannot resample. Install with: pip install librosa")
                print("Proceeding with original sample rate (may affect accuracy)")
        
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
        
        # Normalize
        audio = audio.astype(np.float32)
        
        # Process through wav2vec2 processor
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs['input_values'].to(self.device)
        
        # Run inference
        with torch.no_grad():
            embeddings, age_logits, gender_probs = self.model(input_values)
        
        # Extract predictions
        age = age_logits.cpu().numpy()[0, 0] * 100  # Convert from [0,1] to [0,100]
        
        gender_probs_np = gender_probs.cpu().numpy()[0]
        gender_idx = np.argmax(gender_probs_np)
        gender = self.gender_labels[gender_idx]
        confidence = gender_probs_np[gender_idx]
        
        if return_embeddings:
            embeddings_np = embeddings.cpu().numpy()[0]
            return age, gender, confidence, embeddings_np
        else:
            return age, gender, confidence
    
    def predict(
        self,
        audio_path: Union[str, Path],
        return_embeddings: bool = False
    ) -> Union[Tuple[float, str, float], Tuple[float, str, float, np.ndarray]]:
        """
        Predict age and gender from audio file.
        
        Args:
            audio_path: Path to audio file (wav, mp3, flac, etc.)
            return_embeddings: If True, also return wav2vec2 embeddings
            
        Returns:
            age: Predicted age in years
            gender: Predicted gender
            confidence: Gender prediction confidence
            embeddings: (optional) wav2vec2 embeddings
        """
        # Load audio
        # audio, sr = sf.read(str(audio_path), always_2d=False)
        audio, sr = librosa.load(str(audio_path), sr=None, mono=False)
        
        return self.predict_from_audio(audio, sr, return_embeddings)
    
    def batch_predict(
        self,
        audio_paths: list,
        batch_size: int = 8
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict age and gender for multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            batch_size: Number of files to process at once
            
        Returns:
            ages: Array of predicted ages
            genders: Array of predicted genders (as strings)
            confidences: Array of gender confidences
        """
        ages = []
        genders = []
        confidences = []
        
        for i in range(0, len(audio_paths), batch_size):
            batch_paths = audio_paths[i:i+batch_size]
            
            for path in batch_paths:
                try:
                    age, gender, conf = self.predict(path)
                    ages.append(age)
                    genders.append(gender)
                    confidences.append(conf)
                except Exception as e:
                    print(f"Error processing {path}: {e}")
                    ages.append(np.nan)
                    genders.append('unknown')
                    confidences.append(0.0)
        
        return np.array(ages), np.array(genders), np.array(confidences)


# ============================================================================
# Example Usage
# ============================================================================

def main():
    """Example usage of the age predictor."""
    
    # Initialize predictor
    predictor = AudeeringAgePredictor()
    
    # Example 1: Predict from a single file
    print("\n" + "="*70)
    print("Example 1: Single file prediction")
    print("="*70)
    
    audio_path = "/data/waris/data/Voxceleb/voxceleb1/train/wav/id10001/1zcIwhmdeo4/00001.wav"
    
    # Make prediction
    age, gender, confidence = predictor.predict(audio_path)
    
    print(f"Predicted age: {age:.1f} years")
    print(f"Gender: {gender} (confidence: {confidence:.2%})")
    
    # Example 2: Get embeddings too
    print("\n" + "="*70)
    print("Example 2: With embeddings")
    print("="*70)
    
    age, gender, confidence, embeddings = predictor.predict(
        audio_path,
        return_embeddings=True
    )
    
    print(f"Predicted age: {age:.1f} years")
    print(f"Gender: {gender}")
    print(f"Embedding shape: {embeddings.shape}")
    
    # Example 3: Batch prediction
    print("\n" + "="*70)
    print("Example 3: Batch prediction")
    print("="*70)
    
    audio_files = [
        "/data/waris/data/Voxceleb/voxceleb1/train/wav/id10001/1zcIwhmdeo4/00001.wav",
        "/data/waris/data/Voxceleb/voxceleb2/train/aac/id00012/_raOc3-IRsw/00110.m4a",
    ]
    
    ages, genders, confidences = predictor.batch_predict(audio_files)
    
    for path, age, gender, conf in zip(audio_files, ages, genders, confidences):
        print(f"{path}: {age:.1f} years, {gender} ({conf:.2%})")


if __name__ == '__main__':
    main()
