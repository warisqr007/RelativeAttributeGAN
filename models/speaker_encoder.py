"""
Frozen pre-trained speaker embedding extractor using ECAPA-TDNN
"""
import torch
import torch.nn as nn
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
from speechbrain.processing.features import Filterbank


class SpeakerEmbeddingExtractor(nn.Module):
    """
    Wrapper for pre-trained ECAPA-TDNN speaker encoder.
    This remains FROZEN during GAN training.
    """
    def __init__(
        self,
        embedding_dim=192,
        pretrained_path=None,
        device='cuda'
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device
        
        # Feature extraction
        self.fbank = Filterbank(n_mels=80)
        
        # ECAPA-TDNN architecture
        self.encoder = ECAPA_TDNN(
            input_size=80,
            channels=[1024, 1024, 1024, 1024, 3072],
            kernel_sizes=[5, 3, 3, 3, 1],
            dilations=[1, 2, 3, 4, 1],
            attention_channels=128,
            lin_neurons=embedding_dim
        )
        
        # Load pre-trained weights
        if pretrained_path:
            self.load_pretrained(pretrained_path)
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        
        self.eval()
    
    def load_pretrained(self, path):
        """Load pre-trained ECAPA-TDNN weights from SpeechBrain or custom checkpoint"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.encoder.load_state_dict(checkpoint['model'], strict=False)
            print(f"✓ Loaded pretrained speaker encoder from {path}")
        except Exception as e:
            print(f"⚠ Warning: Could not load pretrained weights: {e}")
            print("  Using random initialization (not recommended for production)")
    
    @torch.no_grad()
    def forward(self, audio_waveforms):
        """
        Extract speaker embeddings from audio waveforms.
        
        Args:
            audio_waveforms: Tensor of shape (batch, time) containing audio at 16kHz
            
        Returns:
            embeddings: Tensor of shape (batch, embedding_dim)
        """
        # Extract log mel-filterbank features
        feats = self.fbank(audio_waveforms)  # (batch, time, n_mels)
        
        # Get speaker embeddings
        embeddings = self.encoder(feats)  # (batch, embedding_dim)
        
        # L2 normalize embeddings (standard practice for speaker verification)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def extract_from_files(self, audio_paths):
        """Utility to extract embeddings from audio file paths"""
        import torchaudio
        
        embeddings = []
        for path in audio_paths:
            waveform, sr = torchaudio.load(path)
            
            # Resample to 16kHz if needed
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Extract embedding
            with torch.no_grad():
                emb = self.forward(waveform.to(self.device))
            embeddings.append(emb)
        
        return torch.stack(embeddings)
