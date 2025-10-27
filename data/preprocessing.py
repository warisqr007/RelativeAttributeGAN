from models.speaker_encoder import SpeakerEmbeddingExtractor
import glob
import torch

# Initialize encoder
encoder = SpeakerEmbeddingExtractor(
    pretrained_path='pretrained/ecapa_tdnn.pt'
)

# Extract embeddings from audio files
audio_files = glob.glob('data/audio/*.wav')
embeddings = encoder.extract_from_files(audio_files)

# Save
torch.save(embeddings, 'data/train_embeddings.pt')
