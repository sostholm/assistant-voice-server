import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.lobes.features import MFCC
from speechbrain.nnet.pooling import StatisticsPooling
from speechbrain.nnet.linear import Linear
from sklearn.preprocessing import normalize

class XVectorNetwork(nn.Module):
    def __init__(self, input_size=40, num_layers=5, hidden_size=512, embedding_size=512):
        super(XVectorNetwork, self).__init__()
        self.tdnn = nn.ModuleList()
        
        # TDNN layers
        in_channels = input_size
        for i in range(num_layers):
            self.tdnn.append(nn.Conv1d(in_channels, hidden_size, kernel_size=5, dilation=1))
            in_channels = hidden_size
        
        # Statistics pooling
        self.stat_pool = StatisticsPooling()
        
        # Fully connected layers
        self.fc1 = Linear(hidden_size * 2, embedding_size)
        self.fc2 = Linear(embedding_size, embedding_size)

    def forward(self, x):
        # x shape: (batch, time, freq)
        x = x.transpose(1, 2)  # (batch, freq, time)
        
        for tdnn_layer in self.tdnn:
            x = F.relu(tdnn_layer(x))
        
        # Statistics pooling
        x = self.stat_pool(x)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

class XVectorSpeakerRecognition:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = XVectorNetwork().to(self.device)
        self.mfcc = MFCC().to(self.device)
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        
        self.model.eval()
        
        self.speaker_embeddings = {}

    def extract_features(self, audio):
        with torch.no_grad():
            mfcc = self.mfcc(audio.to(self.device))
            return self.model(mfcc)

    def register_speaker(self, speaker_id, audio_samples):
        if len(audio_samples) < 3:
            raise ValueError("At least 3 audio samples are required to register a speaker")
        
        embeddings = []
        for audio in audio_samples:
            embedding = self.extract_features(audio)
            embeddings.append(embedding.cpu().numpy())
        
        # Compute average embedding
        average_embedding = np.mean(embeddings, axis=0)
        # Normalize the embedding
        self.speaker_embeddings[speaker_id] = normalize(average_embedding.reshape(1, -1))[0]

    def identify_speaker(self, audio, threshold=0.7):
        embedding = self.extract_features(audio).cpu().numpy()
        embedding = normalize(embedding.reshape(1, -1))[0]
        
        best_score = -np.inf
        best_speaker = None
        
        for speaker_id, speaker_embedding in self.speaker_embeddings.items():
            score = np.dot(embedding, speaker_embedding)
            if score > best_score:
                best_score = score
                best_speaker = speaker_id
        
        if best_score > threshold:
            return best_speaker, best_score
        else:
            return "Unknown", best_score

    def update_speaker_model(self, speaker_id, new_audio):
        if speaker_id not in self.speaker_embeddings:
            raise ValueError(f"Speaker {speaker_id} not found in registered speakers")
        
        new_embedding = self.extract_features(new_audio).cpu().numpy()
        new_embedding = normalize(new_embedding.reshape(1, -1))[0]
        
        # Update the speaker's embedding with a moving average
        alpha = 0.1  # Weight for the new sample
        updated_embedding = (1 - alpha) * self.speaker_embeddings[speaker_id] + alpha * new_embedding
        self.speaker_embeddings[speaker_id] = normalize(updated_embedding.reshape(1, -1))[0]

# Example usage
if __name__ == "__main__":
    speaker_recognition = XVectorSpeakerRecognition()
    
    # Register speakers (in practice, you'd load real audio data)
    speaker_recognition.register_speaker("Alice", [torch.randn(1, 16000) for _ in range(5)])
    speaker_recognition.register_speaker("Bob", [torch.randn(1, 16000) for _ in range(5)])
    
    # Identify a speaker
    test_audio = torch.randn(1, 16000)
    identified_speaker, confidence = speaker_recognition.identify_speaker(test_audio)
    print(f"Identified speaker: {identified_speaker}, confidence: {confidence:.2f}")
    
    # Update a speaker's model with new audio
    speaker_recognition.update_speaker_model("Alice", torch.randn(1, 16000))