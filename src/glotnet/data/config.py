from dataclasses import dataclass

@dataclass
class DataConfig:
    """ GlotNet Data configuration """

    # audio properties
    sample_rate: int = 16000
    channels: int = 1
    segment_len: int = 8000
    padding: int = 0

    # acoustic features
    hop_size: int = 256
    frame_size: int = 1024
    nfft: int = 1024
    num_mels: int = 80
    
    @property
    def segment_dur(self) -> float:
        return float(self.segment_len) / float(self.sample_rate)

    @segment_dur.setter
    def segment_dur(self, dur: float) -> None:
        self.segment_len = int(dur * self.sample_rate)


