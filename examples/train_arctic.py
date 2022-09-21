from glotnet.data.audio_dataset import AudioDataset
from glotnet.data.config import DataConfig
from glotnet.trainer.trainer import Trainer, TrainerConfig

from torchaudio.datasets import CMUARCTIC

if __name__ == "__main__":

    # dataset = CMUARCTIC(root='/Users/lauri/DATA/torchaudio', url='slt', download=True)


    # TODO: make this a model config, otherwise there's a circular dependency
    config = TrainerConfig(
        skip_channels=64,
        residual_channels=64,
        filter_width=3,
        dilations=(1, 2, 4, 8, 16, 32, 64, 128, 256),
        batch_size=4)
    criterion = Trainer.create_criterion(config)
    model = Trainer.create_model(config, criterion)

    data_config = DataConfig(sample_rate=16000, 
        channels=1, segment_len=8000, 
        padding=model.receptive_field)
    audio_dir = '/Users/lauri/DATA/torchaudio/ARCTIC/cmu_us_slt_arctic/wav'
    dataset = AudioDataset(data_config, audio_dir=audio_dir)

    # TODO: you want to give a Dataset, not a DataLoader
    trainer = Trainer(model=model, criterion=criterion, dataset=dataset, config=config)
    trainer.fit(num_iters=1)


    # x = dataset.__getitem__(0)