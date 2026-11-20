import tempfile

import pytest

from glotnet.trainer.trainer import Trainer


@pytest.fixture
def tempdir(monkeypatch):
    """ Temporary directory for audio, logs and checkpoints.

    Every TensorBoard writer opened during the test is closed before the
    directory is removed. Otherwise a writer's background flush can recreate
    an event file while rmtree runs, which gives "Directory not empty"
    (ENOTEMPTY) on slow CI runners.
    """
    writers = []
    create_writer = Trainer.create_writer

    def create_and_track(self):
        writer = create_writer(self)
        writers.append(writer)
        return writer

    monkeypatch.setattr(Trainer, 'create_writer', create_and_track)
    with tempfile.TemporaryDirectory() as path:
        yield path
        for writer in writers:
            writer.close()
