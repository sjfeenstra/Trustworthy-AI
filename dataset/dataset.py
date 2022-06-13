"""dataset dataset."""
import tensorflow as tf
import tensorflow_datasets as tfds

# TODO(dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(dataset): BibTeX citation
_CITATION = """
"""


class Dataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for dataset dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'input_imgs': tfds.features.Tensor(shape=(1, 12, 128, 256), dtype=tf.float32),
                'big_input_imgs': tfds.features.Tensor(shape=(1, 12, 128, 256),dtype=tf.float32),
                'desire': tfds.features.Tensor(shape=(1, 8),dtype=tf.float32),
                'traffic_convention': tfds.features.Tensor(shape=(1, 2), dtype=tf.float32),
                'initial_state': tfds.features.Tensor(shape=(1, 512), dtype=tf.float32),
                'outputs': tfds.features.Tensor(shape=(1, 6524), dtype=tf.float32),
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=None,  # Set to `None` to disable
            homepage='https://dataset-homepage/',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(dataset): Downloads the data and defines the splits
        path = dl_manager.download_and_extract('https://todo-data-url')

        # TODO(dataset): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            'train': self._generate_examples(path / 'train_imgs'),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        # TODO(dataset): Yields (key, example) tuples from the dataset
        for f in path.glob('*.jpeg'):
            yield 'key', {
                'image': f,
                'label': 'yes',
            }
