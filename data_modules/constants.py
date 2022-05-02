from datasets.utd_mhad import UTDDatasetManager
from datasets.mmact import MMActDatasetManager
from data_modules.utd_mhad_data_module import UTDDataModule, UTDDataset
from data_modules.mmact_data_module import MMActDataModule, MMActDataset
from data_modules.mmhar_data_module import MMHarDatasetProperties

DATASET_PROPERTIES = {
    "utd_mhad": MMHarDatasetProperties(
        manager_class=UTDDatasetManager,
        dataset_class=UTDDataset,
        datamodule_class=UTDDataModule
    ),
    "mmact": MMHarDatasetProperties(
        manager_class=MMActDatasetManager,
        dataset_class=MMActDataset,
        datamodule_class=MMActDataModule
    )
}