from dataclasses import dataclass, field

# Creating frozen class for constant configs
@dataclass(frozen=True)
class Configs():
    """
    All configuration needed for Generator, Discriminator, Optimizer, Dataset, and Training process

    * Dangerous configurations have a '#*' at the end
    """

    ## Dataset
    content_images_directory = "../content_images"
    style_images_directory = "../style_images"
    content_batch_size = 4
    content_numworkers = 8

    ## Loss Functions
    content_weight = 1.0
    style_weight = 10.0
