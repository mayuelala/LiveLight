from src.models.pose_guider import PoseGuider


class LightGuider(PoseGuider):
    def __init__(
        self,
        conditioning_embedding_channels=320,
        conditioning_channels=12,
        block_out_channels=(16, 32, 96, 256),
    ):
        super().__init__(
            conditioning_embedding_channels=conditioning_embedding_channels,
            conditioning_channels=conditioning_channels,
            block_out_channels=block_out_channels,
        )
