from arbor.server.services.jobs.job import Job


class GRPOJob(Job):
    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
