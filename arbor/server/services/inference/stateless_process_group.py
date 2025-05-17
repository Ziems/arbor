#from https://github.com/huggingface/trl/issues/2923#issuecomment-2798926030
import datetime
import pickle
from vllm.distributed.utils import StatelessProcessGroup
from torch.distributed import ProcessGroup, TCPStore


class CustomStatelessProcessGroup(StatelessProcessGroup):
    # New field to isolate barrier rounds.
    barrier_counter: int = 0

    def barrier(self):
        """
        A more resilient barrier that uses a dedicated key namespace.
        """
        # Obtain a unique identifier for this barrier round
        barrier_id = self.barrier_counter
        self.barrier_counter += 1

        # Construct a unique namespace for this barrier round.
        barrier_prefix = f"barrier/{barrier_id}"
        my_key = f"{barrier_prefix}/{self.rank}"

        # Each process signals its arrival by setting its token.
        self.store.set(my_key, pickle.dumps("ready"))

        # Wait for tokens from all processes.
        for r in range(self.world_size):
            key = f"{barrier_prefix}/{r}"
            # The get call will block until the key is available.
            pickle.loads(self.store.get(key))


    @staticmethod
    def create(
            host: str,
            port: int,
            rank: int,
            world_size: int,
            data_expiration_seconds: int = 60,
            store_timeout: int = 30,
    ) -> "StatelessProcessGroup":
        """A replacement for `torch.distributed.init_process_group` that does not
        pollute the global state.

        If we have process A and process B called `torch.distributed.init_process_group`
        to form a group, and then we want to form another group with process A, B, C,
        D, it is not possible in PyTorch, because process A and process B have already
        formed a group, and process C and process D cannot join that group. This
        function is a workaround for this issue.

        `torch.distributed.init_process_group` is a global call, while this function
        is a stateless call. It will return a `StatelessProcessGroup` object that can be
        used for exchanging metadata. With this function, process A and process B
        can call `StatelessProcessGroup.create` to form a group, and then process A, B,
        C, and D can call `StatelessProcessGroup.create` to form another group.
        """  # noqa
        store = TCPStore(
            host_name=host,
            port=port,
            world_size=world_size,
            is_master=(rank == 0),
            timeout=datetime.timedelta(seconds=store_timeout),
        )

        return CustomStatelessProcessGroup(
            rank=rank,
            world_size=world_size,
            store=store,
            data_expiration_seconds=data_expiration_seconds
        )