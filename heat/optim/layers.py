import torch

from typing import List
from .dp_optimizer import DASOLayers


__all__ = ["init_bucket_layer"]

def init_bucket_layer(
        layer_to_inheirit_from: torch.nn.Module,
        optimizer: DASOLayers,
        torch_update_layers: List,
        bucket_number: int,
        # last_bucket:
):
    """
    create a comm layer from an existing torch layer

    Parameters
    ----------
    layer_to_inheirit_from

    Returns
    -------

    """
    class BucketReceiveLayer(layer_to_inheirit_from):
        # this is a NN layer which is built to receive the network weights
        # todo: init this for each bucket, not for each layer
        def __init__(
            self,
            optimizer,  # Daso layers instance
            torch_update_layers,  # layers to give the torch bcast wait object to
            bucket_number,  # the bucket number for this layer
            last_bucket=False,  # if this is the last MPI comm bucket
        ):
            # super(WaitLayer, self).__init__()
            # this should get the comm object and the
            # note: the buckets should be created with the __prepare_buckets function of this class
            # bucket key:
            #       names -> names in bucket
            #       bucket -> buffer to put result int
            #       wait -> wait object
            self.torch_update_layers = torch_update_layers
            self.bucket_number = bucket_number
            self.last_bucket = last_bucket
            self.optimizer = optimizer  # reference to the daso optimizer needed for update

        def forward(self, inputs):
            # check to see if this is the next syncing batch
            with torch.no_grad():
                next_rcv_batch = self.optimizer.sync_params_deque[0][0]

                if next_rcv_batch == self.optimizer.current_batch:
                    self._recv_bucket()

                # todo: catch the local bcast wait for the weights on this layer

            # if next_rcv_batch != self.optimizer.current_batch:
            #     # waiting for the next sync still
            #     return

            return inputs

        @torch.no_grad()
        def _recv_bucket(self):
            # take the leftmost sync parameters
            sync_params = self.optimizer.sync_params_deque[self.bucket_number][0]
            # self.current_batch + btw + 1, btw, self.current_sending_ranks
            batches_between = self.optimizer.reduced_ranks[sync_params[1]]
            loc_comm_rank = sync_params[2]
            prev_ranks = self.optimizer.reduced_ranks[loc_comm_rank]

            if self.comm.rank in prev_ranks:
                if self.optimizer.buckets[self.bucket_number]["wait"] is not None:
                    # wait for the bucket to finish sending
                    self.optimizer.buckets[self.bucket_number]["wait"].Wait()

                # update the received parameters
                # NOTE: this assumes that the number of batches to wait DOES NOT CHANGE during an epoch
                numer = batches_between * 2.0 if batches_between > 0.0 else 1.0
                denom = float(self.optimizer.loc_gpus + numer)
                factor = numer / denom

                self.optimizer.buckets[self.bucket_number]["bucket"] /= denom
                bucket = self.optimizer.buckets[self.bucket_number]["bucket"]

                for n in self.optimizer.buckets[self.bucket_number]["names"]:
                    get_slice = self.optimizer.buckets[self.bucket_number]["names"][n]["slice"]
                    shp = self.optimizer.buckets[self.bucket_number]["names"][n]["shape"]
                    update = bucket[get_slice].view(shp)

                    param = self.optimizer.local_model.get_parameter(n)  # get the parameter from the network
                    param *= factor
                    param += update

            if self.last_bucket:  # if this is the last bucket sent, then get rid of these parameters
                self.optimizer.sync_params_deque[self.bucket_number].popleft()

            # start the local bcast op here
            local_waits = {}
            for n in self.optimizer.buckets[self.bucket_number]["names"]:
                param = self.optimizer.local_model.get_parameter(n)  # get the parameter from the network
                # param = self.local_model.get_parameter(param_name)
                local_waits[n] = torch.distributed.broadcast(
                    param, loc_comm_rank, async_op=True
                )  # default is SUM
            # todo: issue the local waits to the respective layers

    return BucketReceiveLayer


class LocalUpdate(torch.nn.Module):
    def __init__(self, next_layer):
        self.wait_list = []
        self.next_layer = next_layer

    def forward(self, inputs):
        with torch.no_grad():
            for w in self.wait_list:
                w.wait()
        return inputs

    @torch.no_grad()
    def __in_layer_local_sync_start(self, root_process):
        # todo: this should be in the network layer class
        # do a bcast from the root process
        # param = self.local_model.get_parameter(param_name)
        # wait = torch.distributed.broadcast(param, root_process, async_op=True)  # default is SUM
        for name, param in self.next_layer.named_parameters():
            if param.requires_grad:
                # overwrite the parameter of the next layer
                wait = torch.distributed.broadcast(param, root_process, async_op=True)

        # pass
        return wait

    # def __in_layer_local_sync_wait(self):
