import torch


__all__ = ["BucketReceiveLayer"]


class BucketReceiveLayer(torch.nn.Module):
    # this is a NN layer which is built to receive the network weights
    # todo: init this for each bucket, not for each layer
    def __init__(
        self,
        comm_group,
        comm_full,
        global_skips,
        batches_to_wait,
        buckets_key,
        buckets,
        next_layer,
        update_fn,
    ):
        # super(WaitLayer, self).__init__()
        # this should get the comm object and the
        self.comm_group = comm_group
        self.comm_full = comm_full
        self.global_skips = global_skips
        self.batches_to_wait = batches_to_wait
        self.batches_waited = 0
        # note: the buckets should be created with the __prepare_buckets function of this class
        # bucket key:
        #       names -> names in bucket
        #       bucket -> buffer to put result int
        #       wait -> wait object
        self.buckets_key = buckets_key
        self.buckets = buckets
        self.data_sent = False
        self.next_layer_name = next_layer
        self.update_fn = update_fn

    @torch.no_grad()
    def forward(self):
        # need to know there is data to receive
        if self.batch_number == 0 or not self.data_sent:
            self.batch_number += 1
            return

        # if we are waiting
        if self.batches_waited < self.batches_to_wait:
            self.batches_waited += 1
            return

        # if going to receive: are there any other cases?
        if self.batches_waited == self.batches_to_wait:
            # need to receive the bias and the weight
            # make this a try except with the weights and biases

            # todo: get the names of the layers for this bucket

            for name in [".bias", ".weight"]:
                if self.buckets[self.next_layer_name + name]["wait"] is not None:
                    self.buckets[self.next_layer_name + name]["wait"].Wait()
                # get the slice of the data from bucket

            # current position: after the first batch, after waiting for data to be sent

            # next step: do the update after getting the data

            # need to reset the number of batches waited
            self.batches_waited = 0
            self.data_sent = False

        pass

    @torch.no_grad()
    def backward(self):
        pass


class LocalUpdate(torch.nn.Module):
    def __init__(self, params_to_sync):
        self.params_to_sync = params_to_sync
        self.wait_list = []

    @torch.no_grad()
    def forward(self):
        for w in self.wait_list:
            w.wait()

    @torch.no_grad()
    def __in_layer_local_sync_start(self, param_name, root_process):
        # todo: this should be in the network layer class
        # do a bcast from the root process
        param = self.local_model.get_parameter(param_name)
        wait = torch.distributed.broadcast(param, root_process, async_op=True)  # default is SUM
        # for name, param in self.module.named_parameters():
        #     if param.requires_grad:
        #         snds[name].wait()
        # pass
        return wait

    # def __in_layer_local_sync_wait(self):
