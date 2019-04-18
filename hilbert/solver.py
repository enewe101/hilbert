import hilbert as h
import torch


class Solver(object):

    def __init__(
        self,
        loader,
        loss,
        learner,
        optimizer,
        schedulers=None,
        dictionary=None,
        verbose=True,
    ):
        """
        This is the base class for a Hilbert Embedder model. It uses pytorch's
        automatic differentiation for the primary heavy lifting. The fundamental
        components of this class are:
            (1) the loss function (loss.py)
            (2) the optimizer *constructor* (from torch.optim)
            (3) the dimensionality (an integer for the size of the embeddings)

        :param loader: a Loader object that iterates gpu-loaded shards
        :param optimizer: a constructor from torch.optim that we 
                will build later
        """

        # Own it like you do
        self.loader = loader
        self.loss = loss
        self.optmizer = optimizer
        self.optimizer = optimizer
        self.learner = learner
        self.schedulers = schedulers or []
        self.dictionary = dictionary
        self.verbose = verbose

        # Other solver state
        self.cur_loss = None


    def describe(self):
        return "I'm the solver."


    def get_embeddings(self):
        detached_embedding_params = (
            p.detach() if p is not None else None
            for p in self.learner.get_embedding_params()
        )
        return h.embeddings.Embeddings(
            *detached_embedding_params, 
            dictionary=self.dictionary,
            verbose=self.verbose
        )
        

    def get_params(self):
        return learner.get_params()


    def cycle(self, updates_per_cycle=1):

        # Run a bunch of updates.
        for update_id in range(updates_per_cycle):

            # Train on as many batches as the loader deems to be one update.
            for batch_id, batch_data in self.loader:

                # Consider this batch and learn.
                self.optimizer.zero_grad()
                response = self.learner(batch_id)
                self.cur_loss = self.loss(response, batch_data)
                self.cur_loss.backward()

                # Take some steps
                self.optimizer.step()
                for scheduler in self.schedulers:
                    scheduler.step()

                # Nan Police.
                if torch.isnan(self.cur_loss):
                    # Drop your tensors! You're under arrest!
                    del response
                    del self.cur_loss
                    torch.cuda.empty_cache()
                    raise h.exceptions.DivergenceError('Model has diverged!')



