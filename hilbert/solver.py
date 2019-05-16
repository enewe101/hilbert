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
        This class is responsible for putting "turning the crank" on the
        learning process.  It takes the model through update steps, by
        iterating bathes from the loader, calculating forward passes through
        both the model and the loss function, calling the backwards pass,
        and ticking forward the optimizer's state, along with ticking forward
        any schedulers.

        The main point is to conveniently package up all of the usual things
        that go within the core training loop.  It lets callers of 
        Solver.cycle() concisely ask to iterate forward by some number
        of updates, without needing to know all the pieces of machinery in the
        training loop and how they work together
        """

        # Own it like you do
        self.loader = loader
        self.loss = loss
        self.optimizer = optimizer
        self.learner = learner
        self.schedulers = schedulers or []
        self.dictionary = dictionary
        self.verbose = verbose

        # Other solver state
        self.cur_loss = None


    def describe(self):
        s  = 'Loader: {}\n'.format(self.loader.__class__.__name__)
        s += 'Loss: {}\n'.format(self.loss.__class__.__name__)
        s += 'Optimizer: {}\n'.format(self.optimizer.__class__.__name__)
        s += 'Learner: {}\n'.format(self.learner.__class__.__name__)
        #s += 'Schedulers: {}\n'.format(self.describe_schedulers())
        s += 'Dictionary: {} words\n'.format(len(self.dictionary))
        h.tracer.tracer.trace(s)


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
        return self.learner.get_params()


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

        return self.cur_loss.item()


