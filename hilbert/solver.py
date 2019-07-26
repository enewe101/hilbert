import torch

import hilbert as h
from hilbert.tracer import tracer


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
            gradient_accumulation=1,
            gradient_clipping=None
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
        training loop and how they work together.

        Solver.cycle() also support gradient accumulation calculation.
        """

        # Own it like you do
        self.loader = loader
        self.loss = loss
        self.optimizer = optimizer
        self.learner = learner
        self.schedulers = schedulers or []
        self.dictionary = dictionary
        self.verbose = verbose
        self.gradient_accumulation = gradient_accumulation
        self.gradient_clipping = gradient_clipping

        # Other solver state
        self.cur_loss = None
        self.V_norm = None
        self.W_norm = None

    def reset(self, lr=None):
        """
        Re-initialize the learner's parameters and the optimizer's state.
        """
        self.learner.reset()
        self.optimizer.reset(lr)

    def describe(self):
        s = 'Loader: {}\n'.format(self.loader.__class__.__name__)
        s += 'Loss: {}\n'.format(self.loss.__class__.__name__)
        s += 'Optimizer: {}\n'.format(self.optimizer.__class__.__name__)
        s += 'Learner: {}\n'.format(self.learner.__class__.__name__)
        # s += 'Schedulers: {}\n'.format(self.describe_schedulers())
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

    def cycle(self, updates_per_cycle=1, monitor_closely=False):
        grad_accumulation_step = self.gradient_accumulation
        # Run a bunch of updates.
        for update_id in range(updates_per_cycle):
            # Train on as many batches as the loader deems to be one update.
            for batch_id, batch_data in self.loader:
                # Consider this batch and learn.
                response = self.learner(batch_id)

                # clear gradient
                if update_id == 0:
                    self.optimizer.zero_grad()

                self.cur_loss = self.loss(response, batch_data)
                self.cur_loss.backward()
                grad_accumulation_step = grad_accumulation_step - 1

                if self.gradient_clipping is not None:
                    torch.nn.utils.clip_grad_norm_(self.learner.parameters(),
                                                   max_norm=self.gradient_clipping)

                if monitor_closely:
                    if self.cur_loss.item() > 1e4:
                        pos_pairs, neg_pairs = self.loader.get_batch_words(
                            batch_id, self.dictionary)

                        UserWarning(
                            "Extreme loss value is detected. current loss is greater than 1e4,\n"
                            "Positive sample word pairs are :{}\n"
                            "Negative sample word pairs are :{}\n".format(
                                ", ".join(map(str, pos_pairs)),
                                ", ".join(map(str, neg_pairs))))

                if grad_accumulation_step == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    grad_accumulation_step = self.gradient_accumulation

                # learning rate scheduler steps regardless of optimizer update
                for scheduler in self.schedulers:
                    scheduler.step()

                # Nan Police.
                if torch.isnan(self.cur_loss):
                    # Drop your tensors! You're under arrest!
                    del response
                    del self.cur_loss
                    torch.cuda.empty_cache()
                    raise h.exceptions.DivergenceError('Model has diverged!')

                if monitor_closely:
                    tracer.declare('loss', self.cur_loss.item())
                tracer.declare('loss', self.cur_loss.item())
        # don't waste the gradient!
        self.optimizer.step()
        self.optimizer.zero_grad()

        return self.cur_loss.item()
