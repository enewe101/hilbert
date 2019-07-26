import sys
from datetime import datetime


class Tracer:

    def __init__(
            self,
            solver=None,
            write_path=None,
            verbose=True
    ):
        self.write_path = write_path
        self.trace_file = None if write_path is None else open(write_path, 'w')
        self.verbose = verbose

    def open(self, path):
        if self.trace_file is not None:
            self.trace_file.close()
        self.trace_file = open(path, 'w')

    def start(self, args):
        self.trace('\n'.join(
            '{} = {}'.format(key, args[key]) for key in sorted(args.keys())
        ))

    def command(self):
        self.trace(' '.join(sys.argv))

    def today(self):
        self.trace(datetime.now().strftime('%d %B %Y -- %H:%M:%S'))

    def trace(self, string):
        if self.trace_file is not None:
            self.trace_file.write(string + '\n')
            self.trace_file.flush()
        if self.verbose:
            print(string)

    def declare(self, key, value):
        self.trace('{} = {}'.format(key, value))

    def declare_many(self, dictionary):
        for key, val in dictionary.items():
            self.declare(key, val)

    def step(self):
        """
        This is called before every write.  Anything you trace is both
        printed and logged!
        """
        pass
        # self.trace('loss = {}'.format(self.solver.cur_loss.item()))


tracer = Tracer()
