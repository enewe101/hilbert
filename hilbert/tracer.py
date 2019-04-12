

class Tracer:

    def __init__(
            self,
            solver=None,
            write_path=None, 
            verbose=True
        ):
        self.solver = solver
        self.write_path = write_path
        self.trace_file = None if write_path is None else open(write_path, 'w')
        self.verbose = verbose


    def start(self, args):
        self.trace('\n'.join(
            '{} = {}'.format(key, args[key]) for key in sorted(args.keys())
        ))
        self.trace(self.solver.describe())


    def trace(self, string):
        if self.trace_file is not None: 
            self.trace_file.write(string+'\n')
        if self.verbose: 
            print(string)


    def step(self):
        """
        This is called before every write.  Anything you trace is both
        printed and logged!
        """
        self.trace('loss = {}'.format(self.solver.cur_loss.item()))



