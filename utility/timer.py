import time


class Timer:

    def __init__(self):

        self.t = time.time()

    def log(self, log_str):

        dt = time.time() - self.t
        print("\n----------\n [{}] Delta t: {} \n----------".format(log_str, dt), flush=True)
        self.t = time.time()

    def reset(self):

        self.t = time.time()