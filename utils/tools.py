from pathlib import Path
import time

class Step():
    def __init__(self):
        self.step = 0
        self.round = {}
    def clear(self):
        self.step = 0
        self.round = {}
    def forward(self, x):
        self.step += x
    def reach_cycle(self, mod, ignore_zero = True):
        now = self.step // mod
        if now==0 and ignore_zero:
            return False
        if mod not in self.round or self.round[mod]!=now: #新过了一个或多个cycle
            self.round[mod] = now
            return True
        return False
    def state_dict(self):
        return {'step': self.step, 'round':self.round}
    def load_state_dict(self, state):
        self.step = state['step']
        self.round = state['round']
    @property
    def value(self):
        return self.step

class Logger():
    def __init__(self, file_name, mode = 'w', buffer = 100):
        (Path(file_name).parent).mkdir(exist_ok = True, parents = True)
        self.file_name = file_name
        self.fp = open(file_name, mode)
        self.cnt = 0
        self.stamp = time.time()
        self.buffer = buffer
    def log(self, *args, end='\n'):
        for x in args:
            if isinstance(x, dict):
                for y in x:
                    self.fp.write(str(y)+':'+str(x[y])+' ')
            else:
                self.fp.write(str(x)+' ')
        self.fp.write(end)
        self.cnt += 1
        if self.cnt>=self.buffer or time.time()-self.stamp>5:
            self.cnt = 0
            self.stamp = time.time()
            self.fp.close()
            self.fp = open(self.file_name, 'a')
        pass
    def close(self):
        self.fp.close()