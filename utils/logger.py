import sys

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
    def flush(self):
        pass

def logger(log_path, console_print=False):
    """
    Log print statement into file.

    - log_path: the path to save file
    - console_print: print to console
    """
    
    f = open(log_path, 'w')
    backup = sys.stdout
    sys.stdout = open(log_path, 'w')
    if console_print:
        sys.stdout = backup
    sys.stdout = Tee(sys.stdout, f)

