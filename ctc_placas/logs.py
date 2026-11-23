import sys
class Logger(object):
    def __init__(self, filename= "default.log",stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, "a")

    def write(self, menssage):
        self.terminal.write(menssage)
        self.log.write(menssage)
        self.log.flush

    def flush(self):
        pass