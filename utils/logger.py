import os

DEFAULT_LOG_DIR = "./logs"

class Logger:
	instance=None
	def __init__(self, log_dir):
		self.log_dir = log_dir
		if not os.path.exists(log_dir):
			os.makedirs(log_dir)
		self.log_file = os.path.join(log_dir, "log.txt")
		with open(self.log_file, 'w') as f:
			f.write("Logger initialized.\n")
	@classmethod
	def get_instance(cls, log_dir):
		if cls.instance is None:
			cls.instance = Logger(log_dir)
		return cls.instance
	@classmethod
	def log(cls, message, dir=DEFAULT_LOG_DIR):
		instance = cls.get_instance(dir)
		with open(instance.log_file, 'a') as f:
			f.write(message + "\n")