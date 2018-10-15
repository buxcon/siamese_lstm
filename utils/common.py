import sys


class ProgressBar:
    def __init__(self, total, headline, bar_length=100):
        self.total = total
        self.bar_length = bar_length

        print(headline)

    def refresh(self, progress):
        percent = progress / self.total

        hashes = '#' * int(percent * self.bar_length)
        spaces = ' ' * (self.bar_length - len(hashes))

        sys.stdout.write("\r|%s[%.2f%%]%s| %s/%d" %
                         (hashes, percent * 100, spaces, str(progress).rjust(len(str(self.total))), self.total))
        sys.stdout.flush()

    def finish(self, report):
        print("\n%s (total: %d)\n" % (report, self.total))
