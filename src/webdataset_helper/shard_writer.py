import os
# import webdataset
import fsspec
import struct

from webdataset import TarWriter
from typing import Any, Callable, Optional, Union, Dict

class ShardWriter:
    """Like TarWriter but splits into multiple shards."""

    def __init__(
        self,
        pattern: str,
        maxcount: int = 100000,
        maxsize: float = 3e9,
        post: Optional[Callable] = None,
        start_shard: int = 0,
        verbose: int = 1,
        **kw,
    ):
        """Create a ShardWriter.

        :param pattern: output file pattern
        :param maxcount: maximum number of records per shard (Default value = 100000)
        :param maxsize: maximum size of each shard (Default value = 3e9)
        :param kw: other options passed to TarWriter
        """
        self.verbose = verbose
        self.kw = kw
        self.maxcount = maxcount
        self.maxsize = maxsize
        self.post = post

        self.tarstream = None
        self.shard = start_shard
        self.pattern = pattern
        self.total = 0
        self.count = 0
        self.size = 0
        self.fname = None

        self.shard_counts = {}

        self.next_stream()

    def next_stream(self):
        """Close the current stream and move to the next."""
        self.finish()
        self.fname = self.pattern % self.shard
        if self.verbose:
            print(
                "# writing",
                self.fname,
                self.count,
                "%.1f GB" % (self.size / 1e9),
                self.total,
            )
        self.shard += 1

        self.fstream = fsspec.open(self.fname, "wb", auto_mkdir=False)
        self.tarstream = TarWriter(self.fstream.__enter__(), **self.kw)

        self.findexstream = fsspec.open(self.fname + ".idx", "wb", auto_mkdir=False)
        self.indexstream = self.findexstream.__enter__()

        self.count = 0
        self.size = 0

    def write(self, obj):
        """Write a sample.

        :param obj: sample to be written
        """
        if (
            self.tarstream is None
            or self.count >= self.maxcount
            or self.size >= self.maxsize
        ):
            self.next_stream()
        
        self.indexstream.write(struct.pack("Q", self.tarstream.tarstream.offset))
        size = self.tarstream.write(obj)
        self.count += 1
        self.total += 1
        self.size += size

    def finish(self):
        """Finish all writing (use close instead)."""
        if self.tarstream is not None:
            # final
            self.indexstream.write(struct.pack("Q", self.tarstream.tarstream.offset))

            self.tarstream.close()
            self.fstream.__exit__()
            self.findexstream.__exit__()

            self.shard_counts[os.path.basename(self.fname)] = self.count

            assert self.fname is not None
            if callable(self.post):
                self.post(self.fname)

            self.tarstream = None
            self.fstream = None
            self.findexstream = None

    def close(self):
        """Close the stream."""
        self.finish()
        del self.tarstream
        del self.fstream
        del self.findexstream
        del self.shard
        del self.count
        del self.size

    def __enter__(self):
        """Enter context."""
        return self

    def __exit__(self, *args, **kw):
        """Exit context."""
        self.close()
