import webrtcvad
import numpy as np

import collections
from enum import (
    Enum,
)
from itertools import (
    chain,
)


class EpState(Enum):
    NO_SPEECH = 0
    SPEECH = 1
    WAIT_END = 2


class WebRTCVad:
    def __init__(
        self,
        mode,
    ):
        self._mode = mode
        self.reset()

    def reset(
        self,
    ):
        self._model = webrtcvad.Vad(self._mode)

    def is_speech(
        self,
        buffer,
        sample_rate,
    ):
        return self._model.is_speech(
            buffer,
            sample_rate,
        )


class RingBuffer(collections.deque):
    def __init__(
        self,
        size,
        default=0,
    ):
        self.size = size
        self.default = default
        self.reset()

    @property
    def average(
        self,
    ):
        return sum(self) / len(self)

    def reset(
        self,
    ):
        super(
            RingBuffer,
            self,
        ).__init__(maxlen=self.size)
        for i in range(self.size):
            self.append(self.default)


class Endpointer:
    def __init__(
        self,
        vad=WebRTCVad(3),
        samplerate=8000,
        start_window=200,
        end_window=100,
        head_margin=0,
        tail_margin=0,
        wait_end=1000,
        start_threshold=1,
        end_threshold=0.08,
        wlen=10,
    ):
        self._sr = samplerate
        self._wlen = wlen
        self._start_window = start_window
        self._end_window = end_window
        self._head_margin = head_margin
        self._tail_margin = tail_margin
        self._wait_end = wait_end
        self._start_threshold = start_threshold
        self._end_threshold = end_threshold
        self._wlen = wlen
        self._dtype = "<i2"  # 16bit signed little-endian
        self._end_of_buffer = False
        self._buffer = None
        self._w = int(self._wlen * self._sr / 1000)  # Window size for VAD in samples
        self._start = 0
        self._stop = 0
        self._last_sample = 0
        self._eos_leftovers = []
        self._vad = vad
        self.reset()

    @property
    def last_sample(
        self,
    ):
        return self._last_sample

    @property
    def state(
        self,
    ):
        if self._state == EpState.SPEECH:
            return "SPEECH"
        if self._state == EpState.NO_SPEECH:
            return "NO_SPEECH"
        if self._state == EpState.WAIT_END:
            return "WAIT_END"

    @property
    def start(
        self,
    ):
        """
        Position of the SPEECH_START byte.
        """
        return self._start

    @property
    def stop(
        self,
    ):
        """
        Position of the SPEECH_STOP byte.
        """
        return self._stop

    def reset(
        self,
        hard=False,
    ):
        self._head = RingBuffer(size=self._start_window // self._wlen)
        self._tail = RingBuffer(
            size=self._end_window // self._wlen,
            default=1,
        )
        self._vad.reset()
        self._state = EpState.NO_SPEECH
        self._wl = np.zeros(
            self._w,
            dtype=self._dtype,
        )
        self._buffer = collections.deque(
            maxlen=(self._head.size + self._head_margin // self._wlen)
        )
        if hard:
            self._start = 0
            self._stop = 0
            self._last_sample = 0
            self._eos_leftovers = []
            self._end_of_buffer = False

    def process_samples(
        self,
        samples,
    ):  # noqa
        # If there are any leftovers from the last eos trigger, iterate over them
        # and update the _last_sample position accordingly
        if self._eos_leftovers:
            samples = chain(
                *self._eos_leftovers,
                samples,
            )
            self._last_sample -= len(self._eos_leftovers)
            self._eos_leftovers = []

        for s in samples:
            # Append to VAD window and continue if not filled
            k = self._last_sample % self._w
            self._last_sample += 1
            self._wl[k] = s
            if k != self._w - 1:
                continue

            # Waiting for speech (both NO_SPEECH and WAIT_END)
            if self._state == EpState.NO_SPEECH or self._state == EpState.WAIT_END:
                # Append to _head buffer
                self._buffer.append(np.copy(self._wl))
                self._head.append(
                    int(
                        self._vad.is_speech(
                            self._wl.tobytes(),
                            self._sr,
                        )
                    )
                )
                # Trigger IS_SPEECH when VAD average in the _head buffer exceeds th,
                # yielding contents from head buffer.
                if self._head.average >= self._start_threshold:
                    for ss in self._buffer:
                        for s in ss:
                            yield s
                    self._buffer = collections.deque(maxlen=len(self._tail))
                    self._state = EpState.SPEECH
                    self._start = (
                        self._last_sample
                        - (self._head_margin + self._start_window) // self._wlen
                    )
                    self._head.reset()

                # If WAIT_END state and the buffer has reached the wait_end length,
                # yield contents from tail margin, store the remaining samples for
                # further calls in a TCA regime and return.
                elif (
                    len(self._buffer) == self._wait_end // self._wlen
                    and self._state == EpState.WAIT_END
                ):
                    # Deques don't allow slices, so we explicitly iterate
                    for j in range(self._tail_margin // self._wlen):
                        for s in self._buffer[j]:
                            yield s
                    for j in range(
                        len(self._tail),
                        len(self._buffer),
                    ):
                        self._eos_leftovers.append(self._buffer[j])

                    # Soft reset
                    self._stop = (
                        self._last_sample
                        - (self._wait_end - self._tail_margin) // self._wlen
                    )
                    self.reset()
                    return

            # Yielding speech, waiting for end of speech
            elif self._state == EpState.SPEECH:
                # If buffer size reaches tail length, yield first
                # element from buffer before cycling and discarding
                if len(self._buffer) == self._tail.size:
                    for s in self._buffer[0]:
                        yield s
                self._buffer.append(np.copy(self._wl))
                self._tail.append(
                    int(
                        self._vad.is_speech(
                            self._wl.tobytes(),
                            self._sr,
                        )
                    )
                )
                if self._tail.average <= self._end_threshold:
                    self._buffer = collections.deque(
                        self._buffer,
                        maxlen=self._wait_end // self._wlen,
                    )
                    self._state = EpState.WAIT_END
                    self._tail.reset()
        self._end_of_buffer = True

    def process_bytes(
        self,
        bts,
        yield_samples=False,
    ):
        def gen_samples():
            bs = b""
            for (
                i,
                byte,
            ) in enumerate(bts):
                # Better conversion scheduled for Python 3.9
                # Until then, use this eysore grossness
                bs += byte.to_bytes(
                    1,
                    "little",
                )
                if i % 2 == 1:
                    yield np.frombuffer(
                        bs,
                        dtype=self._dtype,
                    )[0]
                    bs = b""

        for sample in self.process_samples(gen_samples()):
            if yield_samples:
                yield sample
            else:
                for byte in sample.tobytes():
                    yield byte

    def samples_as_chunks(
        self,
        samples,
        reset=True,
    ):
        # Converting to iterator guarantees that even lists will have its position
        # preserved over multiple iterations of "process_samples"
        if not isinstance(
            samples,
            collections.abc.Iterator,
        ):
            samples = iter(samples)
        while not self._end_of_buffer:
            ret = np.array(list(self.process_samples(samples)))
            if len(ret) > 0:
                yield ret
        if reset:
            self.reset(True)

    def bytes_as_chunks(
        self,
        bts,
        yield_samples=False,
        reset=True,
    ):
        if not isinstance(
            bts,
            collections.Iterator,
        ):
            bts = iter(bts)
        while not self._end_of_buffer:
            ret = np.array(
                list(
                    self.process_bytes(
                        bts,
                        yield_samples=True,
                    )
                )
            )
            if len(ret) > 0:
                if not yield_samples:
                    ret = ret.tobytes()
                yield ret
        if reset:
            self.reset(True)


if __name__ == "__main__":
    import soundfile as sf
    from os.path import (
        expanduser as xu,
    )

    res = xu("~/git/kaldi-asr/asr_library/res/audios/8k/")
    path = f"{res}/professor_literatura_8k.wav"

    (wav, rate,) = sf.read(
        path,
        dtype="<i2",
    )
    ep = Endpointer(rate)

    as_chunks = list(ep.samples_as_chunks(wav))

    samples = []
    per_samples = []
    bsize = 10
    for i in range(
        0,
        len(wav) + bsize,
        bsize,
    ):
        block = iter(wav[i : i + bsize])
        while not ep._end_of_buffer:
            samples += list(ep.process_samples(block))
            if ep.state == "NO_SPEECH" and len(samples) != 0:
                per_samples.append(np.array(samples))
                samples = []
        ep._end_of_buffer = False

    print(per_samples)
    assert len(as_chunks) == len(per_samples) == 3
    i = 0
    for (x, y,) in zip(
        as_chunks,
        per_samples,
    ):
        sf.write(
            f"x{i}.wav",
            x,
            rate,
        )
        sf.write(
            f"y{i}.wav",
            y,
            rate,
        )
        i += 1
        print(
            x.shape,
            y.shape,
        )
        print(np.argwhere(x != y))
        assert np.all(x == y)
