import os
import os.path as op
import cv2
import logging
import datetime
import numpy as np
from pathlib2 import Path
from collections import namedtuple

logger = logging.getLogger(__name__)
Size = namedtuple("Size", ["height", "width"])

def get_folder(path):
    """ Return a path to a folder, creating it if it doesn't exist 
    Args:
        path: The path of the new folder.
    Returns:
        _ : The guaranteed path of the folder/file.
    """
    logger.debug(f"Requested path: {path}")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def stem(path):
    """ Return the stem of a file in the input path.
    """
    return os.path.splitext(os.path.basename(path))[0]

class VideoReader:
    """Helper class for reading one or more frames from a video file."""

    def __init__(self, path=None, color_mode='RGB', insets=(0, 0)):
        """Creates a new VideoReader.

        Arguments:
            insets: amount to inset the image by, as a percentage of
                (width, height). This lets you "zoom in" to an image
                to remove unimportant content around the borders.
                Useful for face detection, which may not work if the
                faces are too small.
        """
        self.insets = insets
        self.vidcap = None
        self.path = path
        self.color_mode = color_mode
        
        # self.vidcap = None
        # self._frame_count = None
        # self._fps = None
        # self._size = None
        # self._meta = None
        
    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        if self.vidcap is not None:
            self.vidcap.release()
            self.vidcap = None
            
        self._path = path
        if path is not None:
            try:
                self.vidcap = cv2.VideoCapture(self._path)
            except:
                logger.error(f"Error: Failed to open the video: {self._path}")
                self.vidcap = None
                self._path = None

        self._frame_count = None
        self._fps = None
        self._size = None
        self._meta = None
    
    def close(self):
        if self.vidcap is not None:
            self.vidcap.release()
            self.vidcap = None
    
    def reset(self):
        # reset the frame index to 0
        if self.vidcap is not None:
            self.vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    @property
    def color_mode(self):
        return self._color_mode
    
    @color_mode.setter
    def color_mode(self, value):
        color_mode = value.upper()
        if color_mode in ['RGB', 'BGR', 'GRAY', 'GREY']:
            self._color_mode = color_mode
        else:
            raise ValueError(f"Error: Invalid color mode: {value}")

    @property
    def fps(self):
        if self._fps is None:
            self._fps = self.vidcap.get(cv2.CAP_PROP_FPS)
        return self._fps

    @property
    def frame_count(self):
        if self._frame_count is None:
            self._frame_count = int(self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        return self._frame_count
    
    @frame_count.setter
    def frame_count(self, value):
        self._frame_count = value

    @property
    def width(self):
        return self.size.width
    
    @property
    def height(self):
        return self.size.height

    @property
    def seconds(self):
        return self.frame_count/self.fps

    @property
    def time_string(self):
        return str(datetime.timedelta(seconds=self.seconds))

    @property
    def size(self):
        if self._size is None:
            width = np.int32(self.vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
            height = np.int32(self.vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
            self._size = Size(height, width)
        return self._size

    @property
    def metadata(self):
        """Extract the necessary information from a video file.

        Returns:
            frame_count: The total frame number of the video.
            fps: The `frame per second` value of the video.
            width: The frame width of the video.
            height: The frame height of the video.
        """
        if self._meta is None:
            self._meta = {"fps": self.fps, "frame_count": self.frame_count,
                          "size": self.size}
        return self._meta

    def read_isometric_frames(self, num_frames, jitter=0, seed=None):
        """Reads frames that are always evenly spaced throughout the video.

        Arguments:
            num_frames: how many frames to read, -1 means the entire video
                (warning: this will take up a lot of memory!)
            jitter: if not 0, adds small random offsets to the frame indices;
                this is useful so we don't always land on even or odd frames
            seed: random seed for jittering; if you set this to a fixed value,
                you probably want to set it only on the first video 
        """
        assert num_frames > 0

        if self.frame_count <= 0:
            return None

        frame_idxs = np.linspace(
            0, self.frame_count - 1, num_frames, endpoint=True, dtype=np.int)
        logger.debug(f"Frame indexes: {frame_idxs}")
        if jitter > 0:
            np.random.seed(seed)
            jitter_offsets = np.random.randint(-jitter,
                                               jitter, len(frame_idxs))
            frame_idxs = np.clip(
                frame_idxs + jitter_offsets, 0, self.frame_count - 1)

        result = self._read_frames_at_indices(self._path, self.vidcap, frame_idxs)
        return result

    def read_all_frames(self):
        if self.frame_count <= 0:
            return None
        frame_idxs = np.arange(self.frame_count)
        result = self._read_frames_at_indices(self._path, self.vidcap, frame_idxs)
        return result

    def to_images(self):
        self.reset()
        if self.frame_count <= 0:
            return None

        output_root = get_folder(
            op.join(*op.split(self.path)[:-1], stem(self.path)))

        count = 0
        while self.vidcap.isOpened():
            success, image = self.vidcap.read()
            if success:
                cv2.imwrite(os.path.join(output_root, '%d.png') % count, image)
                count += 1
            else:
                break
        # cv2.destroyAllWindows()

    def read_random_frames(self, num_frames, seed=None):
        """Picks the frame indices at random.

        Arguments:
            num_frames: how many frames to read, -1 means the entire video
                (warning: this will take up a lot of memory!)
        """
        assert num_frames > 0
        np.random.seed(seed)

        if self.frame_count <= 0:
            return None

        frame_idxs = sorted(np.random.choice(
            np.arange(0, self.frame_count), num_frames))
        return self._read_frames_at_indices(frame_idxs)

    def read_frames_at_indices(self, frame_idxs):
        """Reads frames from a video and puts them into a NumPy array.

        Arguments:
            frame_idxs: a list of frame indices. Important: should be
                sorted from low-to-high! If an index appears multiple
                times, the frame is still read only once.

        Returns:
            - a NumPy array of shape (num_frames, height, width, 3)
            - a list of the frame indices that were read

        Reading stops if loading a frame fails, in which case the first
        dimension returned may actually be less than num_frames.

        Returns None if an exception is thrown for any reason, or if no
        frames were read.
        """
        assert len(frame_idxs) > 0
        return self._read_frames_at_indices(frame_idxs)
        
    def _read_frames_at_indices(self, frame_idxs, return_idx=False):
        try:
            frames = []
            idxs_read = []
            for frame_idx in frame_idxs:
                frame = self.read_frame_at_index(frame_idx)
                if frame is not None:
                    frames.append(frame)
                    idxs_read.append(frame_idx)
            
            if len(frames) > 0:
                frames = np.stack(frames, axis=0)
                if return_idx:
                    return frames, idxs_read
                else:
                    return frames

            logger.info("No frames read from movie %s" % self.path)
            return None
        except:
            logger.exception("Exception while reading movie %s" % self.path)
            return None

    def read_middle_frame(self):
        """Reads the frame from the middle of the video."""
        return self._read_frame_at_index(self.frame_count // 2)

    def read_frame_at_index(self, frame_idx):
        """Reads a single frame from a video.

        If you just want to read a single frame from the video, this is more
        efficient than scanning through the video to find the frame. However,
        for reading multiple frames it's not efficient.

        My guess is that a "streaming" approach is more efficient than a 
        "random access" approach because, unless you happen to grab a keyframe, 
        the decoder still needs to read all the previous frames in order to 
        reconstruct the one you're asking for.

        Args:
            frame_idx: Integer. The index of the frame you are going to read.
        """
        return self._read_frame_at_index(frame_idx)

    def read_random_frame(self, seed=None):
        """ Read a random frame from a video.
        Args:
            seed: The random seed for the random number generator.
        """
        np.random.seed(seed)
        frame_idx = np.random.choice(range(self.frame_count))
        return self._read_frame_at_index(frame_idx)

    def _read_frame_at_index(self, frame_idx):
        """ Read a frame from a video.
        Args:
            frame_idx: The index of the frame you are going to read.
        Returns:
            frame: The frame you read from the video.
        """
        self.vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.vidcap.read()
        if not ret or frame is None:
            logger.error("Error: Failed to retrieve frame %d from movie %s" %
                         (frame_idx, self.path))
            return None
        else:
            frame = self._postprocess_frame(frame)
            return frame

    def _postprocess_frame(self, frame):
        if self.color_mode == 'GRAY' or self.color_mode == 'GREY':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif self.color_mode == 'RGB':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.insets[0] > 0:
            W = frame.shape[1]
            p = int(W * self.insets[0])
            frame = frame[:, p:-p]

        if self.insets[1] > 0:
            H = frame.shape[1]
            q = int(H * self.insets[1])
            frame = frame[q:-q, :]

        return frame