import caffe
import json
import yaml
import numpy as np
import os

TOPIC="camera_purple/rgb/image_raw/compressed"
FPS=30.

from .timeline import Timeline, TASKENGAGEMENT, SOCIALENGAGEMENT, SOCIALATTITUDE

class PInSoRoDatasetLayer(caffe.Layer):

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        self._layer_params = layer_params
        # default batch_size = 256
        self._batch_size = int(layer_params.get('batch_size', 256))
        self._dataset = layer_params.get('dataset')
        self._timesteps = layer_params.get('timesteps') # -> training window size, in frames

        annotations_path = os.path.join(self._dataset, "visual_tracking.annotations.mock.yaml")
        print("Loading annotations %s..." % annotations_path)
        try:
            with open(annotations_path, 'r') as f:
                annotations = yaml.load(f)
                
                self._labels = Timeline(SOCIALENGAGEMENT, annotations["purple"])
                self._current_time = self._labels.start

                #self._targets = np.array([d[1] for d in raw])
        except Exception as e:
            print(str(e))

        pose_path = os.path.join(self._dataset, "visual_tracking.poses.json")
        print("Loading poses %s..." % pose_path)
        try:
            with open(pose_path, 'r') as f:
                self._poses = json.load(f)
        except Exception as e:
            print(str(e))


        print("loading done.")

        assert len(bottom) == 0,            'requires no layer.bottom'
        assert len(top) == 2,               'requires a two layer.top'

        self._current_idx = 0




    def get_next_minibatch(self):
        """Generate next mini-batch
        The return value is array of numpy array: [data, label]
        Reshape function will be called based on results of this function
        Needs to implement in each class
        """


        import pdb;pdb.set_trace()
        sequence, next_ts = self.get_sequence(self._current_time)
        self._current_time = next_ts

        if (self._current_idx + self._batch_size) > len(self._data):
            self._current_idx = 0

        res = [   self._data[self._current_idx:self._current_idx + self._batch_size],
               self._targets[self._current_idx:self._current_idx + self._batch_size]]

        self._current_idx += self._batch_size

        return res


    def forward(self, bottom, top):
        blob = self.get_next_minibatch()
        for i in range(len(blob)):
            top[i].reshape(*(blob[i].shape))
            top[i].data[...] = blob[i].astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        pass

    def reshape(self, bottom, top):
        blob = self.get_next_minibatch()
        for i in range(len(blob)):
            top[i].reshape(*(blob[i].shape))

    def get_features(self, frame):

        skels = frame["poses"]
        if len(skels) == 0:
            raise RuntimeError("no skeleton on frame @%f -- not dealing with that yet!" % frame["ts"])
        if len(skels) > 1:
            raise RuntimeError("more than one skeleton on frame @%f -- not dealing with that yet!" % frame["ts"])


        faces = frame["faces"]
        if len(faces) == 0:
            raise RuntimeError("no face on frame @%f -- not dealing with that yet!" % frame["ts"])
        if len(faces) > 1:
            raise RuntimeError("more than one face on frame @%f -- not dealing with that yet!" % frame["ts"])

        socialvector = []
        for poi in range(18): # skeletons have 18 points
            socialvector.append(skels["1"][poi][0])
            socialvector.append(skels["1"][poi][1])
        for poi in range(70): # faces have 70 points
            socialvector.append(faces["1"][poi][0])
            socialvector.append(faces["1"][poi][1])
        return socialvector

    def get_sequence(self, ts):
        """ Returns a tuple ((list of observations at time ts, label at time
        ts), next timestamp ts) for a duration of 'duration' seconds (ie,
        duration x 30 fps frames).
        
        Note that 'next timestamp' might be further away than ts +
        (timesteps/FPS) as 'get_sequence' might skip in time to find the next
        sequence with a constant label over the whole duration of the sequence.

        If the requested timestamp is outside of
        the range of the annotation stream, an IndexError exception is raised.

        """

        duration = self._timesteps / FPS

        if ts < self._labels.start or ts + duration > self._labels.end:
            raise IndexError("timestamp outside of the annotations range")

        label = self._labels.attime(ts)

        socialfeatures = []

        # CHeck whether the next sequence has a constant label over the whole
        # duration. If not, skip this sequence, and jump to the next one.
        if label != self._labels.attime(ts + duration):
            return self.get_sequence(ts + duration, duration)

        frames = self._poses[TOPIC]["frames"]

        for idx, f in enumerate(frames):
            t = f["ts"]
            if t >= ts:
                for frame in frames[idx:idx + self._timesteps]:
                    features = self.get_features(frame)
                    socialfeatures.append(features)
                break

        return ((socialfeatures, label),  ts + duration)
