import numpy as np


class PicameraSource:
    def __init__(self, width: int = 640, height: int = 480) -> None:
        from picamera2 import Picamera2

        self._camera = Picamera2()
        configuration = self._camera.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self._camera.configure(configuration)
        self._camera.start()

    def read(self) -> np.ndarray | None:
        frame = self._camera.capture_array("main")
        return np.ascontiguousarray(frame[:, :, ::-1])

    def release(self) -> None:
        self._camera.stop()
        self._camera.close()
