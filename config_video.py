import cv2 as cv
import video
import sys
import utils
from pathlib import Path


def main(file):
    
    input_file = Path(file)
    config_file = input_file.parent / (input_file.stem + "_config.ini")
    
    print('Opened video file {}'.format(input_file.resolve()))

    parameters = video.Parameters()

    parameters.ask_experiment_settings()

    cap = video.Capture(str(input_file.resolve()))
    cap.capture_frame()
    parameters.video_file = input_file
    parameters.video_frame_count = cap.total_frames
    parameters.video_fps = cap.fps
    parameters.video_length_ms = cap.duration_ms
    parameters.video_original_size = cap.frame.size()
    mid_frame = (parameters.analysis_end_frame - parameters.analysis_start_frame) // 2 + parameters.analysis_start_frame
    cap.capture.set(cv.CAP_PROP_POS_FRAMES, mid_frame)
    cap.capture_frame()
    
    parameters.capture = cap
    
    
    # TODO: Video previous config loading
    # TODO: Add polygon field shape support
    # TODO: Lens correction and other corrections
    # TODO: Add support for editing parameters
    
    parameters.ask_corners()
    parameters.perspective_correction()
    parameters.ask_objects()
    parameters.set_objects()
    parameters.ask_sectors()
    parameters.set_sectors()
    
    utils.save_parameters(config_file, parameters.as_dict())
    

if __name__ == "__main__":
    main(sys.argv[1])