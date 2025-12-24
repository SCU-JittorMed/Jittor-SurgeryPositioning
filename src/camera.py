import pyrealsense2 as rs

def setup_realsense_pipeline(config):
    pipeline = rs.pipeline()
    rs_config = rs.config()
    
    rgb_cfg = config['camera']['rgb']
    depth_cfg = config['camera']['depth']

    rs_config.enable_stream(rs.stream.color, rgb_cfg['width'], rgb_cfg['height'], rs.format.bgr8, rgb_cfg['fps'])
    rs_config.enable_stream(rs.stream.depth, depth_cfg['width'], depth_cfg['height'], rs.format.z16, depth_cfg['fps'])

    align_to = rs.stream.color
    align = rs.align(align_to)
    pipeline.start(rs_config)
    return pipeline, align
