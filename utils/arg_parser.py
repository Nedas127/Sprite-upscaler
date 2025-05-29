"""
Argument Parser
--------------
Command line argument parsing for the pipeline.
"""

import sys
from config.config_models import PipelineConfig


def parse_args() -> PipelineConfig:
    """Parse command line arguments and return a config object"""
    config = PipelineConfig()

    if len(sys.argv) > 1:
        if sys.argv[1].startswith("--resize="):
            resize_str = sys.argv[1].split("=")[1]
            config.resize.enabled = True

            if "x" in resize_str:
                width, height = map(int, resize_str.split("x"))
                config.resize.dimensions = (width, height)
                config.resize.scale = None
                print(f"Command line resize: output will be {width}x{height} pixels")
            else:
                try:
                    scale = float(resize_str)
                    if scale > 1:
                        scale = scale / 100.0
                    config.resize.scale = scale
                    config.resize.dimensions = None
                    print(f"Command line resize: output will be scaled to {int(scale * 100)}%")
                except ValueError:
                    print("Invalid resize parameter. Use --resize=WIDTHxHEIGHT or --resize=SCALE")
                    sys.exit(1)

            if len(sys.argv) > 2 and not sys.argv[2].startswith("--"):
                config.specific_model = sys.argv[2]
                print(f"Command line model: {config.specific_model}")
        else:
            config.specific_model = sys.argv[1]
            print(f"Command line model: {config.specific_model}")

            if len(sys.argv) > 2 and sys.argv[2].startswith("--resize="):
                resize_str = sys.argv[2].split("=")[1]
                config.resize.enabled = True

                if "x" in resize_str:
                    width, height = map(int, resize_str.split("x"))
                    config.resize.dimensions = (width, height)
                    config.resize.scale = None
                    print(f"Command line resize: output will be {width}x{height} pixels")
                else:
                    try:
                        scale = float(resize_str)
                        if scale > 1:
                            scale = scale / 100.0
                        config.resize.scale = scale
                        config.resize.dimensions = None
                        print(f"Command line resize: output will be scaled to {int(scale * 100)}%")
                    except ValueError:
                        print("Invalid resize parameter. Use --resize=WIDTHxHEIGHT or --resize=SCALE")
                        sys.exit(1)

    return config
