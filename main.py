from core.pipeline import UpscalingPipeline
from utils.arg_parser import parse_args


def main():
    config = parse_args()
    pipeline = UpscalingPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()