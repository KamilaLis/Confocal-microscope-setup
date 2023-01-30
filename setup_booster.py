from argparse import ArgumentParser
import os
import logging.config

from src.fov_configuration import FOVConfiguration


logging.config.fileConfig("logging.conf")


class SetupBooster:
    def __init__(self):
        self.app_args = self.app_arguments()
        if not os.path.exists(self.app_args.output_dir):
            os.makedirs(self.app_args.output_dir)

    def app_arguments(self):
        parser = ArgumentParser()
        parser.add_argument(
            '-f',
            '--file',
            type=str,
            help='Path to the configuration file (csv or maf).',
            required=True
        )
        parser.add_argument(
            '-o',
            '--output-dir',
            type=str,
            help='Output directory for created configuration.',
            required=True
        )
        return parser.parse_args()

    def run(self) -> None:
        out_dir = self.app_args.output_dir

        fc = FOVConfiguration(self.app_args.file)
        reg = fc.fitPlane()
        generated_fc = fc.generateNewFOVs(reg)

        fc.draw2D(
            title='Input X, Y, Z positions', 
            output_dir=out_dir,
            enumerate=True)
        fc.draw3D(
            reg,
            title='Input X, Y, Z positions', 
            output_dir=out_dir)

        generated_fc.draw2D(
            title='Generated X, Y, Z positions', 
            output_dir=out_dir)
        generated_fc.draw3D(
            reg,
            title='Generated X, Y, Z positions', 
            output_dir=out_dir)

        generated_fc.save(out_dir)


if __name__ == '__main__':
    SetupBooster().run()
