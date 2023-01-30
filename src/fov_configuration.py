import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib.patches import Rectangle
import matplotlib as mpl
import matplotlib.pyplot as plt

import os
import copy
import logging
from typing import Union, Optional

from src.utils import (
    read_maf, 
    read_xml,
    save_maf
)


class FOVConfiguration():
    def __init__(self, file: str):
        self.__load_extensions = {
            '.csv': pd.read_csv,
            '.maf': read_maf,
        }
        self.metadata = read_xml('metadata.xml')
        self.name, self.extension = os.path.splitext(file)
        self.df = self.load(file)

    def load(self, file: str) -> pd.DataFrame:
        try:
            return self.__load_extensions[self.extension](file)
        except KeyError as e:
            raise ValueError(f"{self.extension} files are not supported.")

    def save(self, 
             output_dir: Optional[str] = None, 
             extension: Optional[str] = None,
             ) -> None:
        output_path = os.path.join(
            output_dir, 
            os.path.basename(self.name) + "_boosted" + self.extension
        )
        if extension is None:
            extension = self.extension
        if extension == '.csv':
            self.df.to_csv(output_path, index=False)
        elif extension == '.maf':
            save_maf(self.df, output_path)
        else:
            raise ValueError(f"Unable to save configuration as {extension}. \
                Supported extensions: {self.__load_extensions.keys()}")
        logging.info(f"New configuration saved in {output_dir}")

    def draw2D(self, 
               fovSize: Optional[Union[tuple, int]] = None, 
               title: Optional[str] = None, 
               output_dir: Optional[str] = None,
               enumerate: Optional[bool] = False,
               ) -> None:
        fig = plt.figure(figsize=(10,10))
        cmap = plt.cm.viridis
        norm = mpl.colors.Normalize(
            vmin=self.df['ZPosition'].min(), 
            vmax=self.df['ZPosition'].max()
        )
        color = cmap(norm(self.df['ZPosition'].values))
        plt.scatter(self.df['PosX'], self.df['PosY'], color=color, s=1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        plt.colorbar(sm, location='bottom', label='Stage Z position [um]')
        if fovSize is not None:
            if isinstance(fovSize, int):
                width, height = fovSize, fovSize
            else:
                width, height = fovSize
        else:
            width = self.metadata['resolutionX'] * self.metadata['pointsPerPixelX']
            height = self.metadata['resolutionY'] * self.metadata['pointsPerPixelY']
        ax = plt.gca()
        for i, row in self.df.iterrows():
            xy = (row['PosX'], row['PosY'])
            if enumerate == True:
                plt.annotate(str(i+1), xy=xy)
            ax.add_patch(Rectangle(
                xy=xy ,width=width, height=height, color=color[i], fill=True))
        ax.axis('equal')
        plt.xlabel('Stage X position [pt]')
        plt.ylabel('Stage Y position [pt]')
        if title is None:
            title = 'X, Y, Z positions'
        plt.title(title)
        fig = plt.gcf()
        if output_dir is not None:
            plt.savefig(os.path.join(output_dir, title))

    def draw3D(self, 
               reg: LinearRegression, 
               title: str = None,
               output_dir: Optional[str] = None
               ) -> None:
        x = self.df['PosX']
        y = self.df['PosY']
        z = self.df['ZPosition']
        x_surf, y_surf = np.meshgrid(
            np.linspace(x.min(), x.max(), 100),
            np.linspace(y.min(), y.max(), 100)
        )
        onlyX = pd.DataFrame({'PosX': x_surf.ravel(), 'PosY': y_surf.ravel()})
        fittedY = np.array(reg.predict(onlyX))

        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z,c='red', marker='o', alpha=0.5)
        ax.plot_surface(x_surf,y_surf,fittedY.reshape(x_surf.shape), color='b', alpha=0.3)
        ax.set_xlabel('PosX [pt]')
        ax.set_ylabel('PosY [pt]')
        ax.set_zlabel('ZPosition [um]')
        if title is not None:
            ax.set_title(title)
        if output_dir is not None:
            plt.savefig(os.path.join(output_dir, "3D_" + title))
        plt.show()

    def fitPlane(self):
        reg = LinearRegression().fit(
            self.df[['PosX', 'PosY']], 
            self.df['ZPosition']
        )
        logging.info(f"Linear model coefficients: {reg.coef_}, {reg.intercept_}")
        return reg

    def generateNewFOVs(self, reg):
        min_x_pt = self.df['PosX'].min()
        min_y_pt = self.df['PosY'].min()
        max_x_pt = self.df['PosX'].max()
        max_y_pt = self.df['PosY'].max()

        fov_size_x_pt = self.metadata['resolutionX'] * self.metadata['pointsPerPixelX']
        fov_size_y_pt = self.metadata['resolutionY'] * self.metadata['pointsPerPixelY']

        specimen_size_x_pt = max_x_pt - min_x_pt + fov_size_x_pt
        specimen_size_y_pt = max_y_pt - min_y_pt + fov_size_y_pt

        x, y = [], []
        i = min_x_pt
        while i < min_x_pt + specimen_size_x_pt:
            j = min_y_pt
            while j < min_y_pt + specimen_size_y_pt:
                x.append(i)
                y.append(j)
                j += fov_size_y_pt
            i += fov_size_x_pt

        X = np.array(list(zip(x, y)))
        z = reg.predict(X)
        no_fovs = len(z)

        new_fovs_df = pd.DataFrame(
            {
                "PosX": x, 
                "PosY": y,
                "PosZ": np.ones(no_fovs) * self.df['PosZ'].unique(),
                "ZMode": np.ones(no_fovs) * self.df['ZMode'].unique(),
                "Begin": (z/3.8*1000000000).astype(int),
                "End": (z/3.8*1000000000).astype(int),
                "Sections": np.ones(no_fovs) * self.df['Sections'].unique(),
                "StepSize": np.ones(no_fovs) * self.df['StepSize'].unique(),
                "CycleCount": np.ones(no_fovs) * self.df['CycleCount'].unique(),
                "CycleTime": np.ones(no_fovs) * self.df['CycleTime'].unique(),
                "WaitTime": np.ones(no_fovs) * self.df['WaitTime'].unique(),
                "ValidStack": np.ones(no_fovs) * self.df['ValidStack'].unique(),
                "PositionIdentifier": [f"Position{id+1}" for id in range(no_fovs)],
                "FileNameBase": [self.df['FileNameBase'].unique()[0] for _ in range(no_fovs)],
                "MartixIdentifier": np.ones(no_fovs) * self.df['MartixIdentifier'].unique(),
                "TileScanIdentifier": np.ones(no_fovs) * self.df['TileScanIdentifier'].unique(),
                "AFCOffset": np.ones(no_fovs) * self.df['AFCOffset'].unique(),
                "Valid": np.ones(no_fovs) * self.df['Valid'].unique(),
                "SuperZMode": np.ones(no_fovs) * self.df['SuperZMode'].unique(),
                "ZPosition": z,
            }
        )
        new_fov = copy.deepcopy(self)
        new_fov.df = new_fovs_df
        return new_fov
