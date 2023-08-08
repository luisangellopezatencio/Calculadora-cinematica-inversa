# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 18:32:21 2023

@author: Luis Ángel López Atencio
"""

import trimesh
import pyvista as pv
import numpy as np
import Utils as hr

#Importar aarchivo stl

# Importar archivo stl
malla_trimesh = trimesh.load('fig3d1.stl')

# Convertir malla de Trimesh a malla de PyVista
malla_pv = pv.wrap(malla_trimesh)


# Visualizar la malla con PyVista
plotter = pv.Plotter()
figu = plotter.add_mesh(malla_pv)
S0 = hr.dibujar_sistema_referencia_MTH(np.eye(4), 25, '0', plotter)
plotter.show_grid()
plotter.show()
