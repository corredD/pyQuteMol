===========
 pyQuteMol
===========

A Python port of C++ QuteMol_

:Author: Naveen Michaud-Agrawal
:Year:   2007
:License: GPL v2

------------------------------------------------------------

.. Warning:: Code is broken but might be useful as a starting point
             for new projects. Use the `Issue Tracker`_ to start a 
             discussion or fork and send pull requests or make it 
             your own!

------------------------------------------------------------


Naveen wrote the code some time in 2007 and it is made public with his
permission. He says:

  pyQuteMol was a direct port of the C++ Qutemol
  (http://qutemol.sourceforge.net/), so I guess it falls under the
  same license (GPL)? Most of the interesting work is done in the
  shader code which was copied verbatim from Qutemol (my only
  improvement was using numpy arrays to populate the opengl buffers so
  I could hook it into MDAnalysis_ to animate trajectories). The code
  should be workable if somebody is familiar with OpenGL (particularly
  the new programmable pipeline) and how to set it up in python (I
  think I had a port to pyglet at one point since that supported
  vertex and fragment shaders). Feel free to use/abuse as you see fit
  (I guess within the constraints of the GPL :)


Installing
==========

Try ::

  python setup.py build

 
.. _QuteMol:  http://qutemol.sourceforge.net/
.. _Issue Tracker: https://github.com/MDAnalysis/pyQuteMol/issues
.. _MDAnalysis: http://www.mdanalysis.org

Usage
=====

Usage::

  qutemol.py PREFIX IS_TRJ IS_COARSEGRAIN

* *PREFIX*: Looks for ``PREFIX.pdb`` or ``PREFIX.psf PREFIX.dcd``.
* *IS_TRJ*: 0: look for pdb; 1: look for psf/dcd combo
* *IS_COARSEGRAIN*: 0: atomistic; 1: hacks for coarse grained structures


pip install pillow
pip install cython
pip install PyOpenGL
pip install mdanalysis
# install freeglut and glew
python setup.py build_ext --inplace --verbose

python qutemol.py 1crn.pdb 0 0

# pygfx
pip install -U pygfx glfw
pip install pylinalg
pip install gemmi
python multi_select.py
 
 Download the file for your platform. If you're not sure which to choose, learn more about  installing packages.
