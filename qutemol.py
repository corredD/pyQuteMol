from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import ctypes

def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    # Your drawing code here
    glutSwapBuffers()

def init_glut():
    # Initialize GLUT
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(640, 480)
    glutCreateWindow("OpenGL Window")

def init_glew():
    # Initialize GLEW
    glewInit = ctypes.windll.glew32.glewInit
    if glewInit() != 0:
        raise RuntimeError("Failed to initialize GLEW")

def main():
    # Initialize GLUT and GLEW
    init_glut()
    init_glew()

    # Set up display function
    glutDisplayFunc(display)
    
    # Enter the GLUT main loop
    glutMainLoop()

if __name__ == "__main__":
    main()
