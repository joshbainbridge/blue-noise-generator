Blue Noise Generator:
=====================

This is a C++ project that generates a blue noise pattern through simulated annealing. Further details can be found in the SIGGRAPH 2016 paper "Blue-noise Dithered Sampling" by Iliyan Georgiev and Marcos Fajardo from Solid Angle. You can access this paper and others on the Solid Angle research page here: https://www.solidangle.com/arnold/research

Compiling and running the program will create two .pgm files, the original white noise pattern, and the new blue noise pattern. It will also produce a text file with the blue noise data in to be used somewhere else. The images just display the first dimension, but the text file should contain as many dimensions as specified by the depth parameter. Sample values in this file are in depth first order. Parameters are currently defined at the beginning of the main function, as members of the sData variable. The only real restriction is that the size parameter (resolution) has to be a multiple of four.

It might also be helpful to note that it was relatively trivial to convert this to use AVX2 instead of SSE3 instruction sets. However this would require at least Haswell architecture, and so I've left the implementation here as is.

![Alt text](../readme-pictures/screenshot_01.png?raw=true)

Required dependencies are:

* CMake 2.8
* TBB 4.2
* SSE 3