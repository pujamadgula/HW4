# Distributed Preconditioned Conjugate Gradient (PCG)

This project implements a distributed Preconditioned Conjugate Gradient (PCG) solver using MPI and Eigen.

## Building the Code

1.  **Create a build directory:**
    ```sh
    mkdir build
    cd build
    ```

2.  **Configure the project using CMake:**
    ```sh
    cmake ..
    ```
    Set your group name:
    ```sh
    cmake .. -DGROUP_NAME="YourGroupName"
    ```

3.  **Compile the code:**
    ```sh
    make
    ```
    This will create an executable named `pcg` in the build directory.

## Running the Code
