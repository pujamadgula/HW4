# Distributed Preconditioned Conjugate Gradient (PCG)

This project aims to implement a distributed Preconditioned Conjugate Gradient (PCG) solver using MPI and Eigen.

## Building the Code

1. **Load CMake:**
    ```sh
    module load cmake
    ```

2.  **Create a build directory:**
    ```sh
    mkdir build
    cd build
    ```

3.  **Configure the project using CMake:**
    ```sh
    cmake -DCMAKE_BUILD_TYPE=Release ..
    ```

4.  **Compile the code:**
    ```sh
    make
    ```
    This will create an executable named `pcg` in the build directory.

## Running the Code

## Submitting the homework
Ensure that your write-up is located in your source directory, next to distributed_pcg.cpp. It should be named cs267XY_hw4.pdf with XY being your group ID.
```sh
cmake -DGROUP_NAME="YourGroupID XY" ..
make package
```
This second command will fail if the PDF is not present. Confirm that it worked using the following command. You should see output like:

```sh
demmel@perlmutter:~/HW4/build> tar tfz cs267XY_hw4.tar.gz 
cs267XY_hw4/cs267XY_hw4.pdf 
cs267XY_hw4/distributed_pcg.cpp
```

Then submit your cs267XY_hw4.tar.gz through bCourses.