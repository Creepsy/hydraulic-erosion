#include <sstream> 
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <CL/opencl.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#undef STB_IMAGE_IMPLEMENTATION

#include "kernel_loader.h"

cl::Platform select_platform();
cl::Device select_device(const cl::Platform& platform);

void vector_add_test(kernel_loader::KernelContext<cl::Buffer, cl::Buffer, cl::Buffer>& vector_add_kernel);

int main() {
    const std::string KERNEL_SOURCE_PATH = "../kernel/vector_addition.cl";

    cl::Platform platform;
    try {
        platform = select_platform();      
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        return 1;
    }

    cl::Device target_device;
    try {
        target_device = select_device(platform);      
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        return 1;
    }

    std::ifstream kernel_source_file(KERNEL_SOURCE_PATH);

    if (!kernel_source_file.is_open()) {
        std::cerr << "Unable to open kernel source file'" << KERNEL_SOURCE_PATH << "'!" << std::endl;
        return 1;
    }

    try {
        kernel_loader::KernelContext vector_add_kernel = kernel_loader::load_kernel_into_context<cl::Buffer, cl::Buffer, cl::Buffer>(
            kernel_source_file, 
            "vector_add", 
            target_device
        );
        vector_add_test(vector_add_kernel);
        kernel_source_file.close();
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        kernel_source_file.close();

        return 1;
    }
}

cl::Platform select_platform() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (platforms.empty()) {        
        throw std::runtime_error("No available platforms found!");
    }

    for (size_t platform_index = 0; platform_index < platforms.size(); platform_index++) {
        std::cout << "[" << platform_index << "] " << platforms[platform_index].getInfo<CL_PLATFORM_NAME>() << "\n";
    }
    std::cout << "Choose one of the available platforms: " << std::flush;
    size_t chosen_platform;
    std::cin >> chosen_platform;

    if (chosen_platform > platforms.size()) {
        throw std::runtime_error("Invalid platform id '" + std::to_string(chosen_platform) + "'!");
    }

    return platforms[chosen_platform];
}

cl::Device select_device(const cl::Platform& platform) {
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

    if (devices.empty()) {        
        throw std::runtime_error("No available devices found!");
    }

    for (size_t device_index = 0; device_index < devices.size(); device_index++) {
        std::cout << "[" << device_index << "] " << devices[device_index].getInfo<CL_DEVICE_NAME>() << "\n";
    }
    std::cout << "Choose one of the available devices: " << std::flush;
    size_t chosen_platform;
    std::cin >> chosen_platform;

    if (chosen_platform > devices.size()) {
        throw std::runtime_error("Invalid device id '" + std::to_string(chosen_platform) + "'!");
    }

    return devices[chosen_platform];
}

void vector_add_test(kernel_loader::KernelContext<cl::Buffer, cl::Buffer, cl::Buffer>& vector_add_kernel) {
    std::vector<int> vec_a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<int> vec_b = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    std::vector<int> output(10, 0xdeadbeef); // funny value

    cl::Buffer buffer_a(vector_add_kernel.context, vec_a.begin(), vec_a.end(), true);
    cl::Buffer buffer_b(vector_add_kernel.context, vec_b.begin(), vec_b.end(), true);
    cl::Buffer buffer_out(vector_add_kernel.context, output.begin(), output.end(), false);

    vector_add_kernel.task_queue.enqueueWriteBuffer(buffer_a, CL_TRUE, 0, sizeof(int)*vec_a.size(), vec_a.data());
    vector_add_kernel.task_queue.enqueueWriteBuffer(buffer_b, CL_TRUE, 0, sizeof(int)*vec_b.size(), vec_b.data());

    vector_add_kernel.kernel_func(cl::EnqueueArgs(vector_add_kernel.task_queue, cl::NDRange(10)), buffer_a, buffer_b, buffer_out);

    vector_add_kernel.task_queue.enqueueReadBuffer(buffer_out, CL_TRUE, 0, sizeof(int)*output.size(), output.data());

    for (const int out : output) {
        std::cout << out << " ";
    }
    std::cout << std::endl;
}