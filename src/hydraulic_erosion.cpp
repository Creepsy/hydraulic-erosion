#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300
 
#include <iostream>
#include <vector>
#include <string>
#include <CL/opencl.hpp>

const std::string KERNEL_CODE = R"(
kernel void vector_add(global const int* vec_a, global const int* vec_b, global int* vec_out) {
    int index = get_global_id(0);
    vec_out[index] = vec_a[index] + vec_b[index]; 
}
)";

cl::Platform select_platform();
cl::Device select_device(const cl::Platform& platform);

int main() {
    cl::Platform platform;
    try {
        platform = select_platform();      
    } catch(const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        return 1;
    }

    cl::Device device;
    try {
        device = select_device(platform);      
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        return 1;
    }

    cl::Context context(device);
    cl::Program::Sources kernel_sources;
    kernel_sources.push_back(KERNEL_CODE);

    cl::Program program = cl::Program(context, kernel_sources);
    if (program.build(device) != CL_SUCCESS) {
        std::cerr << "Errors occured on program compilation: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        return 1;
    }

    std::vector<int> vec_a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<int> vec_b = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    std::vector<int> output(10, 0xdeadbeef); // funny value

    cl::Buffer buffer_a(context, vec_a.begin(), vec_a.end(), true);
    cl::Buffer buffer_b(context, vec_b.begin(), vec_b.end(), true);
    cl::Buffer buffer_out(context, output.begin(), output.end(), false);

    cl::CommandQueue command_queue(context, device);
    command_queue.enqueueWriteBuffer(buffer_a, CL_TRUE, 0, sizeof(int)*10, vec_a.data());
    command_queue.enqueueWriteBuffer(buffer_b, CL_TRUE, 0, sizeof(int)*10, vec_b.data());

    cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer> vector_addition(program, "vector_add");
    vector_addition(cl::EnqueueArgs(command_queue, cl::NDRange(10)), buffer_a, buffer_b, buffer_out); // what exactly are EnqueueArgs?

    command_queue.enqueueReadBuffer(buffer_out, CL_TRUE, 0, sizeof(int)*10, output.data());

    for (const int out : output) {
        std::cout << out << " ";
    }
    std::cout << std::endl;
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