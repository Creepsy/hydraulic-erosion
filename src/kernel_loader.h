#pragma once

#include <istream>
#include <sstream>
#include <CL/opencl.hpp>

namespace kernel_loader {
    template <class... FunctionSignature>
    struct KernelContext {
        cl::Context context;
        cl::CommandQueue task_queue;
        cl::KernelFunctor<FunctionSignature...> kernel_func;
    };

    template <class... FunctionSignature>
    KernelContext<FunctionSignature...> load_kernel_into_context(
        std::istream& input, 
        const std::string& function_name, 
        const cl::Device& target_device
    );

    // helper functions
    std::string read_all(std::istream& input);
    cl::Program build_program(const std::string& program_source, const cl::Context& program_context, const cl::Device& target_device);
}

#include "kernel_loader.ipp"