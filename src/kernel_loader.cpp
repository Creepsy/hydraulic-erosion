#include "kernel_loader.h"

std::string kernel_loader::read_all(std::istream& input) {
    const size_t curr_pos = input.tellg();
    input.seekg(0, std::ios::end);
    const size_t input_size = (size_t)input.tellg() - curr_pos;
    input.seekg(curr_pos, std::ios::beg);

    std::string content(input_size, '\0');
    input.read(content.data(), input_size);

    return std::move(content);
}

cl::Program kernel_loader::build_program(const std::string& program_source, const cl::Context& program_context, const cl::Device& target_device) {
    cl::Program program = cl::Program(program_context, cl::Program::Sources{program_source});
    
    if (program.build(target_device) != CL_SUCCESS) {
        std::stringstream error_msg;
        error_msg << "Errors occured on program compilation: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(target_device) << std::endl;
        throw std::runtime_error(error_msg.str());
    }

    return program;

}