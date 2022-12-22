template <class... FunctionSignature>
kernel_loader::KernelContext<FunctionSignature...> kernel_loader::load_kernel_into_context(
    std::istream& input, 
    const std::string& function_name,
    const cl::Device& target_device) 
{
    const std::string program_source = read_all(input);
    cl::Context program_context(target_device);
    cl::Program program = build_program(program_source, program_context, target_device);

    cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer> kernel_func(program, function_name);

    return KernelContext<FunctionSignature...>{program_context, cl::CommandQueue(program_context, target_device), std::move(kernel_func)};
}