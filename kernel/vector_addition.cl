kernel void vector_add(global const int* vec_a, global const int* vec_b, global int* vec_out) {
    const int index = get_global_id(0);
    vec_out[index] = vec_a[index] + vec_b[index];
}