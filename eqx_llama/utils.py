def find_intermediate_size(dim: int, multiple_of: int = 256) -> int:
    hidden_dim = 4 * dim

    # For swiglu, as glu uses an extra matrix, we use a reduced dimension.
    reduced_dim = int(2 * hidden_dim / 3)

    # However, we also want it to be a multiple of `multiple_of` (a high power of 2),
    # so we clip to the lowest multiple of `multiple_of` greater or equal to `reduced_dim`.
    final_dim = multiple_of * (reduced_dim // multiple_of)
    if reduced_dim % multiple_of > 0:
        final_dim += multiple_of

    return final_dim
