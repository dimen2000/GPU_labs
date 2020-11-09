__kernel void test(global int * input,
global int * output, const size_t count)
{
    int grid = get_group_id(0);
    int lid = get_local_id(0);
    int gid = get_global_id(0);

    printf("I am from %d block, %d thread, (global index: %d)", grid, lid, gid);
	if(gid < count) {
		output[gid] = input[gid] + gid;
	}
}