#define THREADS_PER_HASH 8
#define HASHES_PER_LOOP (GROUP_SIZE / THREADS_PER_HASH)
#define FNV_PRIME	0x01000193

#define fnv(x,y) ((x) * FNV_PRIME ^(y))
#define random() (rand() * rand())

typedef union
{
	unsigned int uint32s[128 / sizeof(unsigned int)];
	uint4	 uint4s[128 / sizeof(uint4)];
	uint2	 uint2s[128 / sizeof(uint2)];
} hash128_t;

static unsigned int fnv_reduce(uint4 v)
{
	return fnv(fnv(fnv(v.x, v.y), v.z), v.w);
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void dagSim(unsigned int search, __global unsigned int * num_results, __global hash128_t * dag, unsigned int num_dag_pages)
{
	__local unsigned int share[HASHES_PER_LOOP];

	const unsigned int gid = get_global_id(0);
	const unsigned int lid = get_local_id(0);

	const int thread_id = lid &  (THREADS_PER_HASH - 1);
	const uint hash_id = (gid % GROUP_SIZE) / THREADS_PER_HASH;

	unsigned int r;
	uint4 mix;
	mix.x = gid;
	mix.y = lid;
	mix.z = hash_id;
	mix.w = gid;

	for (int i = 0; i < THREADS_PER_HASH; i++) {
		if (thread_id == 0)
			share[hash_id] = mix.x;

		uint init0 = share[hash_id];

		barrier(CLK_LOCAL_MEM_FENCE);

		for (int a = 0; a < ACCESSES; a+=4) {
			bool update_share = thread_id == (a / 4) % THREADS_PER_HASH;
			for (uint b = 0; b < 4; b++)
			{
				if (update_share)
				{
					uint m[4] = { mix.x, mix.y, mix.z, mix.w };
					share[hash_id] = fnv(init0 ^ (a + b), m[b]) % num_dag_pages;
				}
				barrier(CLK_LOCAL_MEM_FENCE);
				mix = dag[share[hash_id]].uint4s[thread_id];
			}
		}
		
		share[hash_id] = fnv_reduce(mix);

		if(i == thread_id) {
			r = share[hash_id];
		}
		
	}
	
	if (search == r) {
		atomic_inc(&num_results[0]);
	}
} 