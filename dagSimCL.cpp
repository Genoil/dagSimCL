#include <stddef.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <stdint.h>
#include <fstream>
#include <sstream>
#include <iostream>


#include <CL/cl.h>

using namespace std;
#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0x100000)

#define GRID_SIZE  8192
#define BLOCK_SIZE 256
#define MEGABYTE (1024 * 1024)
#define THREADS_PER_HASH 8
#define ACCESSES 64
#define FNV_PRIME	0x01000193

#define fnv(x,y) ((x) * FNV_PRIME ^(y))

#if RAND_MAX == INT_MAX
#define random_uint() (2 * rand() + (rand() & 0x1))
#elif RAND_MAX == SHRT_MAX
#define random_uint() (4 * (rand() * RAND_MAX + rand()) + (rand() & 0xff))
#endif

#define CL_CHECK(_expr)                                                         \
   do {                                                                         \
     cl_int _err = _expr;                                                       \
     if (_err == CL_SUCCESS)                                                    \
       break;                                                                   \
     fprintf(stderr, "OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);   \
     abort();                                                                   \
    } while (0)

#define CL_CHECK2(_err)                                                         \
   do {                                                                         \
                                                   \
     if (_err == CL_SUCCESS)                                                    \
       break;                                                                   \
     fprintf(stderr, "OpenCL Error: returned %d!\n", (int)_err);   \
     abort();                                                                   \
       } while (0)


#define CL_CHECK_ERR(_expr)                                                     \
   ({                                                                           \
     cl_int _err = CL_INVALID_VALUE;                                            \
     auto _ret = _expr;                                                \
     if (_err != CL_SUCCESS) {                                                  \
       fprintf(stderr, "OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); \
       abort();                                                                 \
	      }                                                                          \
     _ret;                                                                      \
      })

void pfn_notify(const char *errinfo, const void *private_info, size_t cb, void *user_data)
{
	fprintf(stderr, "OpenCL Error (via pfn_notify): %s\n", errinfo);
}

static void addDefinition(string& _source, char const* _id, unsigned _value)
{
	char buf[256];
	sprintf(buf, "#define %s %uu\n", _id, _value);
	_source.insert(_source.begin(), buf, buf + strlen(buf));
}

int main(int argc, char *argv[])
{
	unsigned int buffer_size;
	
	if (argc > 1)
		buffer_size = atoi(argv[1]) * MEGABYTE;
	else
		buffer_size = 1024 * MEGABYTE;

	unsigned int * buffer = (unsigned int *)malloc(buffer_size);

	printf("Genoil's DAGGER simulator\n");
	printf("=========================\n");
	printf("Generating pseudo-DAG of size %u bytes... (will take a few seconds)\n", buffer_size);
	srand(time(NULL));

	for (unsigned int i = 0; i < buffer_size / 4; i++) {
		buffer[i] = random_uint();
	}

	unsigned int h_buffer_size = buffer_size / 128;
	
	unsigned int target;
	target = random_uint();


	cl_platform_id platforms[100];
	cl_device_id devices[100];
	cl_uint platforms_n = 0;
	int platform_id = NULL;
	int device_id = NULL;
	CL_CHECK(clGetPlatformIDs(100, platforms, &platforms_n));

	char strbuf[10240];
	printf("%d OpenCL platform(s) found: \n", platforms_n);
	for (int i = 0; i < platforms_n; i++)
	{
		
		CL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 10240, strbuf, NULL));
		printf("%d: %s\n", i, strbuf);

		
		cl_uint devices_n = 0;
		cl_int ret = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 100, devices, &devices_n);
		if (ret == CL_SUCCESS) {
			printf("   %d OpenCL device(s) found on platform:\n", devices_n);
			for (int j = 0; j < devices_n; j++)
			{
				cl_uint buf_uint;
				cl_ulong buf_ulong;
				CL_CHECK(clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(strbuf), strbuf, NULL));
				printf("%d: %s\n", j, strbuf);
			}
			if (argc > 3 && atoi(argv[3]) == i) {
				platform_id = atoi(argv[3]);
			}
			else {
				platform_id = i;
			}
			if (argc <= 2) {
				device_id = 0;
			}
			else if (argc > 2 && atoi(argv[2]) < devices_n) {
				device_id = atoi(argv[2]);
			}
			else if (argc > 2 && atoi(argv[2]) >= devices_n) {
				printf("   Invalid device id specified, using first device.\n");
				device_id = 0;
			}
		}
		else {
			printf("   No GPU devices found on platform %d.\n", i);
			if (argc > 1 && atoi(argv[1]) == i) {
				if (platforms_n > 1) {
					printf("   Try specifying a different platform id. ");
				}
				printf("Exiting.\n");
				exit(-1);
			}
		}
	}

	if (platforms_n == 0)
		exit(-1);
	
	printf("Using device %d on platform %d\n", device_id, platform_id);

	cl_context_properties contextProperties[] =
	{
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)platforms[platform_id],
		0
	};
	cl_int _err = CL_INVALID_VALUE;
	cl_context context = clCreateContext(contextProperties, 1, &devices[device_id], NULL, NULL, &_err);
	CL_CHECK2(_err);

	ifstream inFile;
	inFile.open("./dagSim.cl");

	stringstream strStream;
	strStream << inFile.rdbuf();
	string code = strStream.str();
	
	printf("Loaded kernel source from %s\n", "./dagSim.cl");

	addDefinition(code, "GROUP_SIZE", BLOCK_SIZE);
	addDefinition(code, "DAG_SIZE", (unsigned)(buffer_size / 128));
	addDefinition(code, "ACCESSES", ACCESSES);
	const char * c = code.c_str();
	const size_t l = code.length();

	cl_program program;
	program = clCreateProgramWithSource(context, 1, &c , &l,  &_err);
	if (clBuildProgram(program, 1, devices, "", NULL, NULL) != CL_SUCCESS) {
		clGetProgramBuildInfo(program, devices[device_id], CL_PROGRAM_BUILD_LOG, sizeof(strbuf), strbuf, NULL);
		fprintf(stderr, "CL Compilation failed:\n%s", strbuf);
		abort();
	}
	CL_CHECK(clUnloadCompiler());

	cl_mem num_results = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint32_t), NULL, &_err);

	cl_mem dag = clCreateBuffer(context, CL_MEM_READ_ONLY, buffer_size, NULL, &_err);


	cl_kernel kernel;
	kernel = clCreateKernel(program, "dagSim", &_err);
	CL_CHECK2(_err);

	CL_CHECK(clSetKernelArg(kernel, 0, sizeof(target), &target));
	CL_CHECK(clSetKernelArg(kernel, 1, sizeof(num_results), &num_results));
	CL_CHECK(clSetKernelArg(kernel, 2, sizeof(dag), &dag));
	
	cl_command_queue queue;
	queue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &_err);
		
	CL_CHECK(clEnqueueWriteBuffer(queue, dag, CL_TRUE, 0, buffer_size, buffer, NULL, NULL, NULL));
	
	printf("Running kernel...\n");
	clFinish(queue);
	size_t g = GRID_SIZE * BLOCK_SIZE;
	size_t w = BLOCK_SIZE;
	cl_event e;
	CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &g, &w, NULL, NULL, &e));
	clWaitForEvents(1, &e);

	cl_ulong time_start, time_end;
	double total_time;

	clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	total_time = time_end - time_start;
	float ms = total_time / 1000000.0;
	float hashes =  GRID_SIZE * BLOCK_SIZE;

	printf("\nExecution time:\t\t%0f ms\n", ms);
	printf("Approximate hashrate:\t%0.1f MH/s\n", hashes / (1000.0 *ms));
	printf("Achieved bandwith:\t%0.1f GB/s\n", (1000.0f / ms) * 16 * hashes * THREADS_PER_HASH * ACCESSES / static_cast<float>(1 << 30));
}



