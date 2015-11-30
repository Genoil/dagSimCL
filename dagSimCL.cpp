#include <stddef.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <stdint.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>

#include <CL/cl.h>

using namespace std;
#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0x100000)

#define GRID_SIZE  8192
#define BLOCK_SIZE 256
#define MEGABYTE (1024 * 1024)
#define STEP 128
#define START 128
#define THREADS_PER_HASH 8
#define ACCESSES 64
#define FNV_PRIME	0x01000193

#define CHUNK_SIZE (256 * MEGABYTE)

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
	unsigned int max_buffer_size;
	unsigned int buffer_size;
	unsigned int chunk_size;

	printf("Genoil's DAGGER simulator\n");
	printf("=========================\n\n");
	printf("usage:\n");
	printf("dagSimCL <c> <m> <d> <p>\n");
	printf("c: chunk size. Defaults to 256\n");
	printf("m: max size. Defaults to max GPU RAM or 4096\n");
	printf("d: device id. Defaults to 0\n");
	printf("p: platform id. Defaults to 0\n\n");

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
				CL_CHECK(clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL));

				if (argc > 2 && ((atoi(argv[2]) * MEGABYTE) <= buf_ulong) && atoi(argv[2]) < 4096) {
					buf_ulong = atoi(argv[2]) * MEGABYTE;
				}

				if (buf_ulong > CL_UINT_MAX) {
					max_buffer_size = CL_UINT_MAX;
				}
				else{
					max_buffer_size = buf_ulong;
				}

				if (argc > 1 && atoi(argv[1]) != 0) {
					chunk_size = atoi(argv[1]) * MEGABYTE;
				}
				else {
					chunk_size = 256;
				}
			}
			if (argc > 4 && atoi(argv[4]) == i) {
				platform_id = atoi(argv[4]);
			}
			else {
				platform_id = i;
			}
			if (argc <= 3) {
				device_id = 0;
			}
			else if (argc > 3 && atoi(argv[3]) < devices_n) {
				device_id = atoi(argv[3]);
			}
			else if (argc > 3 && atoi(argv[3]) >= devices_n) {
				printf("   Invalid device id specified, using first device.\n");
				device_id = 0;
			}
		}
		else {
			printf("   No GPU devices found on platform %d.\n", i);
			if (argc > 4 && atoi(argv[4]) == i) {
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


	printf("\nUsing %dMB chunks\n", chunk_size/MEGABYTE);
	printf("Using device %d on platform %d\n", device_id, platform_id);
	
	unsigned int * buffer = (unsigned int *)malloc(max_buffer_size);

	printf("Generating pseudo-DAG of size %u bytes... (will take a minute)\n", max_buffer_size);
	srand(time(NULL));

	for (unsigned int i = 0; i < max_buffer_size / 4; i++) {
		buffer[i] = random_uint();
	}

	unsigned int target;
	

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
	addDefinition(code, "ACCESSES", ACCESSES);
	const char * c = code.c_str();
	const size_t l = code.length();

	cl_program program;
	program = clCreateProgramWithSource(context, 1, &c, &l, &_err);
	if (clBuildProgram(program, 1, devices, "", NULL, NULL) != CL_SUCCESS) {
		clGetProgramBuildInfo(program, devices[device_id], CL_PROGRAM_BUILD_LOG, sizeof(strbuf), strbuf, NULL);
		fprintf(stderr, "CL Compilation failed:\n%s", strbuf);
		abort();
	}

	cl_kernel kernel;
	kernel = clCreateKernel(program, "dagSim", &_err);
	CL_CHECK2(_err);

	cl_command_queue queue;
	queue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &_err);


	cl_mem num_results = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint32_t), NULL, &_err);
	
	unsigned int max_chunks = max_buffer_size / chunk_size;
	cl_mem * dag = new cl_mem[max_chunks];

	unsigned int num_dag_pages;

	filebuf csvfile;
	csvfile.open("results.csv", std::ios::out);
	ostream csvdata(&csvfile);
	csvdata.imbue(std::locale(""));
	csvdata << "DAG size (MB)\tBandwidth (GB/s)\tHashrate (MH/s)" << endl;

	unsigned int full_chunks, rest_size, chunk;

	for (buffer_size = START * MEGABYTE; buffer_size < max_buffer_size; buffer_size += (STEP * MEGABYTE)) {
		full_chunks = buffer_size / chunk_size;
		rest_size = buffer_size % chunk_size;
		for (chunk = 0; chunk < full_chunks; chunk++) {
			dag[chunk] = clCreateBuffer(context, CL_MEM_READ_ONLY, chunk_size, NULL, &_err);
		}

		if (_err != CL_SUCCESS) {
			printf("Out of memory. Bailing.\n");
			break;
		}

		if (rest_size > 0)
			dag[full_chunks] = clCreateBuffer(context, CL_MEM_READ_ONLY, chunk_size, NULL, &_err);

		if (_err != CL_SUCCESS) {
			printf("Out of memory. Bailing.\n");
			break;
		}
		
		printf("Running kernel with %dMB DAG...\n", buffer_size / MEGABYTE);

		target = random_uint();
		num_dag_pages = buffer_size / 128;

		CL_CHECK(clSetKernelArg(kernel, 0, sizeof(target), &target));
		CL_CHECK(clSetKernelArg(kernel, 1, sizeof(num_results), &num_results));
		CL_CHECK(clSetKernelArg(kernel, 2, sizeof(num_dag_pages), &num_dag_pages));
		CL_CHECK(clSetKernelArg(kernel, 3, sizeof(dag[0]), &dag[0]));
		
		for (chunk = 0; chunk < full_chunks; chunk++)
		{
			_err = clEnqueueWriteBuffer(queue, dag[chunk], CL_TRUE, 0, chunk_size, buffer + chunk * chunk_size / sizeof(unsigned int *), NULL, NULL, NULL);
		}
		if (rest_size > 0)
		{
			_err = clEnqueueWriteBuffer(queue, dag[full_chunks], CL_TRUE, 0, rest_size, buffer + full_chunks * chunk_size / sizeof(unsigned int *), NULL, NULL, NULL);
		}
		if (_err != CL_SUCCESS) {
			printf("Out of memory. Bailing.\n");
			break;
		}

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
		double ms = total_time / 1000000.0;
		double hashes = GRID_SIZE * BLOCK_SIZE;
		double hashrate = hashes / (1000.0 *ms);
		double bandwidth = (1000.0f / ms) * 16 * hashes * THREADS_PER_HASH * ACCESSES / static_cast<float>(1 << 30);
		printf("Execution time:\t\t%0f ms\n", ms);
		printf("Approximate hashrate:\t%0.1f MH/s\n", hashrate);
		printf("Achieved bandwith:\t%0.1f GB/s\n\n", bandwidth);

		csvdata << buffer_size / MEGABYTE << "\t" << bandwidth << "\t" << hashrate << endl;

		for (chunk = 0; chunk < full_chunks; chunk++)
			CL_CHECK(clReleaseMemObject(dag[chunk]));

		if (rest_size > 0)
			CL_CHECK(clReleaseMemObject(dag[full_chunks]));
	}
	printf("Writing CSV file\n");
	csvfile.close();
}



