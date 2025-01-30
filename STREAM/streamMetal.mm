// Objective-C file
#import <Foundation/Foundation.h>
#include <sys/types.h>
#import <Metal/Metal.h>
#import <getopt.h>
#import <stdio.h>


/*
   STREAM benchmark implementation in Metal. Based on CUDA implementation by NVIDIA Corporation.

COPY:       a(i) = b(i)
SCALE:      a(i) = q*b(i)
SUM:        a(i) = b(i) + c(i)
TRIAD:      a(i) = b(i) + q*c(i)

It measures the memory system on the device.
The implementation is in double precision.

Code based on the code developed by John D. McCalpin
http://www.cs.virginia.edu/stream/FTP/Code/stream.c

Written by: Massimiliano Fatica, NVIDIA Corporation

Further modifications by: Ben Cumming, CSCS; Andreas Herten (JSC/FZJ); Sebastian Achilles (JSC/FZJ)

Further modifications by: Gabin Schieffer (KTH)

Metal implementation by: Andong Hu (KTH)
*/

#define NTIMES 20

# ifndef MIN
# define MIN(x,y) ((x)<(y)?(x):(y))
# endif
# ifndef MAX
# define MAX(x,y) ((x)>(y)?(x):(y))
# endif

// Function to load Metal kernel from the .metal file
id<MTLComputePipelineState> loadKernel(NSString *kernelName, id<MTLDevice> device) {
    NSError *error = nil;
    
    NSString *filePath = [[NSBundle mainBundle] pathForResource:@"stream" ofType:@"metal"];
    NSString *source = [NSString stringWithContentsOfFile:filePath encoding:NSUTF8StringEncoding error:&error];
    if (!source) {
        NSLog(@"Error reading Metal file: %@", error.localizedDescription);
        exit(EXIT_FAILURE);
    }

    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    if (!library) {
        NSLog(@"Error creating Metal library: %@", error.localizedDescription);
        exit(EXIT_FAILURE);
    }

    id<MTLFunction> function = [library newFunctionWithName:kernelName];
    if (!function) {
        NSLog(@"Error finding Metal kernel function: %@", kernelName);
        exit(EXIT_FAILURE);
    }

    id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipelineState) {
        NSLog(@"Error creating pipeline state: %@", error.localizedDescription);
        exit(EXIT_FAILURE);
    }

    return pipelineState;
}

void print_help() {
    printf(
        "Usage: stream [-s] [-c [-f]] [-n <elements>] [-b <blocksize>]\n\n"
        "  -s, --si\n"
        "        Print results in SI units (by default IEC units are used)\n\n"
        "  -c, --csv\n"
        "        Print results CSV formatted\n\n"
        "  -f, --full\n"
        "        Print all results in CSV\n\n"
        "  -t, --title\n"
        "        Print CSV header\n\n"
        "  -n <elements>, --nelements <element>\n"
        "        Put <elements> values in the arrays\n"
        "        (default: 1<<26)\n\n"
        "  -b <blocksize>, --blocksize <blocksize>\n"
        "        Use <blocksize> as the number of threads in each block\n"
        "        (default: 192)\n"
    );
}

void parse_options(int argc, char** argv, bool* SI, bool* CSV, bool* CSV_full, bool* CSV_header, int* N, int* blockSize) {
    *SI = false;
    *CSV = false;
    *CSV_full = false;
    *CSV_header = false;
    *N = 1 << 27;
    *blockSize = 192;

    static struct option long_options[] = {
        {"si",        no_argument,       0,  's' },
        {"csv",       no_argument,       0,  'c' },
        {"full",      no_argument,       0,  'f' },
        {"title",     no_argument,       0,  't' },
        {"nelements", required_argument, 0,  'n' },
        {"blocksize", required_argument, 0,  'b' },
        {"help",      no_argument,       0,  'h' },
        {0,           0,                 0,   0  }
    };

    int c;
    int option_index = 0;
    while ((c = getopt_long(argc, argv, "scftn:b:h", long_options, &option_index)) != -1) {
        switch (c) {
            case 's':
                *SI = true;
                break;
            case 'c':
                *CSV = true;
                break;
            case 'f':
                *CSV_full = true;
                break;
            case 't':
                *CSV_header = true;
                break;
            case 'n':
                *N = atoi(optarg);
                break;
            case 'b':
                *blockSize = atoi(optarg);
                break;
            case 'h':
                print_help();
                exit(0);
            default:
                print_help();
                exit(1);
        }
    }
}

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            NSLog(@"Metal is not supported on this device");
            return -1;
        }

        // Parse arguments
        bool SI, CSV, CSV_full, CSV_header;
        int N, blockSize;
        parse_options(argc, (char**)argv, &SI, &CSV, &CSV_full, &CSV_header, &N, &blockSize);

        const float scalar = 3.0f;
        MTLResourceOptions storage = MTLResourceStorageModeShared;
        id<MTLBuffer> bufferA = [device newBufferWithLength:N * sizeof(float) options:storage];
        id<MTLBuffer> bufferB = [device newBufferWithLength:N * sizeof(float) options:storage];
        id<MTLBuffer> bufferC = [device newBufferWithLength:N * sizeof(float) options:storage];

        float *a = (float *)bufferA.contents;
        float *b = (float *)bufferB.contents;
        float *c = (float *)bufferC.contents;

        for (int i = 0; i < N; i++) {
            a[i] = 2.0f;
            b[i] = 0.5f;
            c[i] = 0.5f;
        }

        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        NSArray *kernels = @[@"copyKernel", @"scaleKernel", @"addKernel", @"triadKernel"];

        double times[4][NTIMES] = {0};
        for (int k = 0; k < NTIMES; k++) {
            for (NSUInteger j = 0; j < kernels.count; j++) {
                NSString *kernelName = kernels[j];
                id<MTLComputePipelineState> pipelineState = loadKernel(kernelName, device);

                id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

                switch (j) {
                    case 0:
                        [encoder setLabel:@"Copy"];
                        [encoder setComputePipelineState:pipelineState];
                        [encoder setBuffer:bufferA offset:0 atIndex:0];
                        // [encoder setBuffer:bufferB offset:0 atIndex:1];
                        [encoder setBuffer:bufferC offset:0 atIndex:1];
                        // [encoder setBytes:&scalar length:sizeof(float) atIndex:3];
                        [encoder setBytes:&N length:sizeof(uint) atIndex:2];
                        break;
                    case 1:
                        [encoder setLabel:@"Scale"];
                        [encoder setComputePipelineState:pipelineState];
                        [encoder setBuffer:bufferA offset:0 atIndex:0];
                        //[encoder setBuffer:bufferB offset:0 atIndex:1];
                        [encoder setBuffer:bufferC offset:0 atIndex:1];
                        [encoder setBytes:&scalar length:sizeof(float) atIndex:2];
                        [encoder setBytes:&N length:sizeof(uint) atIndex:3];
                        break;
                    case 2:
                        [encoder setLabel:@"Add"];
                        [encoder setComputePipelineState:pipelineState];
                        [encoder setBuffer:bufferA offset:0 atIndex:0];
                        [encoder setBuffer:bufferB offset:0 atIndex:1];
                        [encoder setBuffer:bufferC offset:0 atIndex:2];
                        // [encoder setBytes:&scalar length:sizeof(float) atIndex:3];
                        [encoder setBytes:&N length:sizeof(uint) atIndex:3];
                        break;
                    case 3:
                        [encoder setLabel:@"Triad"];
                        [encoder setComputePipelineState:pipelineState];
                        [encoder setBuffer:bufferA offset:0 atIndex:0];
                        [encoder setBuffer:bufferB offset:0 atIndex:1];
                        [encoder setBuffer:bufferC offset:0 atIndex:2];
                        [encoder setBytes:&scalar length:sizeof(float) atIndex:3];
                        [encoder setBytes:&N length:sizeof(uint) atIndex:4];
                        break;
                
                }

                MTLSize gridSize = MTLSizeMake(N, 1, 1);
                NSUInteger threadGroupSize = blockSize; // pipelineState.maxTotalThreadsPerThreadgroup;
                MTLSize threadGroupSizeObj = MTLSizeMake(threadGroupSize, 1, 1);

                NSDate *start = [NSDate date];
                [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSizeObj];
                [encoder endEncoding];
                [commandBuffer commit];
                [commandBuffer waitUntilCompleted];
                times[j][k] = [[NSDate date] timeIntervalSinceDate:start];
            }
        }

        double avgtime[4] = {0}, maxtime[4] = {0}, mintime[4] = {DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX};

        for (int k = 1; k < NTIMES; k++) {
            for (int j = 0; j < 4; j++) {
                avgtime[j] += times[j][k];
                mintime[j] = MIN(mintime[j], times[j][k]);
                maxtime[j] = MAX(maxtime[j], times[j][k]);
            }
        }

        for (int j = 0; j < 4; j++) {
            avgtime[j] /= (NTIMES - 1);
        }

        double bytes[4] = {
            2.0 * sizeof(float) * N,
            2.0 * sizeof(float) * N,
            3.0 * sizeof(float) * N,
            3.0 * sizeof(float) * N
        };

        const double G = SI ? 1.e9 : static_cast<double>(1<<30);
        NSString *unit = SI ? @"GB/s" : @"GiB/s";

        printf("\nFunction      Rate %s  Avg time(s)  Min time(s)  Max time(s)\n", [unit UTF8String]);
        printf("-----------------------------------------------------------------\n");
        NSArray *labels = @[@"Copy:      ", @"Scale:     ", @"Add:       ", @"Triad:     "];

        for (int j = 0; j < 4; j++) {
            printf("%s%11.4f     %11.8f  %11.8f  %11.8f\n", [labels[j] UTF8String],
                bytes[j] / mintime[j] / G,
                avgtime[j],
                mintime[j],
                maxtime[j]);
        }
    }
    return 0;
}
