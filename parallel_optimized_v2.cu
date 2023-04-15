#include <stdio.h>
#include <stdint.h>
#include <string>
#include <cmath>
#include <algorithm>

using namespace std;

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(EXIT_FAILURE);\
    }\
}

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);                                                                 
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

void readPnm(char * fileName, int &width, int &height, uchar3 * &pixels)
{
    FILE * f = fopen(fileName, "r");
    if (f == NULL)
    {
        printf("Cannot read %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    char type[3];
    fscanf(f, "%s", type);
    
    if (strcmp(type, "P3") != 0) // In this exercise, we don't touch other types
    {
        fclose(f);
        printf("Cannot read %s\n", fileName); 
        exit(EXIT_FAILURE); 
    }

    fscanf(f, "%i", &width);
    fscanf(f, "%i", &height);
    
    int max_val;
    fscanf(f, "%i", &max_val);
    if (max_val > 255) // In this exercise, we assume 1 byte per value
    {
        fclose(f);
        printf("Cannot read %s\n", fileName); 
        exit(EXIT_FAILURE); 
    }

    pixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
    for (int i = 0; i < width * height; i++)
        fscanf(f, "%hhu%hhu%hhu", &pixels[i].x, &pixels[i].y, &pixels[i].z);

    fclose(f);
}

void writePnm(uchar3 *pixels, int width, int height, int originalWidth, char *fileName)
{
    FILE * f = fopen(fileName, "w");
    if (f == NULL)
    {
        printf("Cannot write %s\n", fileName);
        exit(EXIT_FAILURE);
    }   

    fprintf(f, "P3\n%i\n%i\n255\n", width, height); 

    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            int i = r * originalWidth + c;
            fprintf(f, "%hhu\n%hhu\n%hhu\n", pixels[i].x, pixels[i].y, pixels[i].z);
        }
    }
    
    fclose(f);
}

int xSobel[3][3] = {{1,0,-1},{2,0,-2},{1,0,-1}};
int ySobel[3][3] = {{1,2,1},{0,0,0},{-1,-2,-1}};

__constant__ int dc_xSobel[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
__constant__ int dc_ySobel[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

__device__ int d_originalWidth;

__global__ void Rgb2GrayKernel(uchar3 * inPixels, int width, int height, uint8_t * outPixels) {
    size_t r = blockIdx.y * blockDim.y + threadIdx.y;
    size_t c = blockIdx.x * blockDim.x + threadIdx.x;
    size_t i = r * width + c;
    if (r < height && c < width) {
        outPixels[i] = 0.299f * inPixels[i].x
                    + 0.587f * inPixels[i].y
                    + 0.114f * inPixels[i].z;
    }
}

__global__ void pixelsImportantKernel(uint8_t * inPixels, int width, int height, int * importants) {
    int filterWidth = 3;

    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    size_t s_width = blockDim.x + filterWidth - 1;
    size_t s_height = blockDim.y + filterWidth - 1;

    // Each block loads data from GMEM to SMEM
    extern __shared__ uint8_t s_inPixels[];

    int readRow = row - (filterWidth >> 1), readCol, tmpRow, tmpCol;
    int firstReadCol = col - (filterWidth >> 1);
    size_t virtualRow = threadIdx.y, virtualCol;
    for (; virtualRow < s_height; readRow += blockDim.y, virtualRow += blockDim.y) {
        tmpRow = readRow;
        if (readRow < 0) {
            readRow = 0;
        } else if (readRow >= height) {
            readRow = height - 1;
        }
        readCol = firstReadCol;
        virtualCol = threadIdx.x;
        for (; virtualCol < s_width; readCol += blockDim.x, virtualCol += blockDim.x) {
            tmpCol = readCol;
            if (readCol < 0) {
                readCol = 0;
            } else if (readCol >= width) {
                readCol = width - 1;
            }
            s_inPixels[virtualRow * s_width + virtualCol] = inPixels[readRow * d_originalWidth + readCol];
            readCol = tmpCol;
        }
        readRow = tmpRow;
    } 
    __syncthreads();
    // ---------------------------------------

    // Each thread compute importants on SMEM
    int x = 0, y = 0;
    for (int i = 0; i < filterWidth; ++i) {
        for (int j = 0; j < filterWidth; ++j) {
            uint8_t closest = s_inPixels[(threadIdx.y + i) * s_width + threadIdx.x + j];
            size_t filterIdx = i * filterWidth + j;
            x += closest * dc_xSobel[filterIdx];
            y += closest * dc_ySobel[filterIdx];
        }
    }

    // Each thread writes result from SMEM to GMEM
    if (col < width && row < height) {
        size_t idx = row * d_originalWidth + col;
        importants[idx] = abs(x) + abs(y);
    }
}

__global__ void carvingKernel(int *leastSignificantPixel, uchar3 *outPixels, uint8_t *grayPixels, int * importants, int width) {
    int row = blockIdx.x;
    int baseIdx = row * d_originalWidth;
    for (int i = leastSignificantPixel[row]; i < width - 1; ++i) {
        outPixels[baseIdx + i] = outPixels[baseIdx + i + 1];
        grayPixels[baseIdx + i] = grayPixels[baseIdx + i + 1];
        importants[baseIdx + i] = importants[baseIdx + i + 1];
    }
}

void trace(int *score, int *leastSignificantPixel, int width, int height, int originalWidth) {
    int minCol = 0, r = height - 1;
    for (int c = 1; c < width; ++c) {
        if (score[r * originalWidth + c] < score[r * originalWidth + minCol])
            minCol = c;
    }
    for (; r >= 0; --r) {
        leastSignificantPixel[r] = minCol;
        if (r > 0) {
            int aboveIdx = (r - 1) * originalWidth + minCol;
            int min = score[aboveIdx], minColCpy = minCol;
            if (minColCpy > 0 && score[aboveIdx - 1] < min) {
                min = score[aboveIdx - 1];
                minCol = minColCpy - 1;
            }
            if (minColCpy < width - 1 && score[aboveIdx + 1] < min) {
                minCol = minColCpy + 1;
            }
        }
    }
}

__global__ void seamsScoreKernel(int *importants, int *score, int width, int height, int fromRow) {
    size_t halfBlock = blockDim.x >> 1;

    int col = blockIdx.x * halfBlock - halfBlock + threadIdx.x;

    if (fromRow == 0 && col >= 0 && col < width) {
        score[col] = importants[col];
    }
    __syncthreads();

    for (int stride = fromRow != 0 ? 0 : 1; stride < halfBlock && fromRow + stride < height; ++stride) {
        if (threadIdx.x < blockDim.x - (stride << 1)) {
            int curRow = fromRow + stride;
            int curCol = col + stride;

            if (curCol >= 0 && curCol < width) {
                int idx = curRow * d_originalWidth + curCol;
                int aboveIdx = (curRow - 1) * d_originalWidth + curCol;

                int min = score[aboveIdx];
                if (curCol > 0 && score[aboveIdx - 1] < min) {
                    min = score[aboveIdx - 1];
                }
                if (curCol < width - 1 && score[aboveIdx + 1] < min) {
                    min = score[aboveIdx + 1];
                }

                score[idx] = min + importants[idx];
            }
        }
        __syncthreads();
    }
}

void seamCarvingByDevice(uchar3 *inPixels, int width, int height, int targetWidth, uchar3* outPixels, dim3 blockSize) {
    GpuTimer timer;
    timer.Start();

    // allocate kernel memory
    uchar3 *d_inPixels;
    CHECK(cudaMalloc(&d_inPixels, width * height * sizeof(uchar3)));
    uint8_t * d_grayPixels;
    CHECK(cudaMalloc(&d_grayPixels, width * height * sizeof(uint8_t)));
    int * d_importants;
    CHECK(cudaMalloc(&d_importants, width * height * sizeof(int)));
    int * d_leastSignificantPixel;
    CHECK(cudaMalloc(&d_leastSignificantPixel, height * sizeof(int)));
    int * d_score;
    CHECK(cudaMalloc(&d_score, width * height * sizeof(int)));

   
    int * importants = (int *)malloc(width * height * sizeof(int));
    int * leastSignificantPixel = (int *)malloc(height * sizeof(int));
    int * score = (int *)malloc(width * height * sizeof(int));

    // kích thước shared memory
    size_t smemSize = ((blockSize.x + 3 - 1) * (blockSize.y + 3 - 1)) * sizeof(uint8_t);
    
    // block size use to calculate seam score table
    int blockSizeDp = 256;
    int gridSizeDp = (((width - 1) / blockSizeDp + 1) << 1) + 1;
    int stripHeight = (blockSizeDp >> 1) + 1;

    // cache original width
    CHECK(cudaMemcpyToSymbol(d_originalWidth, &width, sizeof(int)));
    const int originalWidth = width;

    // copy input to device
    CHECK(cudaMemcpy(d_inPixels, inPixels, width * height * sizeof(uchar3), cudaMemcpyHostToDevice));

    // Bước 1: Chuyển ảnh RGB về ảnh xám
    dim3 gridSize((width-1)/blockSize.x + 1, (height-1)/blockSize.y + 1);
    Rgb2GrayKernel<<<gridSize, blockSize>>>(d_inPixels, width, height, d_grayPixels);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    // xoá đường seam
    while (width > targetWidth) {
        // Bước 2: Tính độ quan trọng mỗi pixel
        pixelsImportantKernel<<<gridSize, blockSize, smemSize>>>(d_grayPixels, width, height, d_importants);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());

        // Bước 3: Tính bảng seam để tìm seam có giá trị nhỏ nhất
        for (int i = 0; i < height; i += (stripHeight >> 1)) {
            seamsScoreKernel<<<gridSizeDp, blockSizeDp>>>(d_importants, d_score, width, height, i);
            cudaDeviceSynchronize();
            CHECK(cudaGetLastError());
        }

        // Từ pixel có điểm nhỏ nhất dưới cùng, truy vết lên dòng đầu tiên để tìm seam
        CHECK(cudaMemcpy(score, d_score, originalWidth * height * sizeof(int), cudaMemcpyDeviceToHost));
        trace(score, leastSignificantPixel, width, height, originalWidth);

        // Bước 4: Xoá đường seam vừa tìm được
        CHECK(cudaMemcpy(d_leastSignificantPixel, leastSignificantPixel, height * sizeof(int), cudaMemcpyHostToDevice));
        carvingKernel<<<height, 1>>>(d_leastSignificantPixel, d_inPixels, d_grayPixels, d_importants, width);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());
        
        --width;
    }

    CHECK(cudaMemcpy(outPixels, d_inPixels, originalWidth * height * sizeof(uchar3), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_inPixels));
    CHECK(cudaFree(d_grayPixels));
    CHECK(cudaFree(d_importants));
    CHECK(cudaFree(d_leastSignificantPixel));
    CHECK(cudaFree(d_score));

    free(score);
    free(leastSignificantPixel);
    free(importants);

    timer.Stop();
    float time = timer.Elapsed();
    printf("Processing time (use device): %f ms\n\n", time);    
}

uint8_t getClosest(uint8_t *pixels, int r, int c, int width, int height, int originalWidth)
{
    if (r < 0) {
        r = 0;
    } else if (r >= height) {
        r = height - 1;
    }

    if (c < 0) {
        c = 0;
    } else if (c >= width) {
        c = width - 1;
    }

    return pixels[r * originalWidth + c];
}

int pixelsImportant(uint8_t * grayPixels, int row, int col, int width, int height, int originalWidth) {
    int x = 0, y = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            uint8_t closest = getClosest(grayPixels, row - 1 + i, col - 1 + j, width, height, originalWidth);
            x += closest * xSobel[i][j];
            y += closest * ySobel[i][j];
        }
    }
    return abs(x) + abs(y);
}

void RGB2Gray(uchar3 * inPixels, int width, int height, uint8_t * outPixels) {
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            int i = r * width + c;
            outPixels[i] = 0.299f * inPixels[i].x + 0.587f * inPixels[i].y + 0.114f * inPixels[i].z;
        }
    }
}

void seamsScore(int *importants, int *score, int width, int height, int originalWidth) {
    for (int c = 0; c < width; ++c) {
        score[c] = importants[c];
    }
    for (int r = 1; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            int idx = r * originalWidth + c;
            int aboveIdx = (r - 1) * originalWidth + c;

            int min = score[aboveIdx];
            if (c > 0 && score[aboveIdx - 1] < min) {
                min = score[aboveIdx - 1];
            }
            if (c < width - 1 && score[aboveIdx + 1] < min) {
                min = score[aboveIdx + 1];
            }

            score[idx] = min + importants[idx];
        }
    }
}

void seamCarvingByHost(uchar3 *inPixels, int width, int height, int targetWidth, uchar3* outPixels) {
    GpuTimer timer;
    timer.Start();

    memcpy(outPixels, inPixels, width * height * sizeof(uchar3));

    const int originalWidth = width;

    // allocate memory
    int *importants = (int *)malloc(width * height * sizeof(int));
    int *score = (int *)malloc(width * height * sizeof(int));
    uint8_t *grayPixels= (uint8_t *)malloc(width * height * sizeof(uint8_t));
    
    // chuyển về ảnh xám
    RGB2Gray(inPixels, width, height, grayPixels);

    // độ quan trọng pixels
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            importants[r * originalWidth + c] = pixelsImportant(grayPixels, r, c, width, height, width);
        }
    }

    while (width > targetWidth) {
        seamsScore(importants, score, width, height, originalWidth);

        // giá trị seam nhỏ nhất
        int minCol = 0, r = height - 1, prevMinCol;
        for (int c = 1; c < width; ++c) {
            if (score[r * originalWidth + c] < score[r * originalWidth + minCol])
                minCol = c;
        }

        // truy vết và xoá đường seam vừa tìm được
        for (; r >= 0; --r) {
            for (int i = minCol; i < width - 1; ++i) {
                outPixels[r * originalWidth + i] = outPixels[r * originalWidth + i + 1];
                grayPixels[r * originalWidth + i] = grayPixels[r * originalWidth + i + 1];
                importants[r * originalWidth + i] = importants[r * originalWidth + i + 1];
            }

            if (r < height - 1) {
                for (int affectedCol = max(0, prevMinCol - 2); affectedCol <= prevMinCol + 2 && affectedCol < width - 1; ++affectedCol) {
                    importants[(r + 1) * originalWidth + affectedCol] = pixelsImportant(grayPixels, r + 1, affectedCol, width - 1, height, originalWidth);
                }
            }

            if (r > 0) {
                prevMinCol = minCol;

                int aboveIdx = (r - 1) * originalWidth + minCol;
                int min = score[aboveIdx], minColCpy = minCol;
                if (minColCpy > 0 && score[aboveIdx - 1] < min) {
                    min = score[aboveIdx - 1];
                    minCol = minColCpy - 1;
                }
                if (minColCpy < width - 1 && score[aboveIdx + 1] < min) {
                    minCol = minColCpy + 1;
                }
            }
        }

        for (int affectedCol = max(0, minCol - 2); affectedCol <= minCol + 2 && affectedCol < width - 1; ++affectedCol) {
            importants[affectedCol] = pixelsImportant(grayPixels, 0, affectedCol, width - 1, height, originalWidth);
        }

        --width;
    }
    
    free(grayPixels);
    free(score);
    free(importants);

    timer.Stop();
    float time = timer.Elapsed();
    printf("Processing time (use host): %f ms\n\n", time);
}

float computeError(uchar3 * a1, uchar3 * a2, int n)
{
    float err = 0;
    for (int i = 0; i < n; i++)
    {
        err += abs((int)a1[i].x - (int)a2[i].x);
        err += abs((int)a1[i].y - (int)a2[i].y);
        err += abs((int)a1[i].z - (int)a2[i].z);
    }
    err /= (n * 3);
    return err;
}

char *concatStr(const char * s1, const char * s2)
{
    char * result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

void printDeviceInfo()
{
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %lu bytes\n", devProv.totalGlobalMem);
    printf("CMEM: %lu bytes\n", devProv.totalConstMem);
    printf("L2 cache: %i bytes\n", devProv.l2CacheSize);
    printf("SMEM / one SM: %lu bytes\n", devProv.sharedMemPerMultiprocessor);

    printf("****************************\n\n");

}

int main(int argc, char ** argv)
{   
    if (argc != 4 && argc != 6)
    {
        printf("The number of arguments is invalid\n");
        return EXIT_FAILURE;
    }

    printDeviceInfo();

    // Read input RGB image file
    int width, height;
    uchar3 *inPixels;
    readPnm(argv[1], width, height, inPixels);
    printf("Image size (width x height): %i x %i\n\n", width, height);

    int numSeamRemoved = stoi(argv[3]);
    if (numSeamRemoved <= 0 || numSeamRemoved >= width)
        return EXIT_FAILURE; // invalid ratio
    printf("Number of seam removed: %d\n\n", numSeamRemoved);

    int targetWidth = width - numSeamRemoved;

    // seam carving using host
    uchar3 * correctOutPixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
    seamCarvingByHost(inPixels, width, height, targetWidth, correctOutPixels);

    // seam carving using device
    uchar3 * outPixels= (uchar3 *)malloc(width * height * sizeof(uchar3));
    dim3 blockSize(16, 16); // Default
    if (argc == 6)
    {
        blockSize.x = atoi(argv[4]);
        blockSize.y = atoi(argv[5]);
    } 
    seamCarvingByDevice(inPixels, width, height, targetWidth, outPixels, blockSize);
    printf("Image size after seam carving (new_width x height): %i x %i\n\n", targetWidth, height);
    float err = computeError(outPixels, correctOutPixels, width * height);
    printf("Error between device result and host result: %f\n", err);
    
    // Write results to files
    char *outFileNameBase = strtok(argv[2], "."); // Get rid of extension
    writePnm(correctOutPixels, targetWidth, height, width, concatStr(outFileNameBase, "_host.pnm"));
    writePnm(outPixels, targetWidth, height, width, concatStr(outFileNameBase, "_device.pnm"));

    // Free memories
    free(inPixels);
    free(correctOutPixels);
    free(outPixels);
}