#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>


#define NUM_BANKS 64
#define LOG_NUM_BANKS 5
// Lab4: You can use any other block size you wish.
#define BLOCK_SIZE 128 // This MUST be a power of 2
#define MAX(a, b) (((a) > (b)) ? (a) : (b)) // Used to instantiate grid dimensions

// Lab4: Host Helper Functions (allocate your own data structure...)

// Found this bad boy from StackOverflow.
// Rounds numToRound upward to the nearest multiple. 
int roundUp(int numToRound, int multiple)
{
    assert(multiple);
    return ((numToRound + multiple - 1) / multiple) * multiple;
}

// Calculate the number of intermediate steps we will need to reduce this input
int numIntermediateSteps(int numElements)
{
    int numSteps = 1;
    int bSize = BLOCK_SIZE;
    while (bSize < numElements)
    {
        numSteps++;
        bSize *= BLOCK_SIZE;
    }
    return numSteps;
}

// Calculate the array of offsets for where intermediate values will begin to be stored
int *getOffsets(int numElements)
{
    int numIntermediates = numIntermediateSteps(numElements);
    int *offsets = new int[numIntermediates];
    for (int i = 0; i < numIntermediates; i++)
    {
        offsets[i] = roundUp(numElements, BLOCK_SIZE);
        numElements /= BLOCK_SIZE;
    }
    return offsets;
}

float* allocateIntermediatesOnDevice(int numElements)
{
    /* For our parallel reduction step, we will store log_{BLOCK_SIZE}(numElements)
       intermediate arrays. This is necessary because only the bottommost step (that fits
       in a single block) can perform the entire end-to-end exclusive scan operation to get
       the offsets that must bubble upwards. The topmost intermediate will point towards
       the input array, which has been allocated to the device prior to this function 
       call. */
    // Find sum of offsets to calculate how many floats we must allocate
    int numIntermediates = numIntermediateSteps(numElements); 
    int* offsets = getOffsets(numElements);
    int totalElements = 0; 
    for(int i=0; i<numIntermediates; i++)
        totalElements += offsets[i];

    // Allocate space on the device, set it to 0 originally
    float *intermediates;
    cudaMalloc((void**) &intermediates, totalElements*sizeof(float));
    cudaMemset((void*) intermediates, 0, totalElements*sizeof(float));
    return intermediates;
}

// Frees all intermediate arrays allocated on the device.
void freeIntermediates(float *intermediates)
{
    cudaFree(intermediates);
}


// Lab4: Device Functions
__device__ int roundUpDev(int numToRound, int multiple)
{
    assert(multiple);
    return ((numToRound + multiple - 1) / multiple) * multiple;
}

// Lab4: Kernel Functions

/* This Kernel Performs the Reduction Scan step of the parallel prefix scan.
    -----------------------------------------Parameters:-----------------------------------------------
    float* inArray : the inputs which should be partially scanned blockwise
    float* resArray : the sum of each scanned block; needed to bubble up offsets
    int n : the number of elements in inArray
    int offset : the offset from which this invocation of the kernel should start reading from resArray
    */
__global__ void Reduction_Kernel(float *inArray, float *resArray, int n, int offset) 
{
    /* STEP 1: Declare shared memory and copy over your assigned element */
    __shared__ float temp[BLOCK_SIZE];
    int localId = threadIdx.x;
    int globalId = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (globalId < n) // On first pass, n will be the size of inArray INSTEAD of the full offset (rounded to BLOCK_SIZE)
    {
        if (offset > 0) temp[localId] = resArray[globalId + offset];
        else temp[localId] = inArray[globalId];
    }
    else temp[localId] = 0.0f;

    /* STEP 2: Reduce the block to perform a partial prefix scan */
    int stride = 1;
    for (int d = BLOCK_SIZE >> 1; d > 0; d >>= 1)
    {
        // if (localId == 0) printf("%d\t", d);
        __syncthreads();
        if (localId < d)
        {
            int left = stride * (2 * localId + 1) - 1;
            int right = stride * (2 * localId + 2) - 1;

            temp[right] += temp[left]; // TODO: Can we reduce bank conflicts here?
        }
        stride *= 2;
    }

    /* STEP 3: Store the overall block sum in resArray prior to zeroing it */
    __syncthreads(); // Might not be needed here since only one thread does work in the last step
    if (localId == 0) 
    {
        // Use roundUpDev to get the full offset to where we need to start writing. 
        // Really doesn't do anything beyond the first pass, but is necessary because it's an efficient way
        // to ensure we don't have illegal memory accesses on the first pass when we have to read from inArray
        resArray[offset + roundUpDev(n, BLOCK_SIZE) + blockIdx.x] = temp[BLOCK_SIZE - 1];
        temp[BLOCK_SIZE - 1] = 0; // Clear the last element for exclusive scan
    }

    /* STEP 4: Perform the post-scan step  */
    for (int d = 1; d < BLOCK_SIZE; d *= 2)
    {
        // if (localId == 0) printf("%d\t", d);
        stride >>= 1;
        __syncthreads();

        if (localId < d)
        {
            int left = stride * (2 * localId + 1) - 1;
            int right = stride * (2 * localId + 2) - 1;

            // Swap left with right before we update
            float t = temp[left];
            temp[left] = temp[right];
            temp[right] += t;
        }
    }

    /* STEP 5: Update your assigned element to complete the partial prefix scan over this block */
    __syncthreads();
    resArray[globalId + offset] = temp[localId];
}

/* This kernel performs an in-place parallel prefix scan of the input. This kernel is intended
       to be called once the input is capable of fitting in a single block.
       --------------------------------Parameters:---------------------------------
       float* inArray : the inputs which need to be scanned. Its elements will be edited in-place.
       int offset : the offset by which we should begin reading from inArray. 

       NOTE: We don't need to pass in the number of elements in the input, because this kernel operates
             as if this is the only block. As such, we can just use BLOCK_SIZE.
    */
__global__ void FullScan_Kernel(float *inArray, int offset) 
{
    
    /* STEP 1: Allocate shared memory and copy over your assigned element of the input */
    __shared__ float temp[BLOCK_SIZE]; // Create enough shared mem to contain the block
    int tid = threadIdx.x; // Local/Global ID of this thread since this is the only block
    temp[tid] = inArray[offset + tid];
    
    /* STEP 2: Perform the reduction step */
    int stride = 1;
    for (int d = BLOCK_SIZE>>1; d > 0; d >>= 1) 
    {
        __syncthreads();
        if (tid < d)
        {
            int left = stride * (2 * tid + 1)-1;
            int right = stride * (2 * tid + 2)-1;
            
            temp[right] += temp[left]; // TODO: Can we reduce bank conflicts here?
        }
        stride *= 2;
    }

    /* STEP 3: Clear the last element for an exclusive scan */
    if (tid == 0) temp[BLOCK_SIZE - 1] = 0; // Clear the last element for exclusive scan

    /* STEP 4: Perform the post-scan step */
    for (int d = 1; d < BLOCK_SIZE; d *= 2)
    {
        stride >>= 1;
        __syncthreads();

        if (tid < d) 
        {
            int left = stride * (2 * tid + 1) - 1;
            int right = stride * (2 * tid + 2) - 1;

            // Swap left with right before we update
            float t = temp[left];
            temp[left] = temp[right];
            temp[right] += t;
        }
    }

    /* STEP 5: Copy your element back to the input array (in place update of the scan) */
    __syncthreads();
    inArray[tid + offset] = temp[tid]; 
}

/* Same as FullScan_Kernel, but we might want to translate inputs straignt to outputs if the 
   entire input can fit in a single block. Ideally, I would get the code to work without writing
   a separate kernel for this edge case, but this is the quickest and dirtiest fix. 
*/
__global__ void FullScan_Kernel_InToOut(float *inArray, float* outArray, int n)
{
    /* STEP 1: Allocate shared memory and copy over your assigned element of the input */
    __shared__ float temp[BLOCK_SIZE]; // Create enough shared mem to contain the block
    int tid = threadIdx.x;             // Local/Global ID of this thread since this is the only block
    if (tid < n) temp[tid] = inArray[tid];
    else temp[tid] = 0.0f;

    /* STEP 2: Perform the reduction step */
    int stride = 1;
    for (int d = BLOCK_SIZE >> 1; d > 0; d >>= 1)
    {
        __syncthreads();
        if (tid < d)
        {
            int left = stride * (2 * tid + 1) - 1;
            int right = stride * (2 * tid + 2) - 1;

            temp[right] += temp[left]; // TODO: Can we reduce bank conflicts here?
        }
        stride *= 2;
    }

    /* STEP 3: Clear the last element for an exclusive scan */
    if (tid == 0)
        temp[BLOCK_SIZE - 1] = 0; // Clear the last element for exclusive scan

    /* STEP 4: Perform the post-scan step */
    for (int d = 1; d < BLOCK_SIZE; d *= 2)
    {
        stride >>= 1;
        __syncthreads();

        if (tid < d)
        {
            int left = stride * (2 * tid + 1) - 1;
            int right = stride * (2 * tid + 2) - 1;

            // Swap left with right before we update
            float t = temp[left];
            temp[left] = temp[right];
            temp[right] += t;
        }
    }

    /* STEP 5: Copy your element to the output array */
    __syncthreads();
    if (tid < n) outArray[tid] = temp[tid];
}

__global__ void AddOffset_Kernel(float *outArray, float *resArray, int n, int offset)
{
    /* This kernel bubbles up offset calculations from each intermediate array.
    --------------------------------Parameters:---------------------------------
    float* outArray : final outputted array expected by the host. This will be NULL for any
                      intermediate steps, but valid for the final step so that the prefix
                      scan results can be stored where the host expects them.
    float* resArray : intermediate array who needs offsets to be bubbled up to it
    int n : the number of elements in inArray
    int offset : the offset from which this invocation of the kernel should start reading from resArray
    */
    int globalId = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (outArray == NULL) resArray[globalId + offset] += resArray[offset + n + blockIdx.x];
    else if (globalId < n) outArray[globalId] = resArray[globalId] + resArray[roundUpDev(n, BLOCK_SIZE) + blockIdx.x];
}

// **===-------- Lab4: Modify the body of this function -----------===**
// You may need to make multiple kernel calls, make your own kernel
// function in this file, and then call them from here.
void prescanArray(float *outArray, float *inArray, float *intermediates, int numElements)
{
    /* STEP 1: Reduce the input array results until they can finally fit in a single block */
    // TODO: A lot of functions use the logic of numIntermediates, can I unify this somewhere?
    int* offsets = getOffsets(numElements);
    int currOffset = 0;
    int numIntermediates = numIntermediateSteps(numElements);
    for (int i = 0; i < numIntermediates - 1; i++) // For numIntermediates - 1 iterations
    {
        dim3 gridDims(offsets[i] / BLOCK_SIZE, 1, 1);
        dim3 blockDims(BLOCK_SIZE, 1, 1);
        Reduction_Kernel<<<gridDims, blockDims>>>(inArray, intermediates, offsets[i], currOffset);
        cudaDeviceSynchronize();
        currOffset += offsets[i];

        /*
        // Error detection, delete this later
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
        {
            printf("Error in Reduction_Kernel: %s\n", cudaGetErrorString(err));
            freeIntermediates(intermediates);
            return;
        }
        */
    }

    /* STEP 2: Once the results fit in a single block, perform a full prefix scan on that block */
    dim3 gridDims(1, 1, 1);
    dim3 blockDims(BLOCK_SIZE, 1, 1);

    // NOTE: If the entire input fits in a single block, STEPS 2 and 4 won't execute any kernel
    //       invocations. The quickest way to catch these cases was to write a separate kernel
    //       that does the full scan straight from the input array to the output array.
    if (numIntermediates == 1) FullScan_Kernel_InToOut<<<gridDims, blockDims>>>(inArray, outArray, numElements);
    else FullScan_Kernel<<<gridDims, blockDims>>>(intermediates, currOffset);
    cudaDeviceSynchronize();

    /*
    // Error detection, delete this later
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Error in FullScan_Kernel: %s\n", cudaGetErrorString(err));
        freeIntermediates(intermediates);
        return;
    }
    */

    /* STEP 3: Bubble up offsets to get the final prefix scan of the original array */
    if (numIntermediates > 1) // There is no bubbling if the input fits in one block
    {
        int i = 2;
        do
        {
            currOffset -= offsets[numIntermediates - i];

            /*
            printf("%d\t", numIntermediates);
            printf("%d\t", numIntermediates - i);
            printf("%d\t", currOffset);
            printf("%d\t", offsets[numIntermediates - 1]);
            printf("%d\n", offsets[numIntermediates - 2]);
            */

            dim3 gridDims(MAX(offsets[numIntermediates - i] / BLOCK_SIZE, 1), 1, 1);
            dim3 blockDims(BLOCK_SIZE, 1, 1);
            if (currOffset == 0)
                AddOffset_Kernel<<<gridDims, blockDims>>>(outArray, intermediates,
                                                          offsets[numIntermediates - i], currOffset);
            else
                AddOffset_Kernel<<<gridDims, blockDims>>>(NULL, intermediates,
                                                          offsets[numIntermediates - i], currOffset);

            cudaDeviceSynchronize();
            i += 1;

            /*
            // DELETE THIS ONCE IT IS WORKING WITH NO ERRORS
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                printf("Error in AddOffset_Kernel: %s\n", cudaGetErrorString(err));
                freeIntermediates(intermediates);
                return;
            }
            */
        } while (currOffset > 0);
    }
}
// **===-----------------------------------------------------------===**


#endif // _PRESCAN_CU_
