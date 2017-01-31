#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <pmmintrin.h>

#include <ssemathfun/sse_mathfun.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

#ifdef _WIN32
#include <malloc.h>
#endif

uint32_t hashUint32(uint32_t input)
{
  input = ~input + (input << 15);
  input = input ^ (input >> 12);
  input = input + (input << 2);
  input = input ^ (input >> 4);
  input = input * 2057;
  input = input ^ (input >> 16);

  return input;
}

float bitsToFloat(uint32_t input)
{
  float output;
  memcpy(&output, &input, sizeof(uint32_t));

  return output;
}

uint32_t floatToBits(float input)
{
  uint32_t output;
  memcpy(&output, &input, sizeof(float));

  return output;
}

float uintToFloat(uint32_t input)
{
  uint32_t output = (0x7F << 23) | (input >> 9);

  return bitsToFloat(output) - 1.f;
}

uint32_t floatToUint(float input)
{
  uint32_t output = floatToBits(input + 1.f);

  return output << 9;
}

uint32_t pseudoRandomUint(uint32_t input, uint32_t scramble = 0U)
{
  input ^= scramble;
  input ^= input >> 17;
  input ^= input >> 10;
  input *= 0xb36534e5;
  input ^= input >> 12;
  input ^= input >> 21;
  input *= 0x93fc4795;
  input ^= 0xdf6e307f;
  input ^= input >> 17;
  input *= 1 | scramble >> 18;

  return input;
}

float pseudoRandomFloat(uint32_t input, uint32_t scramble = 0U)
{
  return uintToFloat(pseudoRandomUint(input, scramble));
}

void* allocAligned(uint32_t size)
{
#ifdef _WIN32
  return _aligned_malloc(size, 16);
#else
  void* output;
  posix_memalign(&output, 16, size);

  return output;
#endif
}

void freeAligned(void* pointer)
{
#ifdef _WIN32
  _aligned_free(pointer);
#else
  free(pointer);
#endif
}

void swap(float* a, float* b)
{
  float temp = *a;

  *a = *b;
  *b = temp;
}

void saveImage(const float* buffer, const char* name, uint32_t size, uint32_t dimension = 0U)
{
  const uint32_t sizeSqr = size * size;
  uint8_t* bitmapData = new uint8_t[sizeSqr];

  for(uint32_t i = 0; i < sizeSqr; ++i)
    bitmapData[i] = buffer[dimension * sizeSqr + i] * 255;

  FILE* file = fopen(name, "wb");

  fprintf(file, "P5 %u %u %u\n", size, size, 255);
  fwrite(bitmapData, 1, sizeSqr, file);

  fclose(file);

  delete[] bitmapData;
}

void saveData(const float* buffer, const char* name, uint32_t size, uint32_t depth)
{
  const uint32_t sizeSqr = size * size;
  uint32_t* outputData = new uint32_t[sizeSqr * depth];

  for(uint32_t i = 0; i < sizeSqr; ++i)
    for(uint32_t j = 0; j < depth; ++j)
      outputData[i * depth + j] = floatToUint(buffer[j * sizeSqr + i]);

  FILE* file = fopen(name, "w");

  fprintf(file, "Width: %u Height: %u Depth: %u Interval: 0 %u\n\n", size, size, depth, UINT32_MAX);
  for(uint32_t i = 0; i < sizeSqr * depth; ++i)
    fprintf(file, "%u\n", outputData[i]);

  fclose(file);

  delete[] outputData;
}

struct SimulationData
{
  uint32_t m;
  uint32_t depth;
  uint32_t iterations;
  uint32_t seed;
  float sigmaI;
  float sigmaS;
};

class SimulationSum
{
public:
  float m_result;

  SimulationSum(const float* buffer, const SimulationData& sData) : m_buffer(buffer), m_sData(sData), m_result(0.f) {}
  SimulationSum(SimulationSum& x, tbb::split) : m_buffer(x.m_buffer), m_sData(x.m_sData), m_result(0.f) {}

  void operator()(const tbb::blocked_range< uint32_t >& r)
  {
    const float* buffer = m_buffer;

    const uint32_t depth = m_sData.depth;
    const uint32_t size = 1 << m_sData.m;
    const uint32_t sizeSqr = size * size;

    const float sizeOverTwo = size * 0.5f;
    const float depthOverTwo = depth * 0.5f;
    const float sigmaISqr = m_sData.sigmaI * m_sData.sigmaI;
    const float sigmaSSqr = m_sData.sigmaS * m_sData.sigmaS;

    float result = m_result;

    static const __m128 signmask = _mm_castsi128_ps(_mm_set1_epi32(1 << 31));
    static const __m128 offsetf = _mm_setr_ps(0.f, 1.f, 2.f, 3.f);
    static const __m128i offseti = _mm_setr_epi32(0, 1, 2, 3);

    for(uint32_t i = r.begin(); i < r.end(); ++i) 
    {
      const __m128i iv = _mm_set1_epi32(i);
      const __m128 ix = _mm_set1_ps(i % size);
      const __m128 iy = _mm_set1_ps(i / size);

      for(uint32_t j = 0; j < sizeSqr; j += 4)
      {
        const __m128i jv = _mm_add_epi32(_mm_set1_epi32(j), offseti);
        const __m128 jx = _mm_add_ps(_mm_set1_ps(j % size), offsetf);
        const __m128 jy = _mm_set1_ps(j / size);

        __m128 imageDistX = _mm_andnot_ps(signmask, _mm_sub_ps(ix, jx));
        __m128 imageDistY = _mm_andnot_ps(signmask, _mm_sub_ps(iy, jy));

        const __m128 imageWrapX = _mm_sub_ps(_mm_set1_ps(size), imageDistX);
        const __m128 imageWrapY = _mm_sub_ps(_mm_set1_ps(size), imageDistY);

        const __m128 maskX = _mm_cmplt_ps(imageDistX, _mm_set1_ps(sizeOverTwo));
        const __m128 maskY = _mm_cmplt_ps(imageDistY, _mm_set1_ps(sizeOverTwo));

        imageDistX = _mm_or_ps(_mm_and_ps(maskX, imageDistX), _mm_andnot_ps(maskX, imageWrapX));
        imageDistY = _mm_or_ps(_mm_and_ps(maskY, imageDistY), _mm_andnot_ps(maskY, imageWrapY));

        const __m128 imageDistSqrX = _mm_mul_ps(imageDistX, imageDistX);
        const __m128 imageDistSqrY = _mm_mul_ps(imageDistY, imageDistY);

        const __m128 imageSqr = _mm_add_ps(imageDistSqrX, imageDistSqrY);
        const __m128 imageEnergy = _mm_div_ps(imageSqr, _mm_set1_ps(sigmaISqr));

        __m128 sampleSqr = _mm_setzero_ps();

        for(uint32_t k = 0; k < depth; ++k)
        {
          const __m128 pBuffer = _mm_set1_ps(buffer[k * sizeSqr + i]);
          const __m128 qBuffer = _mm_load_ps(&buffer[k * sizeSqr + j]);
          const __m128 sampleDistance = _mm_andnot_ps(signmask, _mm_sub_ps(pBuffer, qBuffer));
          const __m128 sampleDistanceSqr = _mm_mul_ps(sampleDistance, sampleDistance);

          sampleSqr = _mm_add_ps(sampleSqr, sampleDistanceSqr);
        }

        const __m128 samplePow = exp_ps(_mm_mul_ps(log_ps(_mm_sqrt_ps(sampleSqr)), _mm_set1_ps(depthOverTwo)));
        const __m128 sampleEnergy = _mm_div_ps(samplePow, _mm_set1_ps(sigmaSSqr));

        const __m128 output = exp_ps(_mm_sub_ps(_mm_sub_ps(_mm_setzero_ps(), imageEnergy), sampleEnergy));
        const __m128 mask = _mm_castsi128_ps(_mm_cmpeq_epi32(iv, jv)); 

        __m128 masked = _mm_andnot_ps(mask, output);
        masked = _mm_hadd_ps(masked, masked);
        masked = _mm_hadd_ps(masked, masked);

        result += _mm_cvtss_f32(masked);
      }
    }

    m_result = result;
  }

  void join(const SimulationSum& y)
  {
    m_result += y.m_result;
  }

private:
  const float* m_buffer;
  const SimulationData& m_sData;

};

float E(const float* buffer, const SimulationData& sData)
{
  const uint32_t size = 1 << sData.m;
  const uint32_t sizeSqr = size * size;

  SimulationSum sum(buffer, sData);
  tbb::parallel_deterministic_reduce(tbb::blocked_range< uint32_t >(0, sizeSqr), sum);
  
  return sum.m_result;
}

void fourierTransform1D(const float* inReal, const float* inImag, float* outReal, float* outImag, uint32_t size)
{
  const float invSize = 1.f / size;

  for(uint32_t i = 0; i < size; ++i)
  {
    const float constant = 2.f * M_PI * i * invSize;

    float sumReal = 0.f;
    float sumImag = 0.f;

    for(uint32_t j = 0; j < size; ++j)
    {
      const float cosConstant = cos(j * constant);
      const float sinConstant = sin(j * constant);

      sumReal +=  inReal[j] * cosConstant + inImag[j] * sinConstant;
      sumImag += -inReal[j] * sinConstant + inImag[j] * cosConstant;
    }

    outReal[i] = sumReal * invSize;
    outImag[i] = sumImag * invSize;
  }
}

void fourierTransform2D(const float* inReal, float* output, uint32_t size, uint32_t dimension = 0U)
{
  const uint32_t sizeSqr = size * size;

  float* realTemp1 = new float[sizeSqr];
  float* imagTemp1 = new float[sizeSqr];
  float* realTemp2 = new float[sizeSqr];
  float* imagTemp2 = new float[sizeSqr];

  for(uint32_t i = 0; i < sizeSqr; ++i)
  {
    realTemp1[i] = inReal[dimension * sizeSqr + i] * pow(-1.f, (i % size) + (i / size));
    imagTemp1[i] = 0.f;
  }

  for(uint32_t i = 0; i < size; ++i)
  {
    const uint32_t index = i * size;
    fourierTransform1D(&realTemp1[index], &imagTemp1[index], &realTemp2[index], &imagTemp2[index], size);
  }

  for(uint32_t i = 0; i < sizeSqr; ++i)
  {
    realTemp1[i] = realTemp2[(i % size) * size + (i / size)];
    imagTemp1[i] = imagTemp2[(i % size) * size + (i / size)];
  }

  for(uint32_t i = 0; i < size; ++i)
  {
    const uint32_t index = i * size;
    fourierTransform1D(&realTemp1[index], &imagTemp1[index], &realTemp2[index], &imagTemp2[index], size);
  }

  for(uint32_t i = 0; i < sizeSqr; ++i)
    output[i] = log(sqrt(realTemp2[i] * realTemp2[i] + imagTemp2[i] * imagTemp2[i]) + 1.f);

  delete[] realTemp1;
  delete[] imagTemp1;
  delete[] realTemp2;
  delete[] imagTemp2;
}

int main(int argc, char const *argv[])
{
  SimulationData sData;

  sData.m = 5;
  sData.depth = 1;
  sData.iterations = 4096;
  sData.seed = 0;
  sData.sigmaI = 2.1f;
  sData.sigmaS = 1.f;

  assert(sData.m > 1);

  const uint32_t size = 1 << sData.m;
  const uint32_t sizeSqr = size * size;

  float* whiteNoiseBuffer = (float*) allocAligned(sizeof(float) * sData.depth * sizeSqr);
  float* blueNoiseBuffer = (float*) allocAligned(sizeof(float) * sData.depth * sizeSqr);
  float* proposalBuffer = (float*) allocAligned(sizeof(float) * sData.depth * sizeSqr);
  float* fourierBuffer = (float*) allocAligned(sizeof(float) * sizeSqr);

  uint32_t seedHash = hashUint32(sData.seed);

  for(uint32_t i = 0; i < sData.depth; ++i)
  {
    uint32_t depthHash = hashUint32(i);

    for(uint32_t j = 0; j < sizeSqr; ++j)
    {
      float pseudoRandomValue = pseudoRandomFloat(j, seedHash ^ depthHash);

      whiteNoiseBuffer[i * sizeSqr + j] = pseudoRandomValue;
      blueNoiseBuffer[i * sizeSqr + j] = pseudoRandomValue;
      proposalBuffer[i * sizeSqr + j] = pseudoRandomValue;
    }
  }

  tbb::task_scheduler_init scheduler;

  float initialDistrib = E(whiteNoiseBuffer, sData);

  float whiteNoiseDistrib = initialDistrib;
  float blueNoiseDistrib = initialDistrib;

  for(uint32_t i = 0; i < sData.iterations; ++i)
  {
    uint32_t u1 = pseudoRandomFloat(i * 4 + 0) * size;
    uint32_t u2 = pseudoRandomFloat(i * 4 + 1) * size;
    uint32_t u3 = pseudoRandomFloat(i * 4 + 2) * size;
    uint32_t u4 = pseudoRandomFloat(i * 4 + 3) * size;

    uint32_t p1 = u1 * size + u2;
    uint32_t p2 = u3 * size + u4;

    for(uint32_t j = 0; j < sData.depth; ++j)
      swap(&proposalBuffer[j * sizeSqr + p1], &proposalBuffer[j * sizeSqr + p2]);

    float proposalDistrib = E(proposalBuffer, sData);

    if(proposalDistrib > blueNoiseDistrib)
    {
      for(uint32_t j = 0; j < sData.depth; ++j)
      {
        proposalBuffer[j * sizeSqr + p1] = blueNoiseBuffer[j * sizeSqr + p1];
        proposalBuffer[j * sizeSqr + p2] = blueNoiseBuffer[j * sizeSqr + p2];
      }

      continue;
    }

    blueNoiseDistrib = proposalDistrib;

    for(uint32_t j = 0; j < sData.depth; ++j)
    {
      blueNoiseBuffer[j * sizeSqr + p1] = proposalBuffer[j * sizeSqr + p1];
      blueNoiseBuffer[j * sizeSqr + p2] = proposalBuffer[j * sizeSqr + p2];
    }
  }

  saveData(blueNoiseBuffer, "outputBlueNoise.txt", size, sData.depth);

  saveImage(whiteNoiseBuffer, "outputWhiteNoise.pgm", size);
  saveImage(blueNoiseBuffer, "outputBlueNoise.pgm", size);

  fourierTransform2D(whiteNoiseBuffer, fourierBuffer, size);

  for(uint32_t i = 0; i < sizeSqr; ++i)
    fourierBuffer[i] *= 50.f;
  
  saveImage(fourierBuffer, "outputFourierWhite.pgm", size);

  fourierTransform2D(blueNoiseBuffer, fourierBuffer, size);

  for(uint32_t i = 0; i < sizeSqr; ++i)
    fourierBuffer[i] *= 50.f;
  
  saveImage(fourierBuffer, "outputFourierBlue.pgm", size);

  freeAligned(whiteNoiseBuffer);
  freeAligned(blueNoiseBuffer);
  freeAligned(proposalBuffer);
  freeAligned(fourierBuffer);

  return 0;
}