#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <pmmintrin.h>

#include <ssemathfun/sse_mathfun.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

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

float uintToFloat(uint32_t input)
{
  uint32_t output = (0x7F << 23) | (input >> 0x9);

  return bitsToFloat(output) - 1.f;
}

uint32_t pseudoRandomUint(uint32_t input, uint32_t scramble = 0x0)
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

float pseudoRandomFloat(uint32_t input, uint32_t scramble = 0x0)
{
  return uintToFloat(pseudoRandomUint(input, scramble));
}

void* allocAligned(uint32_t size)
{
  void* output;
  posix_memalign(&output, 16, size);

  return output;
}

void swap(float* a, float* b)
{
  float temp = *a;

  *a = *b;
  *b = temp;
}

void saveImage(const float* buffer, const char* name, const uint32_t size, const uint32_t dimension)
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

struct SimulationData
{
  uint32_t size;
  uint32_t depth;
  uint32_t iterations;
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
    const SimulationData sData = m_sData;

    const float sizeOverTwo = sData.size * 0.5f;
    const float depthOverTwo = sData.depth * 0.5f;
    const float sigmaISqr = sData.sigmaI * sData.sigmaI;
    const float sigmaSSqr = sData.sigmaS * sData.sigmaS;

    const uint32_t sizeSqr = sData.size * sData.size;

    float result = m_result;

    static const __m128 signmask = _mm_castsi128_ps(_mm_set1_epi32(1 << 31));
    static const __m128 offsetf = _mm_setr_ps(0.f, 1.f, 2.f, 3.f);
    static const __m128i offseti = _mm_setr_epi32(0, 1, 2, 3);

    for(uint32_t i = r.begin(); i < r.end(); ++i) 
    {
      const __m128i iv = _mm_set1_epi32(i);
      const __m128 ix = _mm_set1_ps(i % sData.size);
      const __m128 iy = _mm_set1_ps(i / sData.size);

      for(uint32_t j = 0; j < sizeSqr; j += 4)
      {
        const __m128i jv = _mm_add_epi32(_mm_set1_epi32(j), offseti);
        const __m128 jx = _mm_add_ps(_mm_set1_ps(j % sData.size), offsetf);
        const __m128 jy = _mm_set1_ps(j / sData.size);

        __m128 imageDistX = _mm_andnot_ps(signmask, _mm_sub_ps(ix, jx));
        __m128 imageDistY = _mm_andnot_ps(signmask, _mm_sub_ps(iy, jy));

        const __m128 imageWrapX = _mm_sub_ps(_mm_set1_ps(sData.size), imageDistX);
        const __m128 imageWrapY = _mm_sub_ps(_mm_set1_ps(sData.size), imageDistY);

        const __m128 maskX = _mm_cmplt_ps(imageDistX, _mm_set1_ps(sizeOverTwo));
        const __m128 maskY = _mm_cmplt_ps(imageDistY, _mm_set1_ps(sizeOverTwo));

        imageDistX = _mm_or_ps(_mm_and_ps(maskX, imageDistX), _mm_andnot_ps(maskX, imageWrapX));
        imageDistY = _mm_or_ps(_mm_and_ps(maskY, imageDistY), _mm_andnot_ps(maskY, imageWrapY));

        const __m128 imageDistSqrX = _mm_mul_ps(imageDistX, imageDistX);
        const __m128 imageDistSqrY = _mm_mul_ps(imageDistY, imageDistY);

        const __m128 imageSqr = _mm_add_ps(imageDistSqrX, imageDistSqrY);
        const __m128 imageEnergy = _mm_div_ps(imageSqr, _mm_set1_ps(sigmaISqr));

        __m128 sampleSqr = _mm_setzero_ps();

        for(uint32_t k = 0; k < sData.depth; ++k)
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

        __m128 masked = _mm_andnot_si128(mask, output);
        masked = _mm_hadd_ps(masked, masked);
        masked = _mm_hadd_ps(masked, masked);

        result += _mm_cvtss_f32(masked);
      }
    }

    // for(uint32_t i = r.begin(); i < r.end(); ++i)
    // {
    //   const float ix = i % sData.size;
    //   const float iy = i / sData.size;

    //   for(uint32_t j = 0; j < sizeSqr; ++j)
    //   {
    //     if(i == j)
    //       continue;

    //     const float jx = j % sData.size;
    //     const float jy = j / sData.size;

    //     float imageDistX = fabs(ix - jx);
    //     float imageDistY = fabs(iy - jy);

    //     if(imageDistX > sizeOverTwo)
    //       imageDistX = sData.size - imageDistX;

    //     if(imageDistY > sizeOverTwo)
    //       imageDistY = sData.size - imageDistY;

    //     float imageDistSqrX = imageDistX * imageDistX;
    //     float imageDistSqrY = imageDistY * imageDistY;

    //     float imageSqr = imageDistSqrX + imageDistSqrY;
    //     float imageEnergy = imageSqr / sigmaISqr;

    //     float sampleSqr = 0.f;

    //     for(uint32_t k = 0; k < sData.depth; ++k)
    //     {
    //       float sampleDistance = fabs(buffer[k * sizeSqr + i] - buffer[k * sizeSqr + j]);
    //       float sampleDistanceSqr = sampleDistance * sampleDistance;

    //       sampleSqr += sampleDistanceSqr;
    //     }

    //     float samplePow = pow(sqrt(sampleSqr), depthOverTwo);
    //     float sampleEnergy = samplePow / sigmaSSqr;

    //     float output = exp(-imageEnergy - sampleEnergy);

    //     result += output;
    //   }
    // }

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

float E(float* buffer, const SimulationData& sData)
{
  SimulationSum sum(buffer, sData);
  tbb::parallel_deterministic_reduce(tbb::blocked_range< uint32_t >(0, sData.size * sData.size), sum);
  
  return sum.m_result;
}

int main(int argc, char const *argv[])
{
  SimulationData sData;

  sData.size = 32;
  sData.depth = 1;
  sData.iterations = 4096;
  sData.sigmaI = 2.1f;
  sData.sigmaS = 1.f;

  const uint32_t sizeSqr = sData.size * sData.size;

  float* whiteNoiseBuffer = (float*) allocAligned(sizeof(float) * sData.depth * sizeSqr);
  float* blueNoiseBuffer = (float*) allocAligned(sizeof(float) * sData.depth * sizeSqr);
  float* proposalBuffer = (float*) allocAligned(sizeof(float) * sData.depth * sizeSqr);

  for(uint32_t i = 0; i < sData.depth; ++i)
  {
    uint32_t hash = hashUint32(i);

    for(uint32_t j = 0; j < sizeSqr; ++j)
    {
      float pseudoRandomValue = pseudoRandomFloat(j, hash);

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
    uint32_t u1 = pseudoRandomFloat(i * 4 + 0) * sData.size;
    uint32_t u2 = pseudoRandomFloat(i * 4 + 1) * sData.size;
    uint32_t u3 = pseudoRandomFloat(i * 4 + 2) * sData.size;
    uint32_t u4 = pseudoRandomFloat(i * 4 + 3) * sData.size;

    uint32_t p1 = u1 * sData.size + u2;
    uint32_t p2 = u3 * sData.size + u4;

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

  const char whiteNoiseImage[] = "outputWhiteNoise.pgm";
  const char blueNoiseImage[] = "outputBlueNoise.pgm";

  saveImage(whiteNoiseBuffer, whiteNoiseImage, sData.size, 0);
  saveImage(blueNoiseBuffer, blueNoiseImage, sData.size, 0);

  printf("white noise result: %f\n", whiteNoiseDistrib);
  printf("blue noise result: %f\n", blueNoiseDistrib);

  free(whiteNoiseBuffer);
  free(blueNoiseBuffer);
  free(proposalBuffer);

  return 0;
}