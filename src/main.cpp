#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include <smmintrin.h>
#include <sse_mathfun.h>

template<typename T>
void swap(T& a, T& b)
{
  T temp = a;

  a = b;
  b = temp;
}

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

void* allocAligned(size_t size)
{
  void* output;
  posix_memalign(&output, 16, size);

  return output;
}

void saveImage(const float* buffer, const char* name, const uint32_t size, const uint32_t dimension)
{
  const uint32_t pixelCount = size * size;

  uint8_t* bitmapData = new uint8_t[pixelCount];

  for(uint32_t i = 0; i < pixelCount; ++i)
    bitmapData[i] = buffer[dimension * pixelCount + i] * 255;

  FILE* file = fopen(name, "wb");

  fprintf(file, "P5 %u %u %u\n", size, size, 255);
  fwrite(bitmapData, 1, pixelCount, file);

  fclose(file);

  delete[] bitmapData;
}

float E(const float* buffer, const uint32_t size, const uint32_t depth, const float sigmaI, const float sigmaS)
{
  const float sigmaISqr = sigmaI * sigmaI;
  const float sigmaSSqr = sigmaS * sigmaS;
  const float depthOverTwo = depth * 0.5f;
  const uint32_t sizeOverTwo = size / 2;
  const uint32_t pixelCount = size * size;

  float energySum = 0.f;
  
  static const __m128 signmask = _mm_castsi128_ps(_mm_set1_epi32(1 << 31));
  static const __m128 offsetf = _mm_setr_ps(0.f, 1.f, 2.f, 3.f);
  static const __m128i offseti = _mm_setr_epi32(0, 1, 2, 3);

  for(uint32_t i = 0; i < pixelCount; ++i)
  {
    const __m128i iv = _mm_set1_epi32(i);
    const __m128 ix = _mm_set1_ps(i % size);
    const __m128 iy = _mm_set1_ps(i / size);

    for(uint32_t j = 0; j < pixelCount; j += 4)
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
        const __m128 pBuffer = _mm_set1_ps(buffer[k * pixelCount + i]);
        const __m128 qBuffer = _mm_load_ps(&buffer[k * pixelCount + j]);
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

      energySum += _mm_cvtss_f32(masked);
    }
  }

  // for(uint32_t i = 0; i < pixelCount; ++i)
  // {
  //   const float ix = i % size;
  //   const float iy = i / size;

  //   for(uint32_t j = 0; j < pixelCount; ++j)
  //   {
  //     if(i == j)
  //       continue;

  //     const float jx = j % size;
  //     const float jy = j / size;

  //     float imageDistX = fabs(ix - jx);
  //     float imageDistY = fabs(iy - jy);

  //     if(imageDistX > sizeOverTwo)
  //       imageDistX = size - imageDistX;

  //     if(imageDistY > sizeOverTwo)
  //       imageDistY = size - imageDistY;

  //     float imageDistSqrX = imageDistX * imageDistX;
  //     float imageDistSqrY = imageDistY * imageDistY;

  //     float imageSqr = imageDistSqrX + imageDistSqrY;
  //     float imageEnergy = imageSqr / sigmaISqr;

  //     float sampleSqr = 0.f;

  //     for(uint32_t k = 0; k < depth; ++k)
  //     {
  //       float sampleDistance = fabs(buffer[k * pixelCount + i] - buffer[k * pixelCount + j]);
  //       float sampleDistanceSqr = sampleDistance * sampleDistance;

  //       sampleSqr += sampleDistanceSqr;
  //     }

  //     float samplePow = pow(sqrt(sampleSqr), depthOverTwo);
  //     float sampleEnergy = samplePow / sigmaSSqr;

  //     float output = exp(-imageEnergy - sampleEnergy);

  //     energySum += output;
  //   }
  // }

  return energySum;
}

int main(int argc, char const *argv[])
{
  const uint32_t depth = 1;
  const uint32_t resolution = 32;
  const uint32_t iterations = 4096;
  const float sigmaI = 2.1f;
  const float sigmaS = 1.f;

  const uint32_t pixelCount = resolution * resolution;
  const uint32_t arraySize = pixelCount * depth;

  float* whiteNoiseBuffer = (float*) allocAligned(sizeof(float) * arraySize);
  float* blueNoiseBuffer = (float*) allocAligned(sizeof(float) * arraySize);
  float* proposalBuffer = (float*) allocAligned(sizeof(float) * arraySize);

  for(uint32_t i = 0; i < depth; ++i)
  {
    uint32_t hash = hashUint32(i);

    for(uint32_t j = 0; j < pixelCount; ++j)
    {
      float pseudoRandomValue = pseudoRandomFloat(j, hash);

      whiteNoiseBuffer[i * pixelCount + j] = pseudoRandomValue;
      blueNoiseBuffer[i * pixelCount + j] = pseudoRandomValue;
      proposalBuffer[i * pixelCount + j] = pseudoRandomValue;
    }
  }

  float startingDistrib = E(whiteNoiseBuffer, resolution, depth, sigmaI, sigmaS);

  float whiteNoiseDistrib = startingDistrib;
  float blueNoiseDistrib = startingDistrib;

  for(uint32_t i = 0; i < iterations; ++i)
  {
    uint32_t u1 = pseudoRandomFloat(i * 4 + 0) * resolution;
    uint32_t u2 = pseudoRandomFloat(i * 4 + 1) * resolution;
    uint32_t u3 = pseudoRandomFloat(i * 4 + 2) * resolution;
    uint32_t u4 = pseudoRandomFloat(i * 4 + 3) * resolution;

    uint32_t p1 = u1 * resolution + u2;
    uint32_t p2 = u3 * resolution + u4;

    for(uint32_t j = 0; j < depth; ++j)
      swap(proposalBuffer[j * pixelCount + p1], proposalBuffer[j * pixelCount + p2]);

    float proposalDistrib = E(proposalBuffer, resolution, depth, sigmaI, sigmaS);

    if(proposalDistrib > blueNoiseDistrib)
    {
      for(uint32_t j = 0; j < depth; ++j)
      {
        proposalBuffer[j * pixelCount + p1] = blueNoiseBuffer[j * pixelCount + p1];
        proposalBuffer[j * pixelCount + p2] = blueNoiseBuffer[j * pixelCount + p2];
      }

      continue;
    }

    blueNoiseDistrib = proposalDistrib;

    for(uint32_t j = 0; j < depth; ++j)
    {
      blueNoiseBuffer[j * pixelCount + p1] = proposalBuffer[j * pixelCount + p1];
      blueNoiseBuffer[j * pixelCount + p2] = proposalBuffer[j * pixelCount + p2];
    }
  }

  const char blueNoiseImage[] = "outputBlueNoise.pgm";
  const char whiteNoiseImage[] = "outputWhiteNoise.pgm";

  saveImage(blueNoiseBuffer, blueNoiseImage, resolution, 0);
  saveImage(whiteNoiseBuffer, whiteNoiseImage, resolution, 0);

  printf("blue noise result: %f\n", blueNoiseDistrib);
  printf("white noise result: %f\n", whiteNoiseDistrib);

  free(whiteNoiseBuffer);
  free(blueNoiseBuffer);
  free(proposalBuffer);

  return 0;
}