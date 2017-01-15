#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

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

  for(int32_t ix = 0; ix < size; ++ix)
  {
    for(int32_t iy = 0; iy < size; ++iy)
    {
      uint32_t i = ix * size + iy;

      for(int32_t jx = 0; jx < size; ++jx)
      {
        for(int32_t jy = 0; jy < size; ++jy)
        {
          uint32_t j = jx * size + jy;

          uint32_t imageDistanceX = abs(ix - jx);
          uint32_t imageDistanceY = abs(iy - jy);

          if(imageDistanceX > sizeOverTwo)
            imageDistanceX = size - imageDistanceX;

          if(imageDistanceY > sizeOverTwo)
            imageDistanceY = size - imageDistanceY;

          uint32_t imageDistanceSqrX = imageDistanceX * imageDistanceX;
          uint32_t imageDistanceSqrY = imageDistanceY * imageDistanceY;

          float imageSqr = imageDistanceSqrX + imageDistanceSqrY;

          float sampleSqr = 0.f;

          for(uint32_t k = 0; k < depth; ++k)
          {
            float sampleDistance = fabs(buffer[k * pixelCount + i] - buffer[k * pixelCount + j]);
            float sampleDistanceSqr = sampleDistance * sampleDistance;

            sampleSqr += sampleDistanceSqr;
          }

          float imageEnergy = imageSqr / sigmaISqr;
          float sampleEnergy = pow(sqrt(sampleSqr), depthOverTwo) / sigmaSSqr;

          float output = exp(-imageEnergy - sampleEnergy);
          bool mask = !(i == j);

          energySum += output * mask;
        }
      }
    }
  }

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

  float* whiteNoiseBuffer = new float[arraySize];
  float* blueNoiseBuffer = new float[arraySize];
  float* proposalBuffer = new float[arraySize];

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

  delete[] whiteNoiseBuffer;
  delete[] blueNoiseBuffer;
  delete[] proposalBuffer;

  return 0;
}