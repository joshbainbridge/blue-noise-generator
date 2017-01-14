#include <stdint.h>
#include <string.h>
#include <stdio.h>

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

uint32_t pseudoRandomUint(uint32_t input, uint32_t scramble)
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

template<typename T>
void swap(T* a, T* b)
{
  T temp = *a;

  *a = *b;
  *b = temp;
}

int main(int argc, char const *argv[])
{
  const uint32_t depth = 1;
  const uint32_t res = 256;
  const uint32_t resSqr = res * res;

  const uint32_t arraySize = resSqr * depth;

  float* whiteNoiseBuffer = new float[arraySize];
  float* blueNoiseBuffer = new float[arraySize];
  float* proposalBuffer = new float[arraySize];

  for(uint32_t i = 0; i < depth; ++i)
  {
    uint32_t hash = hashUint32(i);

    for(uint32_t j = 0; j < resSqr; ++j)
    {
      float pseudoRandomValue = pseudoRandomFloat(j, hash);

      whiteNoiseBuffer[i * resSqr + j] = pseudoRandomValue;
      blueNoiseBuffer[i * resSqr + j] = pseudoRandomValue;
      proposalBuffer[i * resSqr + j] = pseudoRandomValue;
    }
  }

  uint8_t* bitmapData = new uint8_t[arraySize];

  for(uint32_t i = 0; i < resSqr; ++i)
    bitmapData[i] = blueNoiseBuffer[0 * resSqr + i] * 255;

  FILE* file = fopen("output.pgm", "wb");

  fprintf(file, "P5 %u %u %u\n", res, res, 255);
  fwrite(bitmapData, 1, resSqr, file);

  fclose(file);

  delete[] whiteNoiseBuffer;
  delete[] blueNoiseBuffer;
  delete[] proposalBuffer;

  delete[] bitmapData;

  return 0;
}