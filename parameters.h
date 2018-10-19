#define debug 0
#define scaleForRandom 100
#define sizeOfPopulation 32
#define sizeOfChromosome 3
#define blocksPerGrid 32
#define threadsPerBlock 32
#define probabilityOfMutation 0.08
#define penalty 0
#define checkEach 1
#define REVERSIBLEFITNESS 1
#define verbose 0
#define probabilityOfCrossover 0.8
#define numberOfWires 2
#define mutationRange 2
#define window 5
#define PI 3.14159265358979323846
#define tol 0.001
#define X 0
#define Y 1
#define Z 2
#include <cuComplex.h>
typedef struct Gates
{
	int index1;
	int index2;
	int direction;
	cuDoubleComplex parameter;
} Gate;
