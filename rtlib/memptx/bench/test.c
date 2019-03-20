#include <stdio.h>
#include <stdlib.h>

#define LOOP_CNT 1000

#define BLOCK_SIZE (4<<10)

static int lc = LOOP_CNT;

void *p;

int main(int argc, char** argv)
{

	if (argc==2)
		lc = atol(argv[1]);

	for (int i=0;i<lc;i++)
	{
		p = malloc(sizeof(char)*BLOCK_SIZE);
		free(p);
	}
	return 0;
}

