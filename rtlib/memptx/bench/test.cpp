#include <stdio.h>
#include <stdlib.h>

#define LOOP_CNT 1000

#define BLOCK_SIZE (4<<10)

static int lc = LOOP_CNT;

class P
{
	public:
		P(){};
		~P(){};
		char buf[BLOCK_SIZE];
};

class P *p;

int main(int argc, char** argv)
{

	if (argc==2)
		lc = atol(argv[1]);

	for (int i=0;i<lc;i++)
	{
		p = new P();
		delete p;
	}
	return 0;
}

