all:
	g++ -std=c++14 -march=native -fopenmp -Ofast -fno-finite-math-only main.cpp -o TD_CNTFET -lblas -lgomp -lsuperlu
