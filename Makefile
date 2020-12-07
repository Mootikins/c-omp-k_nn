debug: src/knn.c
	gcc -Wall -Wextra -pedantic -g $^ -o $@ -fopenmp -lm -DDEBUG

knn: src/knn.c
	gcc -Wall $^ -o $@ -O2 -fopenmp -lm

clean:
	@rm knn debug
