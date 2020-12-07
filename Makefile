debug: src/knn.c
	gcc -Wall -Wextra -pedantic -g $^ -o $@ -lm -DDEBUG

knn: src/knn.c
	gcc -Wall $^ -o $@ -lm

clean:
	@rm knn debug
