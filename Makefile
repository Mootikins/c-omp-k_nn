debug: src/knn.c
	gcc -Wall -g $^ -o $@ -DDEBUG

knn: src/knn.c
	gcc -Wall $^ -o $@

clean:
	@rm knn debug
