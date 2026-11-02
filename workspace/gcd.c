#include <stdio.h>

/* 返回 a 和 b 的最大公约数 */
int gcd(int a, int b) {
    /* TODO: to be implemented */
    return a % b;
}

int main(void) {
    printf("gcd(36,48)=%d\n", gcd(36, 48));
    return 0;
}
