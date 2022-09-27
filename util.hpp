#ifndef __UTIL_H
#define __UTIL_H

#define TENSOR_IDX(i, j, k, n) \
    ((n) * ((n) * (i) + (j)) + (k))

#endif