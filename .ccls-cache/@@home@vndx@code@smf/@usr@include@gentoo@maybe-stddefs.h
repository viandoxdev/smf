/* __has_include is an extension, but it's fine, because this is only
for Clang anyway. */
#if defined __has_include && __has_include (<stdc-predef.h>) && !defined(__GLIBC__)
# include <stdc-predef.h>
#endif
