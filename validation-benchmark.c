#pragma once

/*
 * Headers
*/

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>


/*
 * Initialisation
*/

struct futhark_context_config ;
struct futhark_context_config *futhark_context_config_new();
void futhark_context_config_free(struct futhark_context_config *cfg);
void futhark_context_config_set_debugging(struct futhark_context_config *cfg,
                                          int flag);
void futhark_context_config_set_logging(struct futhark_context_config *cfg,
                                        int flag);
struct futhark_context ;
struct futhark_context *futhark_context_new(struct futhark_context_config *cfg);
void futhark_context_free(struct futhark_context *ctx);
int futhark_context_sync(struct futhark_context *ctx);
char *futhark_context_get_error(struct futhark_context *ctx);

/*
 * Arrays
*/

struct futhark_f32_1d ;
struct futhark_f32_1d *futhark_new_f32_1d(struct futhark_context *ctx,
                                          float *data, int64_t dim0);
struct futhark_f32_1d *futhark_new_raw_f32_1d(struct futhark_context *ctx,
                                              char *data, int offset,
                                              int64_t dim0);
int futhark_free_f32_1d(struct futhark_context *ctx,
                        struct futhark_f32_1d *arr);
int futhark_values_f32_1d(struct futhark_context *ctx,
                          struct futhark_f32_1d *arr, float *data);
char *futhark_values_raw_f32_1d(struct futhark_context *ctx,
                                struct futhark_f32_1d *arr);
int64_t *futhark_shape_f32_1d(struct futhark_context *ctx,
                              struct futhark_f32_1d *arr);
struct futhark_f32_2d ;
struct futhark_f32_2d *futhark_new_f32_2d(struct futhark_context *ctx,
                                          float *data, int64_t dim0,
                                          int64_t dim1);
struct futhark_f32_2d *futhark_new_raw_f32_2d(struct futhark_context *ctx,
                                              char *data, int offset,
                                              int64_t dim0, int64_t dim1);
int futhark_free_f32_2d(struct futhark_context *ctx,
                        struct futhark_f32_2d *arr);
int futhark_values_f32_2d(struct futhark_context *ctx,
                          struct futhark_f32_2d *arr, float *data);
char *futhark_values_raw_f32_2d(struct futhark_context *ctx,
                                struct futhark_f32_2d *arr);
int64_t *futhark_shape_f32_2d(struct futhark_context *ctx,
                              struct futhark_f32_2d *arr);
struct futhark_f32_3d ;
struct futhark_f32_3d *futhark_new_f32_3d(struct futhark_context *ctx,
                                          float *data, int64_t dim0,
                                          int64_t dim1, int64_t dim2);
struct futhark_f32_3d *futhark_new_raw_f32_3d(struct futhark_context *ctx,
                                              char *data, int offset,
                                              int64_t dim0, int64_t dim1,
                                              int64_t dim2);
int futhark_free_f32_3d(struct futhark_context *ctx,
                        struct futhark_f32_3d *arr);
int futhark_values_f32_3d(struct futhark_context *ctx,
                          struct futhark_f32_3d *arr, float *data);
char *futhark_values_raw_f32_3d(struct futhark_context *ctx,
                                struct futhark_f32_3d *arr);
int64_t *futhark_shape_f32_3d(struct futhark_context *ctx,
                              struct futhark_f32_3d *arr);
struct futhark_i32_1d ;
struct futhark_i32_1d *futhark_new_i32_1d(struct futhark_context *ctx,
                                          int32_t *data, int64_t dim0);
struct futhark_i32_1d *futhark_new_raw_i32_1d(struct futhark_context *ctx,
                                              char *data, int offset,
                                              int64_t dim0);
int futhark_free_i32_1d(struct futhark_context *ctx,
                        struct futhark_i32_1d *arr);
int futhark_values_i32_1d(struct futhark_context *ctx,
                          struct futhark_i32_1d *arr, int32_t *data);
char *futhark_values_raw_i32_1d(struct futhark_context *ctx,
                                struct futhark_i32_1d *arr);
int64_t *futhark_shape_i32_1d(struct futhark_context *ctx,
                              struct futhark_i32_1d *arr);
struct futhark_i32_2d ;
struct futhark_i32_2d *futhark_new_i32_2d(struct futhark_context *ctx,
                                          int32_t *data, int64_t dim0,
                                          int64_t dim1);
struct futhark_i32_2d *futhark_new_raw_i32_2d(struct futhark_context *ctx,
                                              char *data, int offset,
                                              int64_t dim0, int64_t dim1);
int futhark_free_i32_2d(struct futhark_context *ctx,
                        struct futhark_i32_2d *arr);
int futhark_values_i32_2d(struct futhark_context *ctx,
                          struct futhark_i32_2d *arr, int32_t *data);
char *futhark_values_raw_i32_2d(struct futhark_context *ctx,
                                struct futhark_i32_2d *arr);
int64_t *futhark_shape_i32_2d(struct futhark_context *ctx,
                              struct futhark_i32_2d *arr);

/*
 * Opaque values
*/


/*
 * Entry points
*/

int futhark_entry_main(struct futhark_context *ctx, bool *out0, bool *out1,
                       bool *out2, bool *out3, bool *out4, bool *out5,
                       bool *out6, bool *out7, bool *out8, bool *out9,
                       bool *out10, bool *out11, bool *out12, bool *out13,
                       bool *out14, bool *out15, bool *out16, const
                       struct futhark_f32_2d *in0, const
                       struct futhark_f32_3d *in1, const
                       struct futhark_f32_3d *in2, const
                       struct futhark_f32_2d *in3, const
                       struct futhark_f32_2d *in4, const
                       struct futhark_f32_2d *in5, const
                       struct futhark_i32_1d *in6, const
                       struct futhark_f32_2d *in7, const
                       struct futhark_i32_2d *in8, const
                       struct futhark_i32_1d *in9, const
                       struct futhark_i32_1d *in10, const
                       struct futhark_f32_1d *in11, const
                       struct futhark_f32_1d *in12, const
                       struct futhark_f32_2d *in13, const
                       struct futhark_f32_2d *in14, const
                       struct futhark_i32_1d *in15, const
                       struct futhark_f32_1d *in16, const
                       struct futhark_f32_2d *in17, const
                       struct futhark_f32_3d *in18, const
                       struct futhark_f32_3d *in19, const
                       struct futhark_f32_2d *in20, const
                       struct futhark_f32_2d *in21, const
                       struct futhark_f32_2d *in22, const
                       struct futhark_i32_1d *in23, const
                       struct futhark_f32_2d *in24, const
                       struct futhark_i32_2d *in25, const
                       struct futhark_i32_1d *in26, const
                       struct futhark_i32_1d *in27, const
                       struct futhark_f32_1d *in28, const
                       struct futhark_f32_1d *in29, const
                       struct futhark_f32_2d *in30, const
                       struct futhark_f32_2d *in31, const
                       struct futhark_i32_1d *in32, const
                       struct futhark_f32_1d *in33);

/*
 * Miscellaneous
*/

void futhark_debugging_report(struct futhark_context *ctx);
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <stdint.h>
#undef NDEBUG
#include <assert.h>
// Start of panic.h.

#include <stdarg.h>

static const char *fut_progname;

static void panic(int eval, const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
        fprintf(stderr, "%s: ", fut_progname);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
        exit(eval);
}

/* For generating arbitrary-sized error messages.  It is the callers
   responsibility to free the buffer at some point. */
static char* msgprintf(const char *s, ...) {
  va_list vl;
  va_start(vl, s);
  size_t needed = 1 + vsnprintf(NULL, 0, s, vl);
  char *buffer = (char*) malloc(needed);
  va_start(vl, s); /* Must re-init. */
  vsnprintf(buffer, needed, s, vl);
  return buffer;
}

// End of panic.h.

// Start of timing.h.

// The function get_wall_time() returns the wall time in microseconds
// (with an unspecified offset).

#ifdef _WIN32

#include <windows.h>

static int64_t get_wall_time(void) {
  LARGE_INTEGER time,freq;
  assert(QueryPerformanceFrequency(&freq));
  assert(QueryPerformanceCounter(&time));
  return ((double)time.QuadPart / freq.QuadPart) * 1000000;
}

#else
/* Assuming POSIX */

#include <time.h>
#include <sys/time.h>

static int64_t get_wall_time(void) {
  struct timeval time;
  assert(gettimeofday(&time,NULL) == 0);
  return time.tv_sec * 1000000 + time.tv_usec;
}

#endif

// End of timing.h.

#include <string.h>
#include <inttypes.h>
#include <errno.h>
#include <ctype.h>
#include <errno.h>
#include <getopt.h>
// Start of values.h.

//// Text I/O

typedef int (*writer)(FILE*, void*);
typedef int (*bin_reader)(void*);
typedef int (*str_reader)(const char *, void*);

struct array_reader {
  char* elems;
  int64_t n_elems_space;
  int64_t elem_size;
  int64_t n_elems_used;
  int64_t *shape;
  str_reader elem_reader;
};

static void skipspaces() {
  int c;
  do {
    c = getchar();
  } while (isspace(c));

  if (c != EOF) {
    ungetc(c, stdin);
  }
}

static int constituent(char c) {
  return isalnum(c) || c == '.' || c == '-' || c == '+' || c == '_';
}

// Produces an empty token only on EOF.
static void next_token(char *buf, int bufsize) {
 start:
  skipspaces();

  int i = 0;
  while (i < bufsize) {
    int c = getchar();
    buf[i] = c;

    if (c == EOF) {
      buf[i] = 0;
      return;
    } else if (c == '-' && i == 1 && buf[0] == '-') {
      // Line comment, so skip to end of line and start over.
      for (; c != '\n' && c != EOF; c = getchar());
      goto start;
    } else if (!constituent(c)) {
      if (i == 0) {
        // We permit single-character tokens that are not
        // constituents; this lets things like ']' and ',' be
        // tokens.
        buf[i+1] = 0;
        return;
      } else {
        ungetc(c, stdin);
        buf[i] = 0;
        return;
      }
    }

    i++;
  }

  buf[bufsize-1] = 0;
}

static int next_token_is(char *buf, int bufsize, const char* expected) {
  next_token(buf, bufsize);
  return strcmp(buf, expected) == 0;
}

static void remove_underscores(char *buf) {
  char *w = buf;

  for (char *r = buf; *r; r++) {
    if (*r != '_') {
      *w++ = *r;
    }
  }

  *w++ = 0;
}

static int read_str_elem(char *buf, struct array_reader *reader) {
  int ret;
  if (reader->n_elems_used == reader->n_elems_space) {
    reader->n_elems_space *= 2;
    reader->elems = (char*) realloc(reader->elems,
                                    reader->n_elems_space * reader->elem_size);
  }

  ret = reader->elem_reader(buf, reader->elems + reader->n_elems_used * reader->elem_size);

  if (ret == 0) {
    reader->n_elems_used++;
  }

  return ret;
}

static int read_str_array_elems(char *buf, int bufsize,
                                struct array_reader *reader, int dims) {
  int ret;
  int first = 1;
  char *knows_dimsize = (char*) calloc(dims,sizeof(char));
  int cur_dim = dims-1;
  int64_t *elems_read_in_dim = (int64_t*) calloc(dims,sizeof(int64_t));

  while (1) {
    next_token(buf, bufsize);

    if (strcmp(buf, "]") == 0) {
      if (knows_dimsize[cur_dim]) {
        if (reader->shape[cur_dim] != elems_read_in_dim[cur_dim]) {
          ret = 1;
          break;
        }
      } else {
        knows_dimsize[cur_dim] = 1;
        reader->shape[cur_dim] = elems_read_in_dim[cur_dim];
      }
      if (cur_dim == 0) {
        ret = 0;
        break;
      } else {
        cur_dim--;
        elems_read_in_dim[cur_dim]++;
      }
    } else if (strcmp(buf, ",") == 0) {
      next_token(buf, bufsize);
      if (strcmp(buf, "[") == 0) {
        if (cur_dim == dims - 1) {
          ret = 1;
          break;
        }
        first = 1;
        cur_dim++;
        elems_read_in_dim[cur_dim] = 0;
      } else if (cur_dim == dims - 1) {
        ret = read_str_elem(buf, reader);
        if (ret != 0) {
          break;
        }
        elems_read_in_dim[cur_dim]++;
      } else {
        ret = 1;
        break;
      }
    } else if (strlen(buf) == 0) {
      // EOF
      ret = 1;
      break;
    } else if (first) {
      if (strcmp(buf, "[") == 0) {
        if (cur_dim == dims - 1) {
          ret = 1;
          break;
        }
        cur_dim++;
        elems_read_in_dim[cur_dim] = 0;
      } else {
        ret = read_str_elem(buf, reader);
        if (ret != 0) {
          break;
        }
        elems_read_in_dim[cur_dim]++;
        first = 0;
      }
    } else {
      ret = 1;
      break;
    }
  }

  free(knows_dimsize);
  free(elems_read_in_dim);
  return ret;
}

static int read_str_empty_array(char *buf, int bufsize,
                                const char *type_name, int64_t *shape, int64_t dims) {
  if (strlen(buf) == 0) {
    // EOF
    return 1;
  }

  if (strcmp(buf, "empty") != 0) {
    return 1;
  }

  if (!next_token_is(buf, bufsize, "(")) {
    return 1;
  }

  for (int i = 0; i < dims-1; i++) {
    if (!next_token_is(buf, bufsize, "[")) {
      return 1;
    }

    if (!next_token_is(buf, bufsize, "]")) {
      return 1;
    }
  }

  if (!next_token_is(buf, bufsize, type_name)) {
    return 1;
  }


  if (!next_token_is(buf, bufsize, ")")) {
    return 1;
  }

  for (int i = 0; i < dims; i++) {
    shape[i] = 0;
  }

  return 0;
}

static int read_str_array(int64_t elem_size, str_reader elem_reader,
                          const char *type_name,
                          void **data, int64_t *shape, int64_t dims) {
  int ret;
  struct array_reader reader;
  char buf[100];

  int dims_seen;
  for (dims_seen = 0; dims_seen < dims; dims_seen++) {
    if (!next_token_is(buf, sizeof(buf), "[")) {
      break;
    }
  }

  if (dims_seen == 0) {
    return read_str_empty_array(buf, sizeof(buf), type_name, shape, dims);
  }

  if (dims_seen != dims) {
    return 1;
  }

  reader.shape = shape;
  reader.n_elems_used = 0;
  reader.elem_size = elem_size;
  reader.n_elems_space = 16;
  reader.elems = (char*) realloc(*data, elem_size*reader.n_elems_space);
  reader.elem_reader = elem_reader;

  ret = read_str_array_elems(buf, sizeof(buf), &reader, dims);

  *data = reader.elems;

  return ret;
}

#define READ_STR(MACRO, PTR, SUFFIX)                                   \
  remove_underscores(buf);                                              \
  int j;                                                                \
  if (sscanf(buf, "%"MACRO"%n", (PTR*)dest, &j) == 1) {                 \
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, SUFFIX) == 0);     \
  } else {                                                              \
    return 1;                                                           \
  }

static int read_str_i8(char *buf, void* dest) {
  /* Some platforms (WINDOWS) does not support scanf %hhd or its
     cousin, %SCNi8.  Read into int first to avoid corrupting
     memory.

     https://gcc.gnu.org/bugzilla/show_bug.cgi?id=63417  */
  remove_underscores(buf);
  int j, x;
  if (sscanf(buf, "%i%n", &x, &j) == 1) {
    *(int8_t*)dest = x;
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, "i8") == 0);
  } else {
    return 1;
  }
}

static int read_str_u8(char *buf, void* dest) {
  /* Some platforms (WINDOWS) does not support scanf %hhd or its
     cousin, %SCNu8.  Read into int first to avoid corrupting
     memory.

     https://gcc.gnu.org/bugzilla/show_bug.cgi?id=63417  */
  remove_underscores(buf);
  int j, x;
  if (sscanf(buf, "%i%n", &x, &j) == 1) {
    *(uint8_t*)dest = x;
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, "u8") == 0);
  } else {
    return 1;
  }
}

static int read_str_i16(char *buf, void* dest) {
  READ_STR(SCNi16, int16_t, "i16");
}

static int read_str_u16(char *buf, void* dest) {
  READ_STR(SCNi16, int16_t, "u16");
}

static int read_str_i32(char *buf, void* dest) {
  READ_STR(SCNi32, int32_t, "i32");
}

static int read_str_u32(char *buf, void* dest) {
  READ_STR(SCNi32, int32_t, "u32");
}

static int read_str_i64(char *buf, void* dest) {
  READ_STR(SCNi64, int64_t, "i64");
}

static int read_str_u64(char *buf, void* dest) {
  // FIXME: This is not correct, as SCNu64 only permits decimal
  // literals.  However, SCNi64 does not handle very large numbers
  // correctly (it's really for signed numbers, so that's fair).
  READ_STR(SCNu64, uint64_t, "u64");
}

static int read_str_f32(char *buf, void* dest) {
  remove_underscores(buf);
  if (strcmp(buf, "f32.nan") == 0) {
    *(float*)dest = NAN;
    return 0;
  } else if (strcmp(buf, "f32.inf") == 0) {
    *(float*)dest = INFINITY;
    return 0;
  } else if (strcmp(buf, "-f32.inf") == 0) {
    *(float*)dest = -INFINITY;
    return 0;
  } else {
    READ_STR("f", float, "f32");
  }
}

static int read_str_f64(char *buf, void* dest) {
  remove_underscores(buf);
  if (strcmp(buf, "f64.nan") == 0) {
    *(double*)dest = NAN;
    return 0;
  } else if (strcmp(buf, "f64.inf") == 0) {
    *(double*)dest = INFINITY;
    return 0;
  } else if (strcmp(buf, "-f64.inf") == 0) {
    *(double*)dest = -INFINITY;
    return 0;
  } else {
    READ_STR("lf", double, "f64");
  }
}

static int read_str_bool(char *buf, void* dest) {
  if (strcmp(buf, "true") == 0) {
    *(char*)dest = 1;
    return 0;
  } else if (strcmp(buf, "false") == 0) {
    *(char*)dest = 0;
    return 0;
  } else {
    return 1;
  }
}

static int write_str_i8(FILE *out, int8_t *src) {
  return fprintf(out, "%hhdi8", *src);
}

static int write_str_u8(FILE *out, uint8_t *src) {
  return fprintf(out, "%hhuu8", *src);
}

static int write_str_i16(FILE *out, int16_t *src) {
  return fprintf(out, "%hdi16", *src);
}

static int write_str_u16(FILE *out, uint16_t *src) {
  return fprintf(out, "%huu16", *src);
}

static int write_str_i32(FILE *out, int32_t *src) {
  return fprintf(out, "%di32", *src);
}

static int write_str_u32(FILE *out, uint32_t *src) {
  return fprintf(out, "%uu32", *src);
}

static int write_str_i64(FILE *out, int64_t *src) {
  return fprintf(out, "%"PRIi64"i64", *src);
}

static int write_str_u64(FILE *out, uint64_t *src) {
  return fprintf(out, "%"PRIu64"u64", *src);
}

static int write_str_f32(FILE *out, float *src) {
  float x = *src;
  if (isnan(x)) {
    return fprintf(out, "f32.nan");
  } else if (isinf(x) && x >= 0) {
    return fprintf(out, "f32.inf");
  } else if (isinf(x)) {
    return fprintf(out, "-f32.inf");
  } else {
    return fprintf(out, "%.6ff32", x);
  }
}

static int write_str_f64(FILE *out, double *src) {
  double x = *src;
  if (isnan(x)) {
    return fprintf(out, "f64.nan");
  } else if (isinf(x) && x >= 0) {
    return fprintf(out, "f64.inf");
  } else if (isinf(x)) {
    return fprintf(out, "-f64.inf");
  } else {
    return fprintf(out, "%.6ff64", *src);
  }
}

static int write_str_bool(FILE *out, void *src) {
  return fprintf(out, *(char*)src ? "true" : "false");
}

//// Binary I/O

#define BINARY_FORMAT_VERSION 2
#define IS_BIG_ENDIAN (!*(unsigned char *)&(uint16_t){1})

// On Windows we need to explicitly set the file mode to not mangle
// newline characters.  On *nix there is no difference.
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
static void set_binary_mode(FILE *f) {
  setmode(fileno(f), O_BINARY);
}
#else
static void set_binary_mode(FILE *f) {
  (void)f;
}
#endif

// Reading little-endian byte sequences.  On big-endian hosts, we flip
// the resulting bytes.

static int read_byte(void* dest) {
  int num_elems_read = fread(dest, 1, 1, stdin);
  return num_elems_read == 1 ? 0 : 1;
}

static int read_le_2byte(void* dest) {
  uint16_t x;
  int num_elems_read = fread(&x, 2, 1, stdin);
  if (IS_BIG_ENDIAN) {
    x = (x>>8) | (x<<8);
  }
  *(uint16_t*)dest = x;
  return num_elems_read == 1 ? 0 : 1;
}

static int read_le_4byte(void* dest) {
  uint32_t x;
  int num_elems_read = fread(&x, 4, 1, stdin);
  if (IS_BIG_ENDIAN) {
    x =
      ((x>>24)&0xFF) |
      ((x>>8) &0xFF00) |
      ((x<<8) &0xFF0000) |
      ((x<<24)&0xFF000000);
  }
  *(uint32_t*)dest = x;
  return num_elems_read == 1 ? 0 : 1;
}

static int read_le_8byte(void* dest) {
  uint64_t x;
  int num_elems_read = fread(&x, 8, 1, stdin);
  if (IS_BIG_ENDIAN) {
    x =
      ((x>>56)&0xFFull) |
      ((x>>40)&0xFF00ull) |
      ((x>>24)&0xFF0000ull) |
      ((x>>8) &0xFF000000ull) |
      ((x<<8) &0xFF00000000ull) |
      ((x<<24)&0xFF0000000000ull) |
      ((x<<40)&0xFF000000000000ull) |
      ((x<<56)&0xFF00000000000000ull);
  }
  *(uint64_t*)dest = x;
  return num_elems_read == 1 ? 0 : 1;
}

static int write_byte(void* dest) {
  int num_elems_written = fwrite(dest, 1, 1, stdin);
  return num_elems_written == 1 ? 0 : 1;
}

static int write_le_2byte(void* dest) {
  uint16_t x = *(uint16_t*)dest;
  if (IS_BIG_ENDIAN) {
    x = (x>>8) | (x<<8);
  }
  int num_elems_written = fwrite(&x, 2, 1, stdin);
  return num_elems_written == 1 ? 0 : 1;
}

static int write_le_4byte(void* dest) {
  uint32_t x = *(uint32_t*)dest;
  if (IS_BIG_ENDIAN) {
    x =
      ((x>>24)&0xFF) |
      ((x>>8) &0xFF00) |
      ((x<<8) &0xFF0000) |
      ((x<<24)&0xFF000000);
  }
  int num_elems_written = fwrite(&x, 4, 1, stdin);
  return num_elems_written == 1 ? 0 : 1;
}

static int write_le_8byte(void* dest) {
  uint64_t x = *(uint64_t*)dest;
  if (IS_BIG_ENDIAN) {
    x =
      ((x>>56)&0xFFull) |
      ((x>>40)&0xFF00ull) |
      ((x>>24)&0xFF0000ull) |
      ((x>>8) &0xFF000000ull) |
      ((x<<8) &0xFF00000000ull) |
      ((x<<24)&0xFF0000000000ull) |
      ((x<<40)&0xFF000000000000ull) |
      ((x<<56)&0xFF00000000000000ull);
  }
  int num_elems_written = fwrite(&x, 8, 1, stdin);
  return num_elems_written == 1 ? 0 : 1;
}

//// Types

struct primtype_info_t {
  const char binname[4]; // Used for parsing binary data.
  const char* type_name; // Same name as in Futhark.
  const int size; // in bytes
  const writer write_str; // Write in text format.
  const str_reader read_str; // Read in text format.
  const writer write_bin; // Write in binary format.
  const bin_reader read_bin; // Read in binary format.
};

static const struct primtype_info_t i8_info =
  {.binname = "  i8", .type_name = "i8",   .size = 1,
   .write_str = (writer)write_str_i8, .read_str = (str_reader)read_str_i8,
   .write_bin = (writer)write_byte, .read_bin = (bin_reader)read_byte};
static const struct primtype_info_t i16_info =
  {.binname = " i16", .type_name = "i16",  .size = 2,
   .write_str = (writer)write_str_i16, .read_str = (str_reader)read_str_i16,
   .write_bin = (writer)write_le_2byte, .read_bin = (bin_reader)read_le_2byte};
static const struct primtype_info_t i32_info =
  {.binname = " i32", .type_name = "i32",  .size = 4,
   .write_str = (writer)write_str_i32, .read_str = (str_reader)read_str_i32,
   .write_bin = (writer)write_le_4byte, .read_bin = (bin_reader)read_le_4byte};
static const struct primtype_info_t i64_info =
  {.binname = " i64", .type_name = "i64",  .size = 8,
   .write_str = (writer)write_str_i64, .read_str = (str_reader)read_str_i64,
   .write_bin = (writer)write_le_8byte, .read_bin = (bin_reader)read_le_8byte};
static const struct primtype_info_t u8_info =
  {.binname = "  u8", .type_name = "u8",   .size = 1,
   .write_str = (writer)write_str_u8, .read_str = (str_reader)read_str_u8,
   .write_bin = (writer)write_byte, .read_bin = (bin_reader)read_byte};
static const struct primtype_info_t u16_info =
  {.binname = " u16", .type_name = "u16",  .size = 2,
   .write_str = (writer)write_str_u16, .read_str = (str_reader)read_str_u16,
   .write_bin = (writer)write_le_2byte, .read_bin = (bin_reader)read_le_2byte};
static const struct primtype_info_t u32_info =
  {.binname = " u32", .type_name = "u32",  .size = 4,
   .write_str = (writer)write_str_u32, .read_str = (str_reader)read_str_u32,
   .write_bin = (writer)write_le_4byte, .read_bin = (bin_reader)read_le_4byte};
static const struct primtype_info_t u64_info =
  {.binname = " u64", .type_name = "u64",  .size = 8,
   .write_str = (writer)write_str_u64, .read_str = (str_reader)read_str_u64,
   .write_bin = (writer)write_le_8byte, .read_bin = (bin_reader)read_le_8byte};
static const struct primtype_info_t f32_info =
  {.binname = " f32", .type_name = "f32",  .size = 4,
   .write_str = (writer)write_str_f32, .read_str = (str_reader)read_str_f32,
   .write_bin = (writer)write_le_4byte, .read_bin = (bin_reader)read_le_4byte};
static const struct primtype_info_t f64_info =
  {.binname = " f64", .type_name = "f64",  .size = 8,
   .write_str = (writer)write_str_f64, .read_str = (str_reader)read_str_f64,
   .write_bin = (writer)write_le_8byte, .read_bin = (bin_reader)read_le_8byte};
static const struct primtype_info_t bool_info =
  {.binname = "bool", .type_name = "bool", .size = 1,
   .write_str = (writer)write_str_bool, .read_str = (str_reader)read_str_bool,
   .write_bin = (writer)write_byte, .read_bin = (bin_reader)read_byte};

static const struct primtype_info_t* primtypes[] = {
  &i8_info, &i16_info, &i32_info, &i64_info,
  &u8_info, &u16_info, &u32_info, &u64_info,
  &f32_info, &f64_info,
  &bool_info,
  NULL // NULL-terminated
};

// General value interface.  All endian business taken care of at
// lower layers.

static int read_is_binary() {
  skipspaces();
  int c = getchar();
  if (c == 'b') {
    int8_t bin_version;
    int ret = read_byte(&bin_version);

    if (ret != 0) { panic(1, "binary-input: could not read version.\n"); }

    if (bin_version != BINARY_FORMAT_VERSION) {
      panic(1, "binary-input: File uses version %i, but I only understand version %i.\n",
            bin_version, BINARY_FORMAT_VERSION);
    }

    return 1;
  }
  ungetc(c, stdin);
  return 0;
}

static const struct primtype_info_t* read_bin_read_type_enum() {
  char read_binname[4];

  int num_matched = scanf("%4c", read_binname);
  if (num_matched != 1) { panic(1, "binary-input: Couldn't read element type.\n"); }

  const struct primtype_info_t **type = primtypes;

  for (; *type != NULL; type++) {
    // I compare the 4 characters manually instead of using strncmp because
    // this allows any value to be used, also NULL bytes
    if (memcmp(read_binname, (*type)->binname, 4) == 0) {
      return *type;
    }
  }
  panic(1, "binary-input: Did not recognize the type '%s'.\n", read_binname);
  return NULL;
}

static void read_bin_ensure_scalar(const struct primtype_info_t *expected_type) {
  int8_t bin_dims;
  int ret = read_byte(&bin_dims);
  if (ret != 0) { panic(1, "binary-input: Couldn't get dims.\n"); }

  if (bin_dims != 0) {
    panic(1, "binary-input: Expected scalar (0 dimensions), but got array with %i dimensions.\n",
          bin_dims);
  }

  const struct primtype_info_t *bin_type = read_bin_read_type_enum();
  if (bin_type != expected_type) {
    panic(1, "binary-input: Expected scalar of type %s but got scalar of type %s.\n",
          expected_type->type_name,
          bin_type->type_name);
  }
}

//// High-level interface

static int read_bin_array(const struct primtype_info_t *expected_type, void **data, int64_t *shape, int64_t dims) {
  int ret;

  int8_t bin_dims;
  ret = read_byte(&bin_dims);
  if (ret != 0) { panic(1, "binary-input: Couldn't get dims.\n"); }

  if (bin_dims != dims) {
    panic(1, "binary-input: Expected %i dimensions, but got array with %i dimensions.\n",
          dims, bin_dims);
  }

  const struct primtype_info_t *bin_primtype = read_bin_read_type_enum();
  if (expected_type != bin_primtype) {
    panic(1, "binary-input: Expected %iD-array with element type '%s' but got %iD-array with element type '%s'.\n",
          dims, expected_type->type_name, dims, bin_primtype->type_name);
  }

  uint64_t elem_count = 1;
  for (int i=0; i<dims; i++) {
    uint64_t bin_shape;
    ret = read_le_8byte(&bin_shape);
    if (ret != 0) { panic(1, "binary-input: Couldn't read size for dimension %i of array.\n", i); }
    elem_count *= bin_shape;
    shape[i] = (int64_t) bin_shape;
  }

  size_t elem_size = expected_type->size;
  void* tmp = realloc(*data, elem_count * elem_size);
  if (tmp == NULL) {
    panic(1, "binary-input: Failed to allocate array of size %i.\n",
          elem_count * elem_size);
  }
  *data = tmp;

  size_t num_elems_read = fread(*data, elem_size, elem_count, stdin);
  if (num_elems_read != elem_count) {
    panic(1, "binary-input: tried to read %i elements of an array, but only got %i elements.\n",
          elem_count, num_elems_read);
  }

  // If we're on big endian platform we must change all multibyte elements
  // from using little endian to big endian
  if (IS_BIG_ENDIAN && elem_size != 1) {
    char* elems = (char*) *data;
    for (uint64_t i=0; i<elem_count; i++) {
      char* elem = elems+(i*elem_size);
      for (unsigned int j=0; j<elem_size/2; j++) {
        char head = elem[j];
        int tail_index = elem_size-1-j;
        elem[j] = elem[tail_index];
        elem[tail_index] = head;
      }
    }
  }

  return 0;
}

static int read_array(const struct primtype_info_t *expected_type, void **data, int64_t *shape, int64_t dims) {
  if (!read_is_binary()) {
    return read_str_array(expected_type->size, (str_reader)expected_type->read_str, expected_type->type_name, data, shape, dims);
  } else {
    return read_bin_array(expected_type, data, shape, dims);
  }
}

static int write_str_array(FILE *out, const struct primtype_info_t *elem_type, unsigned char *data, int64_t *shape, int8_t rank) {
  if (rank==0) {
    elem_type->write_str(out, (void*)data);
  } else {
    int64_t len = shape[0];
    int64_t slice_size = 1;

    int64_t elem_size = elem_type->size;
    for (int64_t i = 1; i < rank; i++) {
      slice_size *= shape[i];
    }

    if (len*slice_size == 0) {
      printf("empty(");
      for (int64_t i = 1; i < rank; i++) {
        printf("[]");
      }
      printf("%s", elem_type->type_name);
      printf(")");
    } else if (rank==1) {
      putchar('[');
      for (int64_t i = 0; i < len; i++) {
        elem_type->write_str(out, (void*) (data + i * elem_size));
        if (i != len-1) {
          printf(", ");
        }
      }
      putchar(']');
    } else {
      putchar('[');
      for (int64_t i = 0; i < len; i++) {
        write_str_array(out, elem_type, data + i * slice_size * elem_size, shape+1, rank-1);
        if (i != len-1) {
          printf(", ");
        }
      }
      putchar(']');
    }
  }
  return 0;
}

static int write_bin_array(FILE *out, const struct primtype_info_t *elem_type, unsigned char *data, int64_t *shape, int8_t rank) {
  int64_t num_elems = 1;
  for (int64_t i = 0; i < rank; i++) {
    num_elems *= shape[i];
  }

  fputc('b', out);
  fputc((char)BINARY_FORMAT_VERSION, out);
  fwrite(&rank, sizeof(int8_t), 1, out);
  fputs(elem_type->binname, out);
  if (shape != NULL) {
    fwrite(shape, sizeof(int64_t), rank, out);
  }

  if (IS_BIG_ENDIAN) {
    for (int64_t i = 0; i < num_elems; i++) {
      unsigned char *elem = data+i*elem_type->size;
      for (int64_t j = 0; j < elem_type->size; j++) {
        fwrite(&elem[elem_type->size-j], 1, 1, out);
      }
    }
  } else {
    fwrite(data, elem_type->size, num_elems, out);
  }

  return 0;
}

static int write_array(FILE *out, int write_binary,
                       const struct primtype_info_t *elem_type, void *data, int64_t *shape, int8_t rank) {
  if (write_binary) {
    return write_bin_array(out, elem_type, data, shape, rank);
  } else {
    return write_str_array(out, elem_type, data, shape, rank);
  }
}

static int read_scalar(const struct primtype_info_t *expected_type, void *dest) {
  if (!read_is_binary()) {
    char buf[100];
    next_token(buf, sizeof(buf));
    return expected_type->read_str(buf, dest);
  } else {
    read_bin_ensure_scalar(expected_type);
    return expected_type->read_bin(dest);
  }
}

static int write_scalar(FILE *out, int write_binary, const struct primtype_info_t *type, void *src) {
  if (write_binary) {
    return write_bin_array(out, type, src, NULL, 0);
  } else {
    return type->write_str(out, src);
  }
}

// End of values.h.

static int binary_output = 0;
static FILE *runtime_file;
static int perform_warmup = 0;
static int num_runs = 1;
static const char *entry_point = "main";
// Start of tuning.h.

static char* load_tuning_file(const char *fname,
                              void *cfg,
                              int (*set_size)(void*, const char*, size_t)) {
  const int max_line_len = 1024;
  char* line = (char*) malloc(max_line_len);

  FILE *f = fopen(fname, "r");

  if (f == NULL) {
    snprintf(line, max_line_len, "Cannot open file: %s", strerror(errno));
    return line;
  }

  int lineno = 0;
  while (fgets(line, max_line_len, f) != NULL) {
    lineno++;
    char *eql = strstr(line, "=");
    if (eql) {
      *eql = 0;
      int value = atoi(eql+1);
      if (set_size(cfg, line, value) != 0) {
        strncpy(eql+1, line, max_line_len-strlen(line)-1);
        snprintf(line, max_line_len, "Unknown name '%s' on line %d.", eql+1, lineno);
        return line;
      }
    } else {
      snprintf(line, max_line_len, "Invalid line %d (must be of form 'name=int').",
               lineno);
      return line;
    }
  }

  free(line);

  return NULL;
}

// End of tuning.h.

int parse_options(struct futhark_context_config *cfg, int argc,
                  char *const argv[])
{
    int ch;
    static struct option long_options[] = {{"write-runtime-to",
                                            required_argument, NULL, 1},
                                           {"runs", required_argument, NULL, 2},
                                           {"debugging", no_argument, NULL, 3},
                                           {"log", no_argument, NULL, 4},
                                           {"entry-point", required_argument,
                                            NULL, 5}, {"binary-output",
                                                       no_argument, NULL, 6},
                                           {0, 0, 0, 0}};
    
    while ((ch = getopt_long(argc, argv, ":t:r:DLe:b", long_options, NULL)) !=
           -1) {
        if (ch == 1 || ch == 't') {
            runtime_file = fopen(optarg, "w");
            if (runtime_file == NULL)
                panic(1, "Cannot open %s: %s\n", optarg, strerror(errno));
        }
        if (ch == 2 || ch == 'r') {
            num_runs = atoi(optarg);
            perform_warmup = 1;
            if (num_runs <= 0)
                panic(1, "Need a positive number of runs, not %s\n", optarg);
        }
        if (ch == 3 || ch == 'D')
            futhark_context_config_set_debugging(cfg, 1);
        if (ch == 4 || ch == 'L')
            futhark_context_config_set_logging(cfg, 1);
        if (ch == 5 || ch == 'e') {
            if (entry_point != NULL)
                entry_point = optarg;
        }
        if (ch == 6 || ch == 'b')
            binary_output = 1;
        if (ch == ':')
            panic(-1, "Missing argument for option %s\n", argv[optind - 1]);
        if (ch == '?') {
            fprintf(stderr, "Usage: %s: %s\n", fut_progname,
                    "[-t/--write-runtime-to FILE] [-r/--runs INT] [-D/--debugging] [-L/--log] [-e/--entry-point NAME] [-b/--binary-output]");
            panic(1, "Unknown option: %s\n", argv[optind - 1]);
        }
    }
    return optind;
}
static void futrts_cli_entry_main(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs;
    
    /* Declare and read input. */
    set_binary_mode(stdin);
    
    struct futhark_f32_2d *read_value_36250;
    int64_t read_shape_36251[2];
    float *read_arr_36252 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_36252, read_shape_36251, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_3d *read_value_36253;
    int64_t read_shape_36254[3];
    float *read_arr_36255 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_36255, read_shape_36254, 3) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1,
              "[][][]", f32_info.type_name, strerror(errno));
    
    struct futhark_f32_3d *read_value_36256;
    int64_t read_shape_36257[3];
    float *read_arr_36258 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_36258, read_shape_36257, 3) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 2,
              "[][][]", f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_36259;
    int64_t read_shape_36260[2];
    float *read_arr_36261 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_36261, read_shape_36260, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 3, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_36262;
    int64_t read_shape_36263[2];
    float *read_arr_36264 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_36264, read_shape_36263, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 4, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_36265;
    int64_t read_shape_36266[2];
    float *read_arr_36267 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_36267, read_shape_36266, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 5, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_i32_1d *read_value_36268;
    int64_t read_shape_36269[1];
    int32_t *read_arr_36270 = NULL;
    
    errno = 0;
    if (read_array(&i32_info, (void **) &read_arr_36270, read_shape_36269, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 6, "[]",
              i32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_36271;
    int64_t read_shape_36272[2];
    float *read_arr_36273 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_36273, read_shape_36272, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 7, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_i32_2d *read_value_36274;
    int64_t read_shape_36275[2];
    int32_t *read_arr_36276 = NULL;
    
    errno = 0;
    if (read_array(&i32_info, (void **) &read_arr_36276, read_shape_36275, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 8, "[][]",
              i32_info.type_name, strerror(errno));
    
    struct futhark_i32_1d *read_value_36277;
    int64_t read_shape_36278[1];
    int32_t *read_arr_36279 = NULL;
    
    errno = 0;
    if (read_array(&i32_info, (void **) &read_arr_36279, read_shape_36278, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 9, "[]",
              i32_info.type_name, strerror(errno));
    
    struct futhark_i32_1d *read_value_36280;
    int64_t read_shape_36281[1];
    int32_t *read_arr_36282 = NULL;
    
    errno = 0;
    if (read_array(&i32_info, (void **) &read_arr_36282, read_shape_36281, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 10, "[]",
              i32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_36283;
    int64_t read_shape_36284[1];
    float *read_arr_36285 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_36285, read_shape_36284, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 11, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_36286;
    int64_t read_shape_36287[1];
    float *read_arr_36288 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_36288, read_shape_36287, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 12, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_36289;
    int64_t read_shape_36290[2];
    float *read_arr_36291 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_36291, read_shape_36290, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 13,
              "[][]", f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_36292;
    int64_t read_shape_36293[2];
    float *read_arr_36294 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_36294, read_shape_36293, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 14,
              "[][]", f32_info.type_name, strerror(errno));
    
    struct futhark_i32_1d *read_value_36295;
    int64_t read_shape_36296[1];
    int32_t *read_arr_36297 = NULL;
    
    errno = 0;
    if (read_array(&i32_info, (void **) &read_arr_36297, read_shape_36296, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 15, "[]",
              i32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_36298;
    int64_t read_shape_36299[1];
    float *read_arr_36300 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_36300, read_shape_36299, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 16, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_36301;
    int64_t read_shape_36302[2];
    float *read_arr_36303 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_36303, read_shape_36302, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 17,
              "[][]", f32_info.type_name, strerror(errno));
    
    struct futhark_f32_3d *read_value_36304;
    int64_t read_shape_36305[3];
    float *read_arr_36306 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_36306, read_shape_36305, 3) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 18,
              "[][][]", f32_info.type_name, strerror(errno));
    
    struct futhark_f32_3d *read_value_36307;
    int64_t read_shape_36308[3];
    float *read_arr_36309 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_36309, read_shape_36308, 3) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 19,
              "[][][]", f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_36310;
    int64_t read_shape_36311[2];
    float *read_arr_36312 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_36312, read_shape_36311, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 20,
              "[][]", f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_36313;
    int64_t read_shape_36314[2];
    float *read_arr_36315 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_36315, read_shape_36314, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 21,
              "[][]", f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_36316;
    int64_t read_shape_36317[2];
    float *read_arr_36318 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_36318, read_shape_36317, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 22,
              "[][]", f32_info.type_name, strerror(errno));
    
    struct futhark_i32_1d *read_value_36319;
    int64_t read_shape_36320[1];
    int32_t *read_arr_36321 = NULL;
    
    errno = 0;
    if (read_array(&i32_info, (void **) &read_arr_36321, read_shape_36320, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 23, "[]",
              i32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_36322;
    int64_t read_shape_36323[2];
    float *read_arr_36324 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_36324, read_shape_36323, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 24,
              "[][]", f32_info.type_name, strerror(errno));
    
    struct futhark_i32_2d *read_value_36325;
    int64_t read_shape_36326[2];
    int32_t *read_arr_36327 = NULL;
    
    errno = 0;
    if (read_array(&i32_info, (void **) &read_arr_36327, read_shape_36326, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 25,
              "[][]", i32_info.type_name, strerror(errno));
    
    struct futhark_i32_1d *read_value_36328;
    int64_t read_shape_36329[1];
    int32_t *read_arr_36330 = NULL;
    
    errno = 0;
    if (read_array(&i32_info, (void **) &read_arr_36330, read_shape_36329, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 26, "[]",
              i32_info.type_name, strerror(errno));
    
    struct futhark_i32_1d *read_value_36331;
    int64_t read_shape_36332[1];
    int32_t *read_arr_36333 = NULL;
    
    errno = 0;
    if (read_array(&i32_info, (void **) &read_arr_36333, read_shape_36332, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 27, "[]",
              i32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_36334;
    int64_t read_shape_36335[1];
    float *read_arr_36336 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_36336, read_shape_36335, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 28, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_36337;
    int64_t read_shape_36338[1];
    float *read_arr_36339 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_36339, read_shape_36338, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 29, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_36340;
    int64_t read_shape_36341[2];
    float *read_arr_36342 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_36342, read_shape_36341, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 30,
              "[][]", f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_36343;
    int64_t read_shape_36344[2];
    float *read_arr_36345 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_36345, read_shape_36344, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 31,
              "[][]", f32_info.type_name, strerror(errno));
    
    struct futhark_i32_1d *read_value_36346;
    int64_t read_shape_36347[1];
    int32_t *read_arr_36348 = NULL;
    
    errno = 0;
    if (read_array(&i32_info, (void **) &read_arr_36348, read_shape_36347, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 32, "[]",
              i32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_36349;
    int64_t read_shape_36350[1];
    float *read_arr_36351 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_36351, read_shape_36350, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 33, "[]",
              f32_info.type_name, strerror(errno));
    
    bool result_36352;
    bool result_36353;
    bool result_36354;
    bool result_36355;
    bool result_36356;
    bool result_36357;
    bool result_36358;
    bool result_36359;
    bool result_36360;
    bool result_36361;
    bool result_36362;
    bool result_36363;
    bool result_36364;
    bool result_36365;
    bool result_36366;
    bool result_36367;
    bool result_36368;
    
    if (perform_warmup) {
        time_runs = 0;
        
        int r;
        
        assert((read_value_36250 = futhark_new_f32_2d(ctx, read_arr_36252,
                                                      read_shape_36251[0],
                                                      read_shape_36251[1])) !=
            0);
        assert((read_value_36253 = futhark_new_f32_3d(ctx, read_arr_36255,
                                                      read_shape_36254[0],
                                                      read_shape_36254[1],
                                                      read_shape_36254[2])) !=
            0);
        assert((read_value_36256 = futhark_new_f32_3d(ctx, read_arr_36258,
                                                      read_shape_36257[0],
                                                      read_shape_36257[1],
                                                      read_shape_36257[2])) !=
            0);
        assert((read_value_36259 = futhark_new_f32_2d(ctx, read_arr_36261,
                                                      read_shape_36260[0],
                                                      read_shape_36260[1])) !=
            0);
        assert((read_value_36262 = futhark_new_f32_2d(ctx, read_arr_36264,
                                                      read_shape_36263[0],
                                                      read_shape_36263[1])) !=
            0);
        assert((read_value_36265 = futhark_new_f32_2d(ctx, read_arr_36267,
                                                      read_shape_36266[0],
                                                      read_shape_36266[1])) !=
            0);
        assert((read_value_36268 = futhark_new_i32_1d(ctx, read_arr_36270,
                                                      read_shape_36269[0])) !=
            0);
        assert((read_value_36271 = futhark_new_f32_2d(ctx, read_arr_36273,
                                                      read_shape_36272[0],
                                                      read_shape_36272[1])) !=
            0);
        assert((read_value_36274 = futhark_new_i32_2d(ctx, read_arr_36276,
                                                      read_shape_36275[0],
                                                      read_shape_36275[1])) !=
            0);
        assert((read_value_36277 = futhark_new_i32_1d(ctx, read_arr_36279,
                                                      read_shape_36278[0])) !=
            0);
        assert((read_value_36280 = futhark_new_i32_1d(ctx, read_arr_36282,
                                                      read_shape_36281[0])) !=
            0);
        assert((read_value_36283 = futhark_new_f32_1d(ctx, read_arr_36285,
                                                      read_shape_36284[0])) !=
            0);
        assert((read_value_36286 = futhark_new_f32_1d(ctx, read_arr_36288,
                                                      read_shape_36287[0])) !=
            0);
        assert((read_value_36289 = futhark_new_f32_2d(ctx, read_arr_36291,
                                                      read_shape_36290[0],
                                                      read_shape_36290[1])) !=
            0);
        assert((read_value_36292 = futhark_new_f32_2d(ctx, read_arr_36294,
                                                      read_shape_36293[0],
                                                      read_shape_36293[1])) !=
            0);
        assert((read_value_36295 = futhark_new_i32_1d(ctx, read_arr_36297,
                                                      read_shape_36296[0])) !=
            0);
        assert((read_value_36298 = futhark_new_f32_1d(ctx, read_arr_36300,
                                                      read_shape_36299[0])) !=
            0);
        assert((read_value_36301 = futhark_new_f32_2d(ctx, read_arr_36303,
                                                      read_shape_36302[0],
                                                      read_shape_36302[1])) !=
            0);
        assert((read_value_36304 = futhark_new_f32_3d(ctx, read_arr_36306,
                                                      read_shape_36305[0],
                                                      read_shape_36305[1],
                                                      read_shape_36305[2])) !=
            0);
        assert((read_value_36307 = futhark_new_f32_3d(ctx, read_arr_36309,
                                                      read_shape_36308[0],
                                                      read_shape_36308[1],
                                                      read_shape_36308[2])) !=
            0);
        assert((read_value_36310 = futhark_new_f32_2d(ctx, read_arr_36312,
                                                      read_shape_36311[0],
                                                      read_shape_36311[1])) !=
            0);
        assert((read_value_36313 = futhark_new_f32_2d(ctx, read_arr_36315,
                                                      read_shape_36314[0],
                                                      read_shape_36314[1])) !=
            0);
        assert((read_value_36316 = futhark_new_f32_2d(ctx, read_arr_36318,
                                                      read_shape_36317[0],
                                                      read_shape_36317[1])) !=
            0);
        assert((read_value_36319 = futhark_new_i32_1d(ctx, read_arr_36321,
                                                      read_shape_36320[0])) !=
            0);
        assert((read_value_36322 = futhark_new_f32_2d(ctx, read_arr_36324,
                                                      read_shape_36323[0],
                                                      read_shape_36323[1])) !=
            0);
        assert((read_value_36325 = futhark_new_i32_2d(ctx, read_arr_36327,
                                                      read_shape_36326[0],
                                                      read_shape_36326[1])) !=
            0);
        assert((read_value_36328 = futhark_new_i32_1d(ctx, read_arr_36330,
                                                      read_shape_36329[0])) !=
            0);
        assert((read_value_36331 = futhark_new_i32_1d(ctx, read_arr_36333,
                                                      read_shape_36332[0])) !=
            0);
        assert((read_value_36334 = futhark_new_f32_1d(ctx, read_arr_36336,
                                                      read_shape_36335[0])) !=
            0);
        assert((read_value_36337 = futhark_new_f32_1d(ctx, read_arr_36339,
                                                      read_shape_36338[0])) !=
            0);
        assert((read_value_36340 = futhark_new_f32_2d(ctx, read_arr_36342,
                                                      read_shape_36341[0],
                                                      read_shape_36341[1])) !=
            0);
        assert((read_value_36343 = futhark_new_f32_2d(ctx, read_arr_36345,
                                                      read_shape_36344[0],
                                                      read_shape_36344[1])) !=
            0);
        assert((read_value_36346 = futhark_new_i32_1d(ctx, read_arr_36348,
                                                      read_shape_36347[0])) !=
            0);
        assert((read_value_36349 = futhark_new_f32_1d(ctx, read_arr_36351,
                                                      read_shape_36350[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_main(ctx, &result_36352, &result_36353, &result_36354,
                               &result_36355, &result_36356, &result_36357,
                               &result_36358, &result_36359, &result_36360,
                               &result_36361, &result_36362, &result_36363,
                               &result_36364, &result_36365, &result_36366,
                               &result_36367, &result_36368, read_value_36250,
                               read_value_36253, read_value_36256,
                               read_value_36259, read_value_36262,
                               read_value_36265, read_value_36268,
                               read_value_36271, read_value_36274,
                               read_value_36277, read_value_36280,
                               read_value_36283, read_value_36286,
                               read_value_36289, read_value_36292,
                               read_value_36295, read_value_36298,
                               read_value_36301, read_value_36304,
                               read_value_36307, read_value_36310,
                               read_value_36313, read_value_36316,
                               read_value_36319, read_value_36322,
                               read_value_36325, read_value_36328,
                               read_value_36331, read_value_36334,
                               read_value_36337, read_value_36340,
                               read_value_36343, read_value_36346,
                               read_value_36349);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_2d(ctx, read_value_36250) == 0);
        assert(futhark_free_f32_3d(ctx, read_value_36253) == 0);
        assert(futhark_free_f32_3d(ctx, read_value_36256) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_36259) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_36262) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_36265) == 0);
        assert(futhark_free_i32_1d(ctx, read_value_36268) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_36271) == 0);
        assert(futhark_free_i32_2d(ctx, read_value_36274) == 0);
        assert(futhark_free_i32_1d(ctx, read_value_36277) == 0);
        assert(futhark_free_i32_1d(ctx, read_value_36280) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_36283) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_36286) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_36289) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_36292) == 0);
        assert(futhark_free_i32_1d(ctx, read_value_36295) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_36298) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_36301) == 0);
        assert(futhark_free_f32_3d(ctx, read_value_36304) == 0);
        assert(futhark_free_f32_3d(ctx, read_value_36307) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_36310) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_36313) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_36316) == 0);
        assert(futhark_free_i32_1d(ctx, read_value_36319) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_36322) == 0);
        assert(futhark_free_i32_2d(ctx, read_value_36325) == 0);
        assert(futhark_free_i32_1d(ctx, read_value_36328) == 0);
        assert(futhark_free_i32_1d(ctx, read_value_36331) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_36334) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_36337) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_36340) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_36343) == 0);
        assert(futhark_free_i32_1d(ctx, read_value_36346) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_36349) == 0);
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
        ;
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        int r;
        
        assert((read_value_36250 = futhark_new_f32_2d(ctx, read_arr_36252,
                                                      read_shape_36251[0],
                                                      read_shape_36251[1])) !=
            0);
        assert((read_value_36253 = futhark_new_f32_3d(ctx, read_arr_36255,
                                                      read_shape_36254[0],
                                                      read_shape_36254[1],
                                                      read_shape_36254[2])) !=
            0);
        assert((read_value_36256 = futhark_new_f32_3d(ctx, read_arr_36258,
                                                      read_shape_36257[0],
                                                      read_shape_36257[1],
                                                      read_shape_36257[2])) !=
            0);
        assert((read_value_36259 = futhark_new_f32_2d(ctx, read_arr_36261,
                                                      read_shape_36260[0],
                                                      read_shape_36260[1])) !=
            0);
        assert((read_value_36262 = futhark_new_f32_2d(ctx, read_arr_36264,
                                                      read_shape_36263[0],
                                                      read_shape_36263[1])) !=
            0);
        assert((read_value_36265 = futhark_new_f32_2d(ctx, read_arr_36267,
                                                      read_shape_36266[0],
                                                      read_shape_36266[1])) !=
            0);
        assert((read_value_36268 = futhark_new_i32_1d(ctx, read_arr_36270,
                                                      read_shape_36269[0])) !=
            0);
        assert((read_value_36271 = futhark_new_f32_2d(ctx, read_arr_36273,
                                                      read_shape_36272[0],
                                                      read_shape_36272[1])) !=
            0);
        assert((read_value_36274 = futhark_new_i32_2d(ctx, read_arr_36276,
                                                      read_shape_36275[0],
                                                      read_shape_36275[1])) !=
            0);
        assert((read_value_36277 = futhark_new_i32_1d(ctx, read_arr_36279,
                                                      read_shape_36278[0])) !=
            0);
        assert((read_value_36280 = futhark_new_i32_1d(ctx, read_arr_36282,
                                                      read_shape_36281[0])) !=
            0);
        assert((read_value_36283 = futhark_new_f32_1d(ctx, read_arr_36285,
                                                      read_shape_36284[0])) !=
            0);
        assert((read_value_36286 = futhark_new_f32_1d(ctx, read_arr_36288,
                                                      read_shape_36287[0])) !=
            0);
        assert((read_value_36289 = futhark_new_f32_2d(ctx, read_arr_36291,
                                                      read_shape_36290[0],
                                                      read_shape_36290[1])) !=
            0);
        assert((read_value_36292 = futhark_new_f32_2d(ctx, read_arr_36294,
                                                      read_shape_36293[0],
                                                      read_shape_36293[1])) !=
            0);
        assert((read_value_36295 = futhark_new_i32_1d(ctx, read_arr_36297,
                                                      read_shape_36296[0])) !=
            0);
        assert((read_value_36298 = futhark_new_f32_1d(ctx, read_arr_36300,
                                                      read_shape_36299[0])) !=
            0);
        assert((read_value_36301 = futhark_new_f32_2d(ctx, read_arr_36303,
                                                      read_shape_36302[0],
                                                      read_shape_36302[1])) !=
            0);
        assert((read_value_36304 = futhark_new_f32_3d(ctx, read_arr_36306,
                                                      read_shape_36305[0],
                                                      read_shape_36305[1],
                                                      read_shape_36305[2])) !=
            0);
        assert((read_value_36307 = futhark_new_f32_3d(ctx, read_arr_36309,
                                                      read_shape_36308[0],
                                                      read_shape_36308[1],
                                                      read_shape_36308[2])) !=
            0);
        assert((read_value_36310 = futhark_new_f32_2d(ctx, read_arr_36312,
                                                      read_shape_36311[0],
                                                      read_shape_36311[1])) !=
            0);
        assert((read_value_36313 = futhark_new_f32_2d(ctx, read_arr_36315,
                                                      read_shape_36314[0],
                                                      read_shape_36314[1])) !=
            0);
        assert((read_value_36316 = futhark_new_f32_2d(ctx, read_arr_36318,
                                                      read_shape_36317[0],
                                                      read_shape_36317[1])) !=
            0);
        assert((read_value_36319 = futhark_new_i32_1d(ctx, read_arr_36321,
                                                      read_shape_36320[0])) !=
            0);
        assert((read_value_36322 = futhark_new_f32_2d(ctx, read_arr_36324,
                                                      read_shape_36323[0],
                                                      read_shape_36323[1])) !=
            0);
        assert((read_value_36325 = futhark_new_i32_2d(ctx, read_arr_36327,
                                                      read_shape_36326[0],
                                                      read_shape_36326[1])) !=
            0);
        assert((read_value_36328 = futhark_new_i32_1d(ctx, read_arr_36330,
                                                      read_shape_36329[0])) !=
            0);
        assert((read_value_36331 = futhark_new_i32_1d(ctx, read_arr_36333,
                                                      read_shape_36332[0])) !=
            0);
        assert((read_value_36334 = futhark_new_f32_1d(ctx, read_arr_36336,
                                                      read_shape_36335[0])) !=
            0);
        assert((read_value_36337 = futhark_new_f32_1d(ctx, read_arr_36339,
                                                      read_shape_36338[0])) !=
            0);
        assert((read_value_36340 = futhark_new_f32_2d(ctx, read_arr_36342,
                                                      read_shape_36341[0],
                                                      read_shape_36341[1])) !=
            0);
        assert((read_value_36343 = futhark_new_f32_2d(ctx, read_arr_36345,
                                                      read_shape_36344[0],
                                                      read_shape_36344[1])) !=
            0);
        assert((read_value_36346 = futhark_new_i32_1d(ctx, read_arr_36348,
                                                      read_shape_36347[0])) !=
            0);
        assert((read_value_36349 = futhark_new_f32_1d(ctx, read_arr_36351,
                                                      read_shape_36350[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_main(ctx, &result_36352, &result_36353, &result_36354,
                               &result_36355, &result_36356, &result_36357,
                               &result_36358, &result_36359, &result_36360,
                               &result_36361, &result_36362, &result_36363,
                               &result_36364, &result_36365, &result_36366,
                               &result_36367, &result_36368, read_value_36250,
                               read_value_36253, read_value_36256,
                               read_value_36259, read_value_36262,
                               read_value_36265, read_value_36268,
                               read_value_36271, read_value_36274,
                               read_value_36277, read_value_36280,
                               read_value_36283, read_value_36286,
                               read_value_36289, read_value_36292,
                               read_value_36295, read_value_36298,
                               read_value_36301, read_value_36304,
                               read_value_36307, read_value_36310,
                               read_value_36313, read_value_36316,
                               read_value_36319, read_value_36322,
                               read_value_36325, read_value_36328,
                               read_value_36331, read_value_36334,
                               read_value_36337, read_value_36340,
                               read_value_36343, read_value_36346,
                               read_value_36349);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_2d(ctx, read_value_36250) == 0);
        assert(futhark_free_f32_3d(ctx, read_value_36253) == 0);
        assert(futhark_free_f32_3d(ctx, read_value_36256) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_36259) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_36262) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_36265) == 0);
        assert(futhark_free_i32_1d(ctx, read_value_36268) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_36271) == 0);
        assert(futhark_free_i32_2d(ctx, read_value_36274) == 0);
        assert(futhark_free_i32_1d(ctx, read_value_36277) == 0);
        assert(futhark_free_i32_1d(ctx, read_value_36280) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_36283) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_36286) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_36289) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_36292) == 0);
        assert(futhark_free_i32_1d(ctx, read_value_36295) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_36298) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_36301) == 0);
        assert(futhark_free_f32_3d(ctx, read_value_36304) == 0);
        assert(futhark_free_f32_3d(ctx, read_value_36307) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_36310) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_36313) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_36316) == 0);
        assert(futhark_free_i32_1d(ctx, read_value_36319) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_36322) == 0);
        assert(futhark_free_i32_2d(ctx, read_value_36325) == 0);
        assert(futhark_free_i32_1d(ctx, read_value_36328) == 0);
        assert(futhark_free_i32_1d(ctx, read_value_36331) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_36334) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_36337) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_36340) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_36343) == 0);
        assert(futhark_free_i32_1d(ctx, read_value_36346) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_36349) == 0);
        if (run < num_runs - 1) {
            ;
            ;
            ;
            ;
            ;
            ;
            ;
            ;
            ;
            ;
            ;
            ;
            ;
            ;
            ;
            ;
            ;
        }
    }
    free(read_arr_36252);
    free(read_arr_36255);
    free(read_arr_36258);
    free(read_arr_36261);
    free(read_arr_36264);
    free(read_arr_36267);
    free(read_arr_36270);
    free(read_arr_36273);
    free(read_arr_36276);
    free(read_arr_36279);
    free(read_arr_36282);
    free(read_arr_36285);
    free(read_arr_36288);
    free(read_arr_36291);
    free(read_arr_36294);
    free(read_arr_36297);
    free(read_arr_36300);
    free(read_arr_36303);
    free(read_arr_36306);
    free(read_arr_36309);
    free(read_arr_36312);
    free(read_arr_36315);
    free(read_arr_36318);
    free(read_arr_36321);
    free(read_arr_36324);
    free(read_arr_36327);
    free(read_arr_36330);
    free(read_arr_36333);
    free(read_arr_36336);
    free(read_arr_36339);
    free(read_arr_36342);
    free(read_arr_36345);
    free(read_arr_36348);
    free(read_arr_36351);
    if (binary_output)
        set_binary_mode(stdout);
    write_scalar(stdout, binary_output, &bool_info, &result_36352);
    printf("\n");
    write_scalar(stdout, binary_output, &bool_info, &result_36353);
    printf("\n");
    write_scalar(stdout, binary_output, &bool_info, &result_36354);
    printf("\n");
    write_scalar(stdout, binary_output, &bool_info, &result_36355);
    printf("\n");
    write_scalar(stdout, binary_output, &bool_info, &result_36356);
    printf("\n");
    write_scalar(stdout, binary_output, &bool_info, &result_36357);
    printf("\n");
    write_scalar(stdout, binary_output, &bool_info, &result_36358);
    printf("\n");
    write_scalar(stdout, binary_output, &bool_info, &result_36359);
    printf("\n");
    write_scalar(stdout, binary_output, &bool_info, &result_36360);
    printf("\n");
    write_scalar(stdout, binary_output, &bool_info, &result_36361);
    printf("\n");
    write_scalar(stdout, binary_output, &bool_info, &result_36362);
    printf("\n");
    write_scalar(stdout, binary_output, &bool_info, &result_36363);
    printf("\n");
    write_scalar(stdout, binary_output, &bool_info, &result_36364);
    printf("\n");
    write_scalar(stdout, binary_output, &bool_info, &result_36365);
    printf("\n");
    write_scalar(stdout, binary_output, &bool_info, &result_36366);
    printf("\n");
    write_scalar(stdout, binary_output, &bool_info, &result_36367);
    printf("\n");
    write_scalar(stdout, binary_output, &bool_info, &result_36368);
    printf("\n");
    ;
    ;
    ;
    ;
    ;
    ;
    ;
    ;
    ;
    ;
    ;
    ;
    ;
    ;
    ;
    ;
    ;
}
typedef void entry_point_fun(struct futhark_context *);
struct entry_point_entry {
    const char *name;
    entry_point_fun *fun;
} ;
int main(int argc, char **argv)
{
    fut_progname = argv[0];
    
    struct entry_point_entry entry_points[] = {{.name ="main", .fun =
                                                futrts_cli_entry_main}};
    struct futhark_context_config *cfg = futhark_context_config_new();
    
    assert(cfg != NULL);
    
    int parsed_options = parse_options(cfg, argc, argv);
    
    argc -= parsed_options;
    argv += parsed_options;
    if (argc != 0)
        panic(1, "Excess non-option: %s\n", argv[0]);
    
    struct futhark_context *ctx = futhark_context_new(cfg);
    
    assert(ctx != NULL);
    if (entry_point != NULL) {
        int num_entry_points = sizeof(entry_points) / sizeof(entry_points[0]);
        entry_point_fun *entry_point_fun = NULL;
        
        for (int i = 0; i < num_entry_points; i++) {
            if (strcmp(entry_points[i].name, entry_point) == 0) {
                entry_point_fun = entry_points[i].fun;
                break;
            }
        }
        if (entry_point_fun == NULL) {
            fprintf(stderr,
                    "No entry point '%s'.  Select another with --entry-point.  Options are:\n",
                    entry_point);
            for (int i = 0; i < num_entry_points; i++)
                fprintf(stderr, "%s\n", entry_points[i].name);
            return 1;
        }
        entry_point_fun(ctx);
        if (runtime_file != NULL)
            fclose(runtime_file);
        futhark_debugging_report(ctx);
    }
    futhark_context_free(ctx);
    futhark_context_config_free(cfg);
    return 0;
}
#ifdef _MSC_VER
#define inline __inline
#endif
#include <string.h>
#include <inttypes.h>
#include <ctype.h>
#include <errno.h>
#include <assert.h>
// Start of lock.h.

/* A very simple cross-platform implementation of locks.  Uses
   pthreads on Unix and some Windows thing there.  Futhark's
   host-level code is not multithreaded, but user code may be, so we
   need some mechanism for ensuring atomic access to API functions.
   This is that mechanism.  It is not exposed to user code at all, so
   we do not have to worry about name collisions. */

#ifdef _WIN32

typedef HANDLE lock_t;

static lock_t create_lock(lock_t *lock) {
  *lock = CreateMutex(NULL,  /* Default security attributes. */
                      FALSE, /* Initially unlocked. */
                      NULL); /* Unnamed. */
}

static void lock_lock(lock_t *lock) {
  assert(WaitForSingleObject(*lock, INFINITE) == WAIT_OBJECT_0);
}

static void lock_unlock(lock_t *lock) {
  assert(ReleaseMutex(*lock));
}

static void free_lock(lock_t *lock) {
  CloseHandle(*lock);
}

#else
/* Assuming POSIX */

#include <pthread.h>

typedef pthread_mutex_t lock_t;

static void create_lock(lock_t *lock) {
  int r = pthread_mutex_init(lock, NULL);
  assert(r == 0);
}

static void lock_lock(lock_t *lock) {
  int r = pthread_mutex_lock(lock);
  assert(r == 0);
}

static void lock_unlock(lock_t *lock) {
  int r = pthread_mutex_unlock(lock);
  assert(r == 0);
}

static void free_lock(lock_t *lock) {
  /* Nothing to do for pthreads. */
  (void)lock;
}

#endif

// End of lock.h.

struct memblock {
    int *references;
    char *mem;
    int64_t size;
    const char *desc;
} ;
struct futhark_context_config {
    int debugging;
} ;
struct futhark_context_config *futhark_context_config_new()
{
    struct futhark_context_config *cfg =
                                  (struct futhark_context_config *) malloc(sizeof(struct futhark_context_config));
    
    if (cfg == NULL)
        return NULL;
    cfg->debugging = 0;
    return cfg;
}
void futhark_context_config_free(struct futhark_context_config *cfg)
{
    free(cfg);
}
void futhark_context_config_set_debugging(struct futhark_context_config *cfg,
                                          int detail)
{
    cfg->debugging = detail;
}
void futhark_context_config_set_logging(struct futhark_context_config *cfg,
                                        int detail)
{
    /* Does nothing for this backend. */
    cfg = cfg;
    detail = detail;
}
struct futhark_context {
    int detail_memory;
    int debugging;
    lock_t lock;
    char *error;
    int64_t peak_mem_usage_default;
    int64_t cur_mem_usage_default;
} ;
struct futhark_context *futhark_context_new(struct futhark_context_config *cfg)
{
    struct futhark_context *ctx =
                           (struct futhark_context *) malloc(sizeof(struct futhark_context));
    
    if (ctx == NULL)
        return NULL;
    ctx->detail_memory = cfg->debugging;
    ctx->debugging = cfg->debugging;
    ctx->error = NULL;
    create_lock(&ctx->lock);
    ctx->peak_mem_usage_default = 0;
    ctx->cur_mem_usage_default = 0;
    return ctx;
}
void futhark_context_free(struct futhark_context *ctx)
{
    free_lock(&ctx->lock);
    free(ctx);
}
int futhark_context_sync(struct futhark_context *ctx)
{
    ctx = ctx;
    return 0;
}
char *futhark_context_get_error(struct futhark_context *ctx)
{
    char *error = ctx->error;
    
    ctx->error = NULL;
    return error;
}
static int memblock_unref(struct futhark_context *ctx, struct memblock *block,
                          const char *desc)
{
    if (block->references != NULL) {
        *block->references -= 1;
        if (ctx->detail_memory)
            fprintf(stderr,
                    "Unreferencing block %s (allocated as %s) in %s: %d references remaining.\n",
                    desc, block->desc, "default space", *block->references);
        if (*block->references == 0) {
            ctx->cur_mem_usage_default -= block->size;
            free(block->mem);
            free(block->references);
            if (ctx->detail_memory)
                fprintf(stderr,
                        "%lld bytes freed (now allocated: %lld bytes)\n",
                        (long long) block->size,
                        (long long) ctx->cur_mem_usage_default);
        }
        block->references = NULL;
    }
    return 0;
}
static int memblock_alloc(struct futhark_context *ctx, struct memblock *block,
                          int64_t size, const char *desc)
{
    if (size < 0)
        panic(1, "Negative allocation of %lld bytes attempted for %s in %s.\n",
              (long long) size, desc, "default space",
              ctx->cur_mem_usage_default);
    
    int ret = memblock_unref(ctx, block, desc);
    
    ctx->cur_mem_usage_default += size;
    if (ctx->detail_memory)
        fprintf(stderr,
                "Allocating %lld bytes for %s in %s (then allocated: %lld bytes)",
                (long long) size, desc, "default space",
                (long long) ctx->cur_mem_usage_default);
    if (ctx->cur_mem_usage_default > ctx->peak_mem_usage_default) {
        ctx->peak_mem_usage_default = ctx->cur_mem_usage_default;
        if (ctx->detail_memory)
            fprintf(stderr, " (new peak).\n");
    } else if (ctx->detail_memory)
        fprintf(stderr, ".\n");
    block->mem = (char *) malloc(size);
    block->references = (int *) malloc(sizeof(int));
    *block->references = 1;
    block->size = size;
    block->desc = desc;
    return ret;
}
static int memblock_set(struct futhark_context *ctx, struct memblock *lhs,
                        struct memblock *rhs, const char *lhs_desc)
{
    int ret = memblock_unref(ctx, lhs, lhs_desc);
    
    (*rhs->references)++;
    *lhs = *rhs;
    return ret;
}
void futhark_debugging_report(struct futhark_context *ctx)
{
    if (ctx->detail_memory) {
        fprintf(stderr, "Peak memory usage for default space: %lld bytes.\n",
                (long long) ctx->peak_mem_usage_default);
    }
    if (ctx->debugging) { }
}
static int futrts_main(struct futhark_context *ctx, bool *out_scalar_out_36233,
                       bool *out_scalar_out_36234, bool *out_scalar_out_36235,
                       bool *out_scalar_out_36236, bool *out_scalar_out_36237,
                       bool *out_scalar_out_36238, bool *out_scalar_out_36239,
                       bool *out_scalar_out_36240, bool *out_scalar_out_36241,
                       bool *out_scalar_out_36242, bool *out_scalar_out_36243,
                       bool *out_scalar_out_36244, bool *out_scalar_out_36245,
                       bool *out_scalar_out_36246, bool *out_scalar_out_36247,
                       bool *out_scalar_out_36248, bool *out_scalar_out_36249,
                       struct memblock X_mem_36153,
                       struct memblock Xsqr_mem_36154,
                       struct memblock Xinv_mem_36155,
                       struct memblock beta0_mem_36156,
                       struct memblock beta_mem_36157,
                       struct memblock y_preds_mem_36158,
                       struct memblock Nss_mem_36159,
                       struct memblock y_errors_mem_36160,
                       struct memblock val_indss_mem_36161,
                       struct memblock hs_mem_36162,
                       struct memblock nss_mem_36163,
                       struct memblock sigmas_mem_36164,
                       struct memblock MO_fsts_mem_36165,
                       struct memblock MOpp_mem_36166,
                       struct memblock MOp_mem_36167,
                       struct memblock breaks_mem_36168,
                       struct memblock means_mem_36169,
                       struct memblock Xseq_mem_36170,
                       struct memblock Xsqrseq_mem_36171,
                       struct memblock Xinvseq_mem_36172,
                       struct memblock beta0seq_mem_36173,
                       struct memblock betaseq_mem_36174,
                       struct memblock y_predsseq_mem_36175,
                       struct memblock Nssseq_mem_36176,
                       struct memblock y_errorsseq_mem_36177,
                       struct memblock val_indssseq_mem_36178,
                       struct memblock hsseq_mem_36179,
                       struct memblock nssseq_mem_36180,
                       struct memblock sigmasseq_mem_36181,
                       struct memblock MO_fstsseq_mem_36182,
                       struct memblock MOppseq_mem_36183,
                       struct memblock MOpseq_mem_36184,
                       struct memblock breaksseq_mem_36185,
                       struct memblock meansseq_mem_36186, int32_t sizze_35491,
                       int32_t sizze_35492, int32_t sizze_35493,
                       int32_t sizze_35494, int32_t sizze_35495,
                       int32_t sizze_35496, int32_t sizze_35497,
                       int32_t sizze_35498, int32_t sizze_35499,
                       int32_t sizze_35500, int32_t sizze_35501,
                       int32_t sizze_35502, int32_t sizze_35503,
                       int32_t sizze_35504, int32_t sizze_35505,
                       int32_t sizze_35506, int32_t sizze_35507,
                       int32_t sizze_35508, int32_t sizze_35509,
                       int32_t sizze_35510, int32_t sizze_35511,
                       int32_t sizze_35512, int32_t sizze_35513,
                       int32_t sizze_35514, int32_t sizze_35515,
                       int32_t sizze_35516, int32_t sizze_35517,
                       int32_t sizze_35518, int32_t sizze_35519,
                       int32_t sizze_35520, int32_t sizze_35521,
                       int32_t sizze_35522, int32_t sizze_35523,
                       int32_t sizze_35524, int32_t sizze_35525,
                       int32_t sizze_35526, int32_t sizze_35527,
                       int32_t sizze_35528, int32_t sizze_35529,
                       int32_t sizze_35530, int32_t sizze_35531,
                       int32_t sizze_35532, int32_t sizze_35533,
                       int32_t sizze_35534, int32_t sizze_35535,
                       int32_t sizze_35536, int32_t sizze_35537,
                       int32_t sizze_35538, int32_t sizze_35539,
                       int32_t sizze_35540, int32_t sizze_35541,
                       int32_t sizze_35542, int32_t sizze_35543,
                       int32_t sizze_35544, int32_t sizze_35545,
                       int32_t sizze_35546, int32_t sizze_35547,
                       int32_t sizze_35548);
static inline int8_t add8(int8_t x, int8_t y)
{
    return x + y;
}
static inline int16_t add16(int16_t x, int16_t y)
{
    return x + y;
}
static inline int32_t add32(int32_t x, int32_t y)
{
    return x + y;
}
static inline int64_t add64(int64_t x, int64_t y)
{
    return x + y;
}
static inline int8_t sub8(int8_t x, int8_t y)
{
    return x - y;
}
static inline int16_t sub16(int16_t x, int16_t y)
{
    return x - y;
}
static inline int32_t sub32(int32_t x, int32_t y)
{
    return x - y;
}
static inline int64_t sub64(int64_t x, int64_t y)
{
    return x - y;
}
static inline int8_t mul8(int8_t x, int8_t y)
{
    return x * y;
}
static inline int16_t mul16(int16_t x, int16_t y)
{
    return x * y;
}
static inline int32_t mul32(int32_t x, int32_t y)
{
    return x * y;
}
static inline int64_t mul64(int64_t x, int64_t y)
{
    return x * y;
}
static inline uint8_t udiv8(uint8_t x, uint8_t y)
{
    return x / y;
}
static inline uint16_t udiv16(uint16_t x, uint16_t y)
{
    return x / y;
}
static inline uint32_t udiv32(uint32_t x, uint32_t y)
{
    return x / y;
}
static inline uint64_t udiv64(uint64_t x, uint64_t y)
{
    return x / y;
}
static inline uint8_t umod8(uint8_t x, uint8_t y)
{
    return x % y;
}
static inline uint16_t umod16(uint16_t x, uint16_t y)
{
    return x % y;
}
static inline uint32_t umod32(uint32_t x, uint32_t y)
{
    return x % y;
}
static inline uint64_t umod64(uint64_t x, uint64_t y)
{
    return x % y;
}
static inline int8_t sdiv8(int8_t x, int8_t y)
{
    int8_t q = x / y;
    int8_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int16_t sdiv16(int16_t x, int16_t y)
{
    int16_t q = x / y;
    int16_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int32_t sdiv32(int32_t x, int32_t y)
{
    int32_t q = x / y;
    int32_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int64_t sdiv64(int64_t x, int64_t y)
{
    int64_t q = x / y;
    int64_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int8_t smod8(int8_t x, int8_t y)
{
    int8_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int16_t smod16(int16_t x, int16_t y)
{
    int16_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int32_t smod32(int32_t x, int32_t y)
{
    int32_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int64_t smod64(int64_t x, int64_t y)
{
    int64_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int8_t squot8(int8_t x, int8_t y)
{
    return x / y;
}
static inline int16_t squot16(int16_t x, int16_t y)
{
    return x / y;
}
static inline int32_t squot32(int32_t x, int32_t y)
{
    return x / y;
}
static inline int64_t squot64(int64_t x, int64_t y)
{
    return x / y;
}
static inline int8_t srem8(int8_t x, int8_t y)
{
    return x % y;
}
static inline int16_t srem16(int16_t x, int16_t y)
{
    return x % y;
}
static inline int32_t srem32(int32_t x, int32_t y)
{
    return x % y;
}
static inline int64_t srem64(int64_t x, int64_t y)
{
    return x % y;
}
static inline int8_t smin8(int8_t x, int8_t y)
{
    return x < y ? x : y;
}
static inline int16_t smin16(int16_t x, int16_t y)
{
    return x < y ? x : y;
}
static inline int32_t smin32(int32_t x, int32_t y)
{
    return x < y ? x : y;
}
static inline int64_t smin64(int64_t x, int64_t y)
{
    return x < y ? x : y;
}
static inline uint8_t umin8(uint8_t x, uint8_t y)
{
    return x < y ? x : y;
}
static inline uint16_t umin16(uint16_t x, uint16_t y)
{
    return x < y ? x : y;
}
static inline uint32_t umin32(uint32_t x, uint32_t y)
{
    return x < y ? x : y;
}
static inline uint64_t umin64(uint64_t x, uint64_t y)
{
    return x < y ? x : y;
}
static inline int8_t smax8(int8_t x, int8_t y)
{
    return x < y ? y : x;
}
static inline int16_t smax16(int16_t x, int16_t y)
{
    return x < y ? y : x;
}
static inline int32_t smax32(int32_t x, int32_t y)
{
    return x < y ? y : x;
}
static inline int64_t smax64(int64_t x, int64_t y)
{
    return x < y ? y : x;
}
static inline uint8_t umax8(uint8_t x, uint8_t y)
{
    return x < y ? y : x;
}
static inline uint16_t umax16(uint16_t x, uint16_t y)
{
    return x < y ? y : x;
}
static inline uint32_t umax32(uint32_t x, uint32_t y)
{
    return x < y ? y : x;
}
static inline uint64_t umax64(uint64_t x, uint64_t y)
{
    return x < y ? y : x;
}
static inline uint8_t shl8(uint8_t x, uint8_t y)
{
    return x << y;
}
static inline uint16_t shl16(uint16_t x, uint16_t y)
{
    return x << y;
}
static inline uint32_t shl32(uint32_t x, uint32_t y)
{
    return x << y;
}
static inline uint64_t shl64(uint64_t x, uint64_t y)
{
    return x << y;
}
static inline uint8_t lshr8(uint8_t x, uint8_t y)
{
    return x >> y;
}
static inline uint16_t lshr16(uint16_t x, uint16_t y)
{
    return x >> y;
}
static inline uint32_t lshr32(uint32_t x, uint32_t y)
{
    return x >> y;
}
static inline uint64_t lshr64(uint64_t x, uint64_t y)
{
    return x >> y;
}
static inline int8_t ashr8(int8_t x, int8_t y)
{
    return x >> y;
}
static inline int16_t ashr16(int16_t x, int16_t y)
{
    return x >> y;
}
static inline int32_t ashr32(int32_t x, int32_t y)
{
    return x >> y;
}
static inline int64_t ashr64(int64_t x, int64_t y)
{
    return x >> y;
}
static inline uint8_t and8(uint8_t x, uint8_t y)
{
    return x & y;
}
static inline uint16_t and16(uint16_t x, uint16_t y)
{
    return x & y;
}
static inline uint32_t and32(uint32_t x, uint32_t y)
{
    return x & y;
}
static inline uint64_t and64(uint64_t x, uint64_t y)
{
    return x & y;
}
static inline uint8_t or8(uint8_t x, uint8_t y)
{
    return x | y;
}
static inline uint16_t or16(uint16_t x, uint16_t y)
{
    return x | y;
}
static inline uint32_t or32(uint32_t x, uint32_t y)
{
    return x | y;
}
static inline uint64_t or64(uint64_t x, uint64_t y)
{
    return x | y;
}
static inline uint8_t xor8(uint8_t x, uint8_t y)
{
    return x ^ y;
}
static inline uint16_t xor16(uint16_t x, uint16_t y)
{
    return x ^ y;
}
static inline uint32_t xor32(uint32_t x, uint32_t y)
{
    return x ^ y;
}
static inline uint64_t xor64(uint64_t x, uint64_t y)
{
    return x ^ y;
}
static inline char ult8(uint8_t x, uint8_t y)
{
    return x < y;
}
static inline char ult16(uint16_t x, uint16_t y)
{
    return x < y;
}
static inline char ult32(uint32_t x, uint32_t y)
{
    return x < y;
}
static inline char ult64(uint64_t x, uint64_t y)
{
    return x < y;
}
static inline char ule8(uint8_t x, uint8_t y)
{
    return x <= y;
}
static inline char ule16(uint16_t x, uint16_t y)
{
    return x <= y;
}
static inline char ule32(uint32_t x, uint32_t y)
{
    return x <= y;
}
static inline char ule64(uint64_t x, uint64_t y)
{
    return x <= y;
}
static inline char slt8(int8_t x, int8_t y)
{
    return x < y;
}
static inline char slt16(int16_t x, int16_t y)
{
    return x < y;
}
static inline char slt32(int32_t x, int32_t y)
{
    return x < y;
}
static inline char slt64(int64_t x, int64_t y)
{
    return x < y;
}
static inline char sle8(int8_t x, int8_t y)
{
    return x <= y;
}
static inline char sle16(int16_t x, int16_t y)
{
    return x <= y;
}
static inline char sle32(int32_t x, int32_t y)
{
    return x <= y;
}
static inline char sle64(int64_t x, int64_t y)
{
    return x <= y;
}
static inline int8_t pow8(int8_t x, int8_t y)
{
    int8_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int16_t pow16(int16_t x, int16_t y)
{
    int16_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int32_t pow32(int32_t x, int32_t y)
{
    int32_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int64_t pow64(int64_t x, int64_t y)
{
    int64_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline bool itob_i8_bool(int8_t x)
{
    return x;
}
static inline bool itob_i16_bool(int16_t x)
{
    return x;
}
static inline bool itob_i32_bool(int32_t x)
{
    return x;
}
static inline bool itob_i64_bool(int64_t x)
{
    return x;
}
static inline int8_t btoi_bool_i8(bool x)
{
    return x;
}
static inline int16_t btoi_bool_i16(bool x)
{
    return x;
}
static inline int32_t btoi_bool_i32(bool x)
{
    return x;
}
static inline int64_t btoi_bool_i64(bool x)
{
    return x;
}
#define sext_i8_i8(x) ((int8_t) (int8_t) x)
#define sext_i8_i16(x) ((int16_t) (int8_t) x)
#define sext_i8_i32(x) ((int32_t) (int8_t) x)
#define sext_i8_i64(x) ((int64_t) (int8_t) x)
#define sext_i16_i8(x) ((int8_t) (int16_t) x)
#define sext_i16_i16(x) ((int16_t) (int16_t) x)
#define sext_i16_i32(x) ((int32_t) (int16_t) x)
#define sext_i16_i64(x) ((int64_t) (int16_t) x)
#define sext_i32_i8(x) ((int8_t) (int32_t) x)
#define sext_i32_i16(x) ((int16_t) (int32_t) x)
#define sext_i32_i32(x) ((int32_t) (int32_t) x)
#define sext_i32_i64(x) ((int64_t) (int32_t) x)
#define sext_i64_i8(x) ((int8_t) (int64_t) x)
#define sext_i64_i16(x) ((int16_t) (int64_t) x)
#define sext_i64_i32(x) ((int32_t) (int64_t) x)
#define sext_i64_i64(x) ((int64_t) (int64_t) x)
#define zext_i8_i8(x) ((uint8_t) (uint8_t) x)
#define zext_i8_i16(x) ((uint16_t) (uint8_t) x)
#define zext_i8_i32(x) ((uint32_t) (uint8_t) x)
#define zext_i8_i64(x) ((uint64_t) (uint8_t) x)
#define zext_i16_i8(x) ((uint8_t) (uint16_t) x)
#define zext_i16_i16(x) ((uint16_t) (uint16_t) x)
#define zext_i16_i32(x) ((uint32_t) (uint16_t) x)
#define zext_i16_i64(x) ((uint64_t) (uint16_t) x)
#define zext_i32_i8(x) ((uint8_t) (uint32_t) x)
#define zext_i32_i16(x) ((uint16_t) (uint32_t) x)
#define zext_i32_i32(x) ((uint32_t) (uint32_t) x)
#define zext_i32_i64(x) ((uint64_t) (uint32_t) x)
#define zext_i64_i8(x) ((uint8_t) (uint64_t) x)
#define zext_i64_i16(x) ((uint16_t) (uint64_t) x)
#define zext_i64_i32(x) ((uint32_t) (uint64_t) x)
#define zext_i64_i64(x) ((uint64_t) (uint64_t) x)
static inline float fdiv32(float x, float y)
{
    return x / y;
}
static inline float fadd32(float x, float y)
{
    return x + y;
}
static inline float fsub32(float x, float y)
{
    return x - y;
}
static inline float fmul32(float x, float y)
{
    return x * y;
}
static inline float fmin32(float x, float y)
{
    return x < y ? x : y;
}
static inline float fmax32(float x, float y)
{
    return x < y ? y : x;
}
static inline float fpow32(float x, float y)
{
    return pow(x, y);
}
static inline char cmplt32(float x, float y)
{
    return x < y;
}
static inline char cmple32(float x, float y)
{
    return x <= y;
}
static inline float sitofp_i8_f32(int8_t x)
{
    return x;
}
static inline float sitofp_i16_f32(int16_t x)
{
    return x;
}
static inline float sitofp_i32_f32(int32_t x)
{
    return x;
}
static inline float sitofp_i64_f32(int64_t x)
{
    return x;
}
static inline float uitofp_i8_f32(uint8_t x)
{
    return x;
}
static inline float uitofp_i16_f32(uint16_t x)
{
    return x;
}
static inline float uitofp_i32_f32(uint32_t x)
{
    return x;
}
static inline float uitofp_i64_f32(uint64_t x)
{
    return x;
}
static inline int8_t fptosi_f32_i8(float x)
{
    return x;
}
static inline int16_t fptosi_f32_i16(float x)
{
    return x;
}
static inline int32_t fptosi_f32_i32(float x)
{
    return x;
}
static inline int64_t fptosi_f32_i64(float x)
{
    return x;
}
static inline uint8_t fptoui_f32_i8(float x)
{
    return x;
}
static inline uint16_t fptoui_f32_i16(float x)
{
    return x;
}
static inline uint32_t fptoui_f32_i32(float x)
{
    return x;
}
static inline uint64_t fptoui_f32_i64(float x)
{
    return x;
}
static inline double fdiv64(double x, double y)
{
    return x / y;
}
static inline double fadd64(double x, double y)
{
    return x + y;
}
static inline double fsub64(double x, double y)
{
    return x - y;
}
static inline double fmul64(double x, double y)
{
    return x * y;
}
static inline double fmin64(double x, double y)
{
    return x < y ? x : y;
}
static inline double fmax64(double x, double y)
{
    return x < y ? y : x;
}
static inline double fpow64(double x, double y)
{
    return pow(x, y);
}
static inline char cmplt64(double x, double y)
{
    return x < y;
}
static inline char cmple64(double x, double y)
{
    return x <= y;
}
static inline double sitofp_i8_f64(int8_t x)
{
    return x;
}
static inline double sitofp_i16_f64(int16_t x)
{
    return x;
}
static inline double sitofp_i32_f64(int32_t x)
{
    return x;
}
static inline double sitofp_i64_f64(int64_t x)
{
    return x;
}
static inline double uitofp_i8_f64(uint8_t x)
{
    return x;
}
static inline double uitofp_i16_f64(uint16_t x)
{
    return x;
}
static inline double uitofp_i32_f64(uint32_t x)
{
    return x;
}
static inline double uitofp_i64_f64(uint64_t x)
{
    return x;
}
static inline int8_t fptosi_f64_i8(double x)
{
    return x;
}
static inline int16_t fptosi_f64_i16(double x)
{
    return x;
}
static inline int32_t fptosi_f64_i32(double x)
{
    return x;
}
static inline int64_t fptosi_f64_i64(double x)
{
    return x;
}
static inline uint8_t fptoui_f64_i8(double x)
{
    return x;
}
static inline uint16_t fptoui_f64_i16(double x)
{
    return x;
}
static inline uint32_t fptoui_f64_i32(double x)
{
    return x;
}
static inline uint64_t fptoui_f64_i64(double x)
{
    return x;
}
static inline float fpconv_f32_f32(float x)
{
    return x;
}
static inline double fpconv_f32_f64(float x)
{
    return x;
}
static inline float fpconv_f64_f32(double x)
{
    return x;
}
static inline double fpconv_f64_f64(double x)
{
    return x;
}
static inline float futrts_log32(float x)
{
    return log(x);
}
static inline float futrts_log2_32(float x)
{
    return log2(x);
}
static inline float futrts_log10_32(float x)
{
    return log10(x);
}
static inline float futrts_sqrt32(float x)
{
    return sqrt(x);
}
static inline float futrts_exp32(float x)
{
    return exp(x);
}
static inline float futrts_cos32(float x)
{
    return cos(x);
}
static inline float futrts_sin32(float x)
{
    return sin(x);
}
static inline float futrts_tan32(float x)
{
    return tan(x);
}
static inline float futrts_acos32(float x)
{
    return acos(x);
}
static inline float futrts_asin32(float x)
{
    return asin(x);
}
static inline float futrts_atan32(float x)
{
    return atan(x);
}
static inline float futrts_atan2_32(float x, float y)
{
    return atan2(x, y);
}
static inline float futrts_gamma32(float x)
{
    return tgamma(x);
}
static inline float futrts_lgamma32(float x)
{
    return lgamma(x);
}
static inline float futrts_round32(float x)
{
    return rint(x);
}
static inline char futrts_isnan32(float x)
{
    return isnan(x);
}
static inline char futrts_isinf32(float x)
{
    return isinf(x);
}
static inline int32_t futrts_to_bits32(float x)
{
    union {
        float f;
        int32_t t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline float futrts_from_bits32(int32_t x)
{
    union {
        int32_t f;
        float t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline double futrts_log64(double x)
{
    return log(x);
}
static inline double futrts_log2_64(double x)
{
    return log2(x);
}
static inline double futrts_log10_64(double x)
{
    return log10(x);
}
static inline double futrts_sqrt64(double x)
{
    return sqrt(x);
}
static inline double futrts_exp64(double x)
{
    return exp(x);
}
static inline double futrts_cos64(double x)
{
    return cos(x);
}
static inline double futrts_sin64(double x)
{
    return sin(x);
}
static inline double futrts_tan64(double x)
{
    return tan(x);
}
static inline double futrts_acos64(double x)
{
    return acos(x);
}
static inline double futrts_asin64(double x)
{
    return asin(x);
}
static inline double futrts_atan64(double x)
{
    return atan(x);
}
static inline double futrts_atan2_64(double x, double y)
{
    return atan2(x, y);
}
static inline double futrts_gamma64(double x)
{
    return tgamma(x);
}
static inline double futrts_lgamma64(double x)
{
    return lgamma(x);
}
static inline double futrts_round64(double x)
{
    return rint(x);
}
static inline char futrts_isnan64(double x)
{
    return isnan(x);
}
static inline char futrts_isinf64(double x)
{
    return isinf(x);
}
static inline int64_t futrts_to_bits64(double x)
{
    union {
        double f;
        int64_t t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline double futrts_from_bits64(int64_t x)
{
    union {
        int64_t f;
        double t;
    } p;
    
    p.f = x;
    return p.t;
}
static int futrts_main(struct futhark_context *ctx, bool *out_scalar_out_36233,
                       bool *out_scalar_out_36234, bool *out_scalar_out_36235,
                       bool *out_scalar_out_36236, bool *out_scalar_out_36237,
                       bool *out_scalar_out_36238, bool *out_scalar_out_36239,
                       bool *out_scalar_out_36240, bool *out_scalar_out_36241,
                       bool *out_scalar_out_36242, bool *out_scalar_out_36243,
                       bool *out_scalar_out_36244, bool *out_scalar_out_36245,
                       bool *out_scalar_out_36246, bool *out_scalar_out_36247,
                       bool *out_scalar_out_36248, bool *out_scalar_out_36249,
                       struct memblock X_mem_36153,
                       struct memblock Xsqr_mem_36154,
                       struct memblock Xinv_mem_36155,
                       struct memblock beta0_mem_36156,
                       struct memblock beta_mem_36157,
                       struct memblock y_preds_mem_36158,
                       struct memblock Nss_mem_36159,
                       struct memblock y_errors_mem_36160,
                       struct memblock val_indss_mem_36161,
                       struct memblock hs_mem_36162,
                       struct memblock nss_mem_36163,
                       struct memblock sigmas_mem_36164,
                       struct memblock MO_fsts_mem_36165,
                       struct memblock MOpp_mem_36166,
                       struct memblock MOp_mem_36167,
                       struct memblock breaks_mem_36168,
                       struct memblock means_mem_36169,
                       struct memblock Xseq_mem_36170,
                       struct memblock Xsqrseq_mem_36171,
                       struct memblock Xinvseq_mem_36172,
                       struct memblock beta0seq_mem_36173,
                       struct memblock betaseq_mem_36174,
                       struct memblock y_predsseq_mem_36175,
                       struct memblock Nssseq_mem_36176,
                       struct memblock y_errorsseq_mem_36177,
                       struct memblock val_indssseq_mem_36178,
                       struct memblock hsseq_mem_36179,
                       struct memblock nssseq_mem_36180,
                       struct memblock sigmasseq_mem_36181,
                       struct memblock MO_fstsseq_mem_36182,
                       struct memblock MOppseq_mem_36183,
                       struct memblock MOpseq_mem_36184,
                       struct memblock breaksseq_mem_36185,
                       struct memblock meansseq_mem_36186, int32_t sizze_35491,
                       int32_t sizze_35492, int32_t sizze_35493,
                       int32_t sizze_35494, int32_t sizze_35495,
                       int32_t sizze_35496, int32_t sizze_35497,
                       int32_t sizze_35498, int32_t sizze_35499,
                       int32_t sizze_35500, int32_t sizze_35501,
                       int32_t sizze_35502, int32_t sizze_35503,
                       int32_t sizze_35504, int32_t sizze_35505,
                       int32_t sizze_35506, int32_t sizze_35507,
                       int32_t sizze_35508, int32_t sizze_35509,
                       int32_t sizze_35510, int32_t sizze_35511,
                       int32_t sizze_35512, int32_t sizze_35513,
                       int32_t sizze_35514, int32_t sizze_35515,
                       int32_t sizze_35516, int32_t sizze_35517,
                       int32_t sizze_35518, int32_t sizze_35519,
                       int32_t sizze_35520, int32_t sizze_35521,
                       int32_t sizze_35522, int32_t sizze_35523,
                       int32_t sizze_35524, int32_t sizze_35525,
                       int32_t sizze_35526, int32_t sizze_35527,
                       int32_t sizze_35528, int32_t sizze_35529,
                       int32_t sizze_35530, int32_t sizze_35531,
                       int32_t sizze_35532, int32_t sizze_35533,
                       int32_t sizze_35534, int32_t sizze_35535,
                       int32_t sizze_35536, int32_t sizze_35537,
                       int32_t sizze_35538, int32_t sizze_35539,
                       int32_t sizze_35540, int32_t sizze_35541,
                       int32_t sizze_35542, int32_t sizze_35543,
                       int32_t sizze_35544, int32_t sizze_35545,
                       int32_t sizze_35546, int32_t sizze_35547,
                       int32_t sizze_35548)
{
    bool scalar_out_36187;
    bool scalar_out_36188;
    bool scalar_out_36189;
    bool scalar_out_36190;
    bool scalar_out_36191;
    bool scalar_out_36192;
    bool scalar_out_36193;
    bool scalar_out_36194;
    bool scalar_out_36195;
    bool scalar_out_36196;
    bool scalar_out_36197;
    bool scalar_out_36198;
    bool scalar_out_36199;
    bool scalar_out_36200;
    bool scalar_out_36201;
    bool scalar_out_36202;
    bool scalar_out_36203;
    bool dim_zzero_35583 = 0 == sizze_35520;
    bool dim_zzero_35584 = 0 == sizze_35521;
    bool old_empty_35585 = dim_zzero_35583 || dim_zzero_35584;
    bool dim_zzero_35586 = 0 == sizze_35491;
    bool new_empty_35587 = dim_zzero_35584 || dim_zzero_35586;
    bool both_empty_35588 = old_empty_35585 && new_empty_35587;
    bool dim_match_35589 = sizze_35491 == sizze_35520;
    bool empty_or_match_35590 = both_empty_35588 || dim_match_35589;
    bool empty_or_match_cert_35591;
    
    if (!empty_or_match_35590) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-136:17 -> validation-benchmark.fut:28:12-30:25",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_35592 = 0 == sizze_35492;
    bool both_empty_35593 = dim_zzero_35584 && dim_zzero_35592;
    bool dim_match_35594 = sizze_35492 == sizze_35521;
    bool empty_or_match_35595 = both_empty_35593 || dim_match_35594;
    bool empty_or_match_cert_35596;
    
    if (!empty_or_match_35595) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-136:17 -> validation-benchmark.fut:28:12-30:25 -> /futlib/soacs.fut:51:3-37 -> /futlib/soacs.fut:51:19-23 -> validation-benchmark.fut:29:18-62",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool res_35598;
    bool redout_36041 = 1;
    
    for (int32_t i_36042 = 0; i_36042 < sizze_35491; i_36042++) {
        bool res_35605;
        bool redout_36039 = 1;
        
        for (int32_t i_36040 = 0; i_36040 < sizze_35492; i_36040++) {
            float x_35609 = ((float *) X_mem_36153.mem)[i_36042 * sizze_35492 +
                                                        i_36040];
            float x_35610 = ((float *) Xseq_mem_36170.mem)[i_36042 *
                                                           sizze_35521 +
                                                           i_36040];
            float abs_arg_35611 = x_35609 - x_35610;
            float res_35612 = (float) fabs(abs_arg_35611);
            bool res_35613 = res_35612 < 1.0e-2F;
            bool x_35608 = res_35613 && redout_36039;
            bool redout_tmp_36205 = x_35608;
            
            redout_36039 = redout_tmp_36205;
        }
        res_35605 = redout_36039;
        
        bool x_35601 = res_35605 && redout_36041;
        bool redout_tmp_36204 = x_35601;
        
        redout_36041 = redout_tmp_36204;
    }
    res_35598 = redout_36041;
    
    bool dim_zzero_35614 = 0 == sizze_35522;
    bool dim_zzero_35615 = 0 == sizze_35523;
    bool dim_zzero_35616 = 0 == sizze_35524;
    bool y_35617 = dim_zzero_35615 || dim_zzero_35616;
    bool old_empty_35618 = dim_zzero_35614 || y_35617;
    bool dim_zzero_35619 = 0 == sizze_35493;
    bool new_empty_35620 = y_35617 || dim_zzero_35619;
    bool both_empty_35621 = old_empty_35618 && new_empty_35620;
    bool dim_match_35622 = sizze_35493 == sizze_35522;
    bool empty_or_match_35623 = both_empty_35621 || dim_match_35622;
    bool empty_or_match_cert_35624;
    
    if (!empty_or_match_35623) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-136:17 -> validation-benchmark.fut:35:15-39:39",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_35625 = 0 == sizze_35494;
    bool new_empty_35626 = dim_zzero_35616 || dim_zzero_35625;
    bool both_empty_35627 = y_35617 && new_empty_35626;
    bool dim_match_35628 = sizze_35494 == sizze_35523;
    bool empty_or_match_35629 = both_empty_35627 || dim_match_35628;
    bool empty_or_match_cert_35630;
    
    if (!empty_or_match_35629) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-136:17 -> validation-benchmark.fut:35:15-39:39 -> /futlib/soacs.fut:51:3-37 -> /futlib/soacs.fut:51:19-23 -> validation-benchmark.fut:36:25-38:35",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_35631 = 0 == sizze_35495;
    bool both_empty_35632 = dim_zzero_35616 && dim_zzero_35631;
    bool dim_match_35633 = sizze_35495 == sizze_35524;
    bool empty_or_match_35634 = both_empty_35632 || dim_match_35633;
    bool empty_or_match_cert_35635;
    
    if (!empty_or_match_35634) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-136:17 -> validation-benchmark.fut:35:15-39:39 -> /futlib/soacs.fut:51:3-37 -> /futlib/soacs.fut:51:19-23 -> validation-benchmark.fut:36:25-38:35 -> /futlib/soacs.fut:51:3-37 -> /futlib/soacs.fut:51:19-23 -> validation-benchmark.fut:37:31-80",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool res_35637;
    bool redout_36047 = 1;
    
    for (int32_t i_36048 = 0; i_36048 < sizze_35493; i_36048++) {
        bool res_35644;
        bool redout_36045 = 1;
        
        for (int32_t i_36046 = 0; i_36046 < sizze_35494; i_36046++) {
            bool res_35651;
            bool redout_36043 = 1;
            
            for (int32_t i_36044 = 0; i_36044 < sizze_35495; i_36044++) {
                float x_35655 = ((float *) Xsqr_mem_36154.mem)[i_36048 *
                                                               (sizze_35495 *
                                                                sizze_35494) +
                                                               i_36046 *
                                                               sizze_35495 +
                                                               i_36044];
                float x_35656 = ((float *) Xsqrseq_mem_36171.mem)[i_36048 *
                                                                  (sizze_35524 *
                                                                   sizze_35523) +
                                                                  i_36046 *
                                                                  sizze_35524 +
                                                                  i_36044];
                float abs_arg_35657 = x_35655 - x_35656;
                float res_35658 = (float) fabs(abs_arg_35657);
                bool res_35659 = res_35658 < 0.1F;
                bool x_35654 = res_35659 && redout_36043;
                bool redout_tmp_36208 = x_35654;
                
                redout_36043 = redout_tmp_36208;
            }
            res_35651 = redout_36043;
            
            bool x_35647 = res_35651 && redout_36045;
            bool redout_tmp_36207 = x_35647;
            
            redout_36045 = redout_tmp_36207;
        }
        res_35644 = redout_36045;
        
        bool x_35640 = res_35644 && redout_36047;
        bool redout_tmp_36206 = x_35640;
        
        redout_36047 = redout_tmp_36206;
    }
    res_35637 = redout_36047;
    
    bool dim_zzero_35660 = 0 == sizze_35525;
    bool dim_zzero_35661 = 0 == sizze_35526;
    bool dim_zzero_35662 = 0 == sizze_35527;
    bool y_35663 = dim_zzero_35661 || dim_zzero_35662;
    bool old_empty_35664 = dim_zzero_35660 || y_35663;
    bool dim_zzero_35665 = 0 == sizze_35496;
    bool new_empty_35666 = y_35663 || dim_zzero_35665;
    bool both_empty_35667 = old_empty_35664 && new_empty_35666;
    bool dim_match_35668 = sizze_35496 == sizze_35525;
    bool empty_or_match_35669 = both_empty_35667 || dim_match_35668;
    bool empty_or_match_cert_35670;
    
    if (!empty_or_match_35669) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-136:17 -> validation-benchmark.fut:46:15-50:39",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_35671 = 0 == sizze_35497;
    bool new_empty_35672 = dim_zzero_35662 || dim_zzero_35671;
    bool both_empty_35673 = y_35663 && new_empty_35672;
    bool dim_match_35674 = sizze_35497 == sizze_35526;
    bool empty_or_match_35675 = both_empty_35673 || dim_match_35674;
    bool empty_or_match_cert_35676;
    
    if (!empty_or_match_35675) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-136:17 -> validation-benchmark.fut:46:15-50:39 -> /futlib/soacs.fut:51:3-37 -> /futlib/soacs.fut:51:19-23 -> validation-benchmark.fut:47:25-49:35",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_35677 = 0 == sizze_35498;
    bool both_empty_35678 = dim_zzero_35662 && dim_zzero_35677;
    bool dim_match_35679 = sizze_35498 == sizze_35527;
    bool empty_or_match_35680 = both_empty_35678 || dim_match_35679;
    bool empty_or_match_cert_35681;
    
    if (!empty_or_match_35680) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-136:17 -> validation-benchmark.fut:46:15-50:39 -> /futlib/soacs.fut:51:3-37 -> /futlib/soacs.fut:51:19-23 -> validation-benchmark.fut:47:25-49:35 -> /futlib/soacs.fut:51:3-37 -> /futlib/soacs.fut:51:19-23 -> validation-benchmark.fut:48:31-86",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool res_35683;
    bool redout_36053 = 1;
    
    for (int32_t i_36054 = 0; i_36054 < sizze_35496; i_36054++) {
        bool res_35690;
        bool redout_36051 = 1;
        
        for (int32_t i_36052 = 0; i_36052 < sizze_35497; i_36052++) {
            bool res_35697;
            bool redout_36049 = 1;
            
            for (int32_t i_36050 = 0; i_36050 < sizze_35498; i_36050++) {
                float x_35701 = ((float *) Xinv_mem_36155.mem)[i_36054 *
                                                               (sizze_35498 *
                                                                sizze_35497) +
                                                               i_36052 *
                                                               sizze_35498 +
                                                               i_36050];
                float x_35702 = ((float *) Xinvseq_mem_36172.mem)[i_36054 *
                                                                  (sizze_35527 *
                                                                   sizze_35526) +
                                                                  i_36052 *
                                                                  sizze_35527 +
                                                                  i_36050];
                float abs_arg_35703 = x_35701 - x_35702;
                float res_35704 = (float) fabs(abs_arg_35703);
                bool res_35705 = res_35704 < 1.0e-7F;
                bool x_35700 = res_35705 && redout_36049;
                bool redout_tmp_36211 = x_35700;
                
                redout_36049 = redout_tmp_36211;
            }
            res_35697 = redout_36049;
            
            bool x_35693 = res_35697 && redout_36051;
            bool redout_tmp_36210 = x_35693;
            
            redout_36051 = redout_tmp_36210;
        }
        res_35690 = redout_36051;
        
        bool x_35686 = res_35690 && redout_36053;
        bool redout_tmp_36209 = x_35686;
        
        redout_36053 = redout_tmp_36209;
    }
    res_35683 = redout_36053;
    
    bool dim_zzero_35706 = 0 == sizze_35528;
    bool dim_zzero_35707 = 0 == sizze_35529;
    bool old_empty_35708 = dim_zzero_35706 || dim_zzero_35707;
    bool dim_zzero_35709 = 0 == sizze_35499;
    bool new_empty_35710 = dim_zzero_35707 || dim_zzero_35709;
    bool both_empty_35711 = old_empty_35708 && new_empty_35710;
    bool dim_match_35712 = sizze_35499 == sizze_35528;
    bool empty_or_match_35713 = both_empty_35711 || dim_match_35712;
    bool empty_or_match_cert_35714;
    
    if (!empty_or_match_35713) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-136:17 -> validation-benchmark.fut:57:16-59:33",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_35715 = 0 == sizze_35500;
    bool both_empty_35716 = dim_zzero_35707 && dim_zzero_35715;
    bool dim_match_35717 = sizze_35500 == sizze_35529;
    bool empty_or_match_35718 = both_empty_35716 || dim_match_35717;
    bool empty_or_match_cert_35719;
    
    if (!empty_or_match_35718) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-136:17 -> validation-benchmark.fut:57:16-59:33 -> /futlib/soacs.fut:51:3-37 -> /futlib/soacs.fut:51:19-23 -> validation-benchmark.fut:58:18-61",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool res_35721;
    bool redout_36057 = 1;
    
    for (int32_t i_36058 = 0; i_36058 < sizze_35499; i_36058++) {
        bool res_35728;
        bool redout_36055 = 1;
        
        for (int32_t i_36056 = 0; i_36056 < sizze_35500; i_36056++) {
            float x_35732 = ((float *) beta0_mem_36156.mem)[i_36058 *
                                                            sizze_35500 +
                                                            i_36056];
            float x_35733 = ((float *) beta0seq_mem_36173.mem)[i_36058 *
                                                               sizze_35529 +
                                                               i_36056];
            float abs_arg_35734 = x_35732 - x_35733;
            float res_35735 = (float) fabs(abs_arg_35734);
            bool res_35736 = res_35735 < 1.1F;
            bool x_35731 = res_35736 && redout_36055;
            bool redout_tmp_36213 = x_35731;
            
            redout_36055 = redout_tmp_36213;
        }
        res_35728 = redout_36055;
        
        bool x_35724 = res_35728 && redout_36057;
        bool redout_tmp_36212 = x_35724;
        
        redout_36057 = redout_tmp_36212;
    }
    res_35721 = redout_36057;
    
    bool dim_zzero_35737 = 0 == sizze_35530;
    bool dim_zzero_35738 = 0 == sizze_35531;
    bool old_empty_35739 = dim_zzero_35737 || dim_zzero_35738;
    bool dim_zzero_35740 = 0 == sizze_35501;
    bool new_empty_35741 = dim_zzero_35738 || dim_zzero_35740;
    bool both_empty_35742 = old_empty_35739 && new_empty_35741;
    bool dim_match_35743 = sizze_35501 == sizze_35530;
    bool empty_or_match_35744 = both_empty_35742 || dim_match_35743;
    bool empty_or_match_cert_35745;
    
    if (!empty_or_match_35744) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-136:17 -> validation-benchmark.fut:64:15-66:31",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_35746 = 0 == sizze_35502;
    bool both_empty_35747 = dim_zzero_35738 && dim_zzero_35746;
    bool dim_match_35748 = sizze_35502 == sizze_35531;
    bool empty_or_match_35749 = both_empty_35747 || dim_match_35748;
    bool empty_or_match_cert_35750;
    
    if (!empty_or_match_35749) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-136:17 -> validation-benchmark.fut:64:15-66:31 -> /futlib/soacs.fut:51:3-37 -> /futlib/soacs.fut:51:19-23 -> validation-benchmark.fut:65:18-61",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool res_35752;
    bool redout_36061 = 1;
    
    for (int32_t i_36062 = 0; i_36062 < sizze_35501; i_36062++) {
        bool res_35759;
        bool redout_36059 = 1;
        
        for (int32_t i_36060 = 0; i_36060 < sizze_35502; i_36060++) {
            float x_35763 = ((float *) beta_mem_36157.mem)[i_36062 *
                                                           sizze_35502 +
                                                           i_36060];
            float x_35764 = ((float *) betaseq_mem_36174.mem)[i_36062 *
                                                              sizze_35531 +
                                                              i_36060];
            float abs_arg_35765 = x_35763 - x_35764;
            float res_35766 = (float) fabs(abs_arg_35765);
            bool res_35767 = res_35766 < 0.1F;
            bool x_35762 = res_35767 && redout_36059;
            bool redout_tmp_36215 = x_35762;
            
            redout_36059 = redout_tmp_36215;
        }
        res_35759 = redout_36059;
        
        bool x_35755 = res_35759 && redout_36061;
        bool redout_tmp_36214 = x_35755;
        
        redout_36061 = redout_tmp_36214;
    }
    res_35752 = redout_36061;
    
    bool dim_zzero_35768 = 0 == sizze_35532;
    bool dim_zzero_35769 = 0 == sizze_35533;
    bool old_empty_35770 = dim_zzero_35768 || dim_zzero_35769;
    bool dim_zzero_35771 = 0 == sizze_35503;
    bool new_empty_35772 = dim_zzero_35769 || dim_zzero_35771;
    bool both_empty_35773 = old_empty_35770 && new_empty_35772;
    bool dim_match_35774 = sizze_35503 == sizze_35532;
    bool empty_or_match_35775 = both_empty_35773 || dim_match_35774;
    bool empty_or_match_cert_35776;
    
    if (!empty_or_match_35775) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-136:17 -> validation-benchmark.fut:71:18-73:37",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_35777 = 0 == sizze_35504;
    bool both_empty_35778 = dim_zzero_35769 && dim_zzero_35777;
    bool dim_match_35779 = sizze_35504 == sizze_35533;
    bool empty_or_match_35780 = both_empty_35778 || dim_match_35779;
    bool empty_or_match_cert_35781;
    
    if (!empty_or_match_35780) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-136:17 -> validation-benchmark.fut:71:18-73:37 -> /futlib/soacs.fut:51:3-37 -> /futlib/soacs.fut:51:19-23 -> validation-benchmark.fut:72:18-61",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool res_35783;
    bool redout_36065 = 1;
    
    for (int32_t i_36066 = 0; i_36066 < sizze_35503; i_36066++) {
        bool res_35790;
        bool redout_36063 = 1;
        
        for (int32_t i_36064 = 0; i_36064 < sizze_35504; i_36064++) {
            float x_35794 = ((float *) y_preds_mem_36158.mem)[i_36066 *
                                                              sizze_35504 +
                                                              i_36064];
            float x_35795 = ((float *) y_predsseq_mem_36175.mem)[i_36066 *
                                                                 sizze_35533 +
                                                                 i_36064];
            float abs_arg_35796 = x_35794 - x_35795;
            float res_35797 = (float) fabs(abs_arg_35796);
            bool res_35798 = res_35797 < 0.1F;
            bool x_35793 = res_35798 && redout_36063;
            bool redout_tmp_36217 = x_35793;
            
            redout_36063 = redout_tmp_36217;
        }
        res_35790 = redout_36063;
        
        bool x_35786 = res_35790 && redout_36065;
        bool redout_tmp_36216 = x_35786;
        
        redout_36065 = redout_tmp_36216;
    }
    res_35783 = redout_36065;
    
    bool dim_zzero_35799 = 0 == sizze_35534;
    bool dim_zzero_35800 = 0 == sizze_35505;
    bool both_empty_35801 = dim_zzero_35799 && dim_zzero_35800;
    bool dim_match_35802 = sizze_35505 == sizze_35534;
    bool empty_or_match_35803 = both_empty_35801 || dim_match_35802;
    bool empty_or_match_cert_35804;
    
    if (!empty_or_match_35803) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-136:17 -> validation-benchmark.fut:78:18-66",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool res_35806;
    bool redout_36067 = 1;
    
    for (int32_t i_36068 = 0; i_36068 < sizze_35505; i_36068++) {
        int32_t x_35810 = ((int32_t *) Nss_mem_36159.mem)[i_36068];
        int32_t x_35811 = ((int32_t *) Nssseq_mem_36176.mem)[i_36068];
        int32_t abs_arg_35812 = x_35810 - x_35811;
        int32_t res_35813 = abs(abs_arg_35812);
        bool res_35814 = slt32(res_35813, 1);
        bool x_35809 = res_35814 && redout_36067;
        bool redout_tmp_36218 = x_35809;
        
        redout_36067 = redout_tmp_36218;
    }
    res_35806 = redout_36067;
    
    bool dim_zzero_35815 = 0 == sizze_35535;
    bool dim_zzero_35816 = 0 == sizze_35536;
    bool old_empty_35817 = dim_zzero_35815 || dim_zzero_35816;
    bool dim_zzero_35818 = 0 == sizze_35506;
    bool new_empty_35819 = dim_zzero_35816 || dim_zzero_35818;
    bool both_empty_35820 = old_empty_35817 && new_empty_35819;
    bool dim_match_35821 = sizze_35506 == sizze_35535;
    bool empty_or_match_35822 = both_empty_35820 || dim_match_35821;
    bool empty_or_match_cert_35823;
    
    if (!empty_or_match_35822) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-136:17 -> validation-benchmark.fut:80:19-84:46",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_35824 = 0 == sizze_35507;
    bool both_empty_35825 = dim_zzero_35816 && dim_zzero_35824;
    bool dim_match_35826 = sizze_35507 == sizze_35536;
    bool empty_or_match_35827 = both_empty_35825 || dim_match_35826;
    bool empty_or_match_cert_35828;
    
    if (!empty_or_match_35827) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-136:17 -> validation-benchmark.fut:80:19-84:46 -> /futlib/soacs.fut:51:3-37 -> /futlib/soacs.fut:51:19-23 -> validation-benchmark.fut:81:25-83:73",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool res_35830;
    bool redout_36071 = 1;
    
    for (int32_t i_36072 = 0; i_36072 < sizze_35506; i_36072++) {
        bool res_35837;
        bool redout_36069 = 1;
        
        for (int32_t i_36070 = 0; i_36070 < sizze_35507; i_36070++) {
            float x_35841 = ((float *) y_errors_mem_36160.mem)[i_36072 *
                                                               sizze_35507 +
                                                               i_36070];
            float x_35842 = ((float *) y_errorsseq_mem_36177.mem)[i_36072 *
                                                                  sizze_35536 +
                                                                  i_36070];
            bool res_35843;
            
            res_35843 = futrts_isnan32(x_35841);
            
            float abs_arg_35844 = x_35841 - x_35842;
            float res_35845 = (float) fabs(abs_arg_35844);
            bool res_35846 = res_35845 < 0.1F;
            bool x_35847 = !res_35843;
            bool y_35848 = res_35846 && x_35847;
            bool res_35849 = res_35843 || y_35848;
            bool x_35840 = res_35849 && redout_36069;
            bool redout_tmp_36220 = x_35840;
            
            redout_36069 = redout_tmp_36220;
        }
        res_35837 = redout_36069;
        
        bool x_35833 = res_35837 && redout_36071;
        bool redout_tmp_36219 = x_35833;
        
        redout_36071 = redout_tmp_36219;
    }
    res_35830 = redout_36071;
    
    bool dim_zzero_35850 = 0 == sizze_35537;
    bool dim_zzero_35851 = 0 == sizze_35538;
    bool old_empty_35852 = dim_zzero_35850 || dim_zzero_35851;
    bool dim_zzero_35853 = 0 == sizze_35508;
    bool new_empty_35854 = dim_zzero_35851 || dim_zzero_35853;
    bool both_empty_35855 = old_empty_35852 && new_empty_35854;
    bool dim_match_35856 = sizze_35508 == sizze_35537;
    bool empty_or_match_35857 = both_empty_35855 || dim_match_35856;
    bool empty_or_match_cert_35858;
    
    if (!empty_or_match_35857) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-136:17 -> validation-benchmark.fut:88:20-90:48",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_35859 = 0 == sizze_35509;
    bool both_empty_35860 = dim_zzero_35851 && dim_zzero_35859;
    bool dim_match_35861 = sizze_35509 == sizze_35538;
    bool empty_or_match_35862 = both_empty_35860 || dim_match_35861;
    bool empty_or_match_cert_35863;
    
    if (!empty_or_match_35862) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-136:17 -> validation-benchmark.fut:88:20-90:48 -> /futlib/soacs.fut:51:3-37 -> /futlib/soacs.fut:51:19-23 -> validation-benchmark.fut:89:25-53",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool res_35865;
    bool redout_36075 = 1;
    
    for (int32_t i_36076 = 0; i_36076 < sizze_35508; i_36076++) {
        bool res_35872;
        bool redout_36073 = 1;
        
        for (int32_t i_36074 = 0; i_36074 < sizze_35509; i_36074++) {
            int32_t x_35876 = ((int32_t *) val_indss_mem_36161.mem)[i_36076 *
                                                                    sizze_35509 +
                                                                    i_36074];
            int32_t x_35877 = ((int32_t *) val_indssseq_mem_36178.mem)[i_36076 *
                                                                       sizze_35538 +
                                                                       i_36074];
            bool res_35878 = x_35876 == x_35877;
            bool x_35875 = res_35878 && redout_36073;
            bool redout_tmp_36222 = x_35875;
            
            redout_36073 = redout_tmp_36222;
        }
        res_35872 = redout_36073;
        
        bool x_35868 = res_35872 && redout_36075;
        bool redout_tmp_36221 = x_35868;
        
        redout_36075 = redout_tmp_36221;
    }
    res_35865 = redout_36075;
    
    bool dim_zzero_35879 = 0 == sizze_35539;
    bool dim_zzero_35880 = 0 == sizze_35510;
    bool both_empty_35881 = dim_zzero_35879 && dim_zzero_35880;
    bool dim_match_35882 = sizze_35510 == sizze_35539;
    bool empty_or_match_35883 = both_empty_35881 || dim_match_35882;
    bool empty_or_match_cert_35884;
    
    if (!empty_or_match_35883) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-136:17 -> validation-benchmark.fut:94:17-50",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool res_35886;
    bool redout_36077 = 1;
    
    for (int32_t i_36078 = 0; i_36078 < sizze_35510; i_36078++) {
        int32_t x_35890 = ((int32_t *) hs_mem_36162.mem)[i_36078];
        int32_t x_35891 = ((int32_t *) hsseq_mem_36179.mem)[i_36078];
        bool res_35892 = x_35890 == x_35891;
        bool x_35889 = res_35892 && redout_36077;
        bool redout_tmp_36223 = x_35889;
        
        redout_36077 = redout_tmp_36223;
    }
    res_35886 = redout_36077;
    
    bool dim_zzero_35893 = 0 == sizze_35540;
    bool dim_zzero_35894 = 0 == sizze_35511;
    bool both_empty_35895 = dim_zzero_35893 && dim_zzero_35894;
    bool dim_match_35896 = sizze_35511 == sizze_35540;
    bool empty_or_match_35897 = both_empty_35895 || dim_match_35896;
    bool empty_or_match_cert_35898;
    
    if (!empty_or_match_35897) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-136:17 -> validation-benchmark.fut:95:18-53",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool res_35900;
    bool redout_36079 = 1;
    
    for (int32_t i_36080 = 0; i_36080 < sizze_35511; i_36080++) {
        int32_t x_35904 = ((int32_t *) nss_mem_36163.mem)[i_36080];
        int32_t x_35905 = ((int32_t *) nssseq_mem_36180.mem)[i_36080];
        bool res_35906 = x_35904 == x_35905;
        bool x_35903 = res_35906 && redout_36079;
        bool redout_tmp_36224 = x_35903;
        
        redout_36079 = redout_tmp_36224;
    }
    res_35900 = redout_36079;
    
    bool dim_zzero_35907 = 0 == sizze_35541;
    bool dim_zzero_35908 = 0 == sizze_35512;
    bool both_empty_35909 = dim_zzero_35907 && dim_zzero_35908;
    bool dim_match_35910 = sizze_35512 == sizze_35541;
    bool empty_or_match_35911 = both_empty_35909 || dim_match_35910;
    bool empty_or_match_cert_35912;
    
    if (!empty_or_match_35911) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-136:17 -> validation-benchmark.fut:96:21-75",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool res_35914;
    bool redout_36081 = 1;
    
    for (int32_t i_36082 = 0; i_36082 < sizze_35512; i_36082++) {
        float x_35918 = ((float *) sigmas_mem_36164.mem)[i_36082];
        float x_35919 = ((float *) sigmasseq_mem_36181.mem)[i_36082];
        float abs_arg_35920 = x_35918 - x_35919;
        float res_35921 = (float) fabs(abs_arg_35920);
        bool res_35922 = res_35921 < 1.0F;
        bool x_35917 = res_35922 && redout_36081;
        bool redout_tmp_36225 = x_35917;
        
        redout_36081 = redout_tmp_36225;
    }
    res_35914 = redout_36081;
    
    bool dim_zzero_35923 = 0 == sizze_35542;
    bool dim_zzero_35924 = 0 == sizze_35513;
    bool both_empty_35925 = dim_zzero_35923 && dim_zzero_35924;
    bool dim_match_35926 = sizze_35513 == sizze_35542;
    bool empty_or_match_35927 = both_empty_35925 || dim_match_35926;
    bool empty_or_match_cert_35928;
    
    if (!empty_or_match_35927) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-136:17 -> validation-benchmark.fut:98:22-78",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool res_35930;
    bool redout_36083 = 1;
    
    for (int32_t i_36084 = 0; i_36084 < sizze_35513; i_36084++) {
        float x_35934 = ((float *) MO_fsts_mem_36165.mem)[i_36084];
        float x_35935 = ((float *) MO_fstsseq_mem_36182.mem)[i_36084];
        float abs_arg_35936 = x_35934 - x_35935;
        float res_35937 = (float) fabs(abs_arg_35936);
        bool res_35938 = res_35937 < 1.0F;
        bool x_35933 = res_35938 && redout_36083;
        bool redout_tmp_36226 = x_35933;
        
        redout_36083 = redout_tmp_36226;
    }
    res_35930 = redout_36083;
    
    bool dim_zzero_35939 = 0 == sizze_35543;
    bool dim_zzero_35940 = 0 == sizze_35544;
    bool old_empty_35941 = dim_zzero_35939 || dim_zzero_35940;
    bool dim_zzero_35942 = 0 == sizze_35514;
    bool new_empty_35943 = dim_zzero_35940 || dim_zzero_35942;
    bool both_empty_35944 = old_empty_35941 && new_empty_35943;
    bool dim_match_35945 = sizze_35514 == sizze_35543;
    bool empty_or_match_35946 = both_empty_35944 || dim_match_35945;
    bool empty_or_match_cert_35947;
    
    if (!empty_or_match_35946) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-136:17 -> validation-benchmark.fut:100:15-104:38",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_35948 = 0 == sizze_35515;
    bool both_empty_35949 = dim_zzero_35940 && dim_zzero_35948;
    bool dim_match_35950 = sizze_35515 == sizze_35544;
    bool empty_or_match_35951 = both_empty_35949 || dim_match_35950;
    bool empty_or_match_cert_35952;
    
    if (!empty_or_match_35951) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-136:17 -> validation-benchmark.fut:100:15-104:38 -> /futlib/soacs.fut:51:3-37 -> /futlib/soacs.fut:51:19-23 -> validation-benchmark.fut:101:25-103:73",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool res_35954;
    bool redout_36087 = 1;
    
    for (int32_t i_36088 = 0; i_36088 < sizze_35514; i_36088++) {
        bool res_35961;
        bool redout_36085 = 1;
        
        for (int32_t i_36086 = 0; i_36086 < sizze_35515; i_36086++) {
            float x_35965 = ((float *) MOpp_mem_36166.mem)[i_36088 *
                                                           sizze_35515 +
                                                           i_36086];
            float x_35966 = ((float *) MOppseq_mem_36183.mem)[i_36088 *
                                                              sizze_35544 +
                                                              i_36086];
            bool res_35967;
            
            res_35967 = futrts_isnan32(x_35965);
            
            float abs_arg_35968 = x_35965 - x_35966;
            float res_35969 = (float) fabs(abs_arg_35968);
            bool res_35970 = res_35969 < 0.1F;
            bool x_35971 = !res_35967;
            bool y_35972 = res_35970 && x_35971;
            bool res_35973 = res_35967 || y_35972;
            bool x_35964 = res_35973 && redout_36085;
            bool redout_tmp_36228 = x_35964;
            
            redout_36085 = redout_tmp_36228;
        }
        res_35961 = redout_36085;
        
        bool x_35957 = res_35961 && redout_36087;
        bool redout_tmp_36227 = x_35957;
        
        redout_36087 = redout_tmp_36227;
    }
    res_35954 = redout_36087;
    
    bool dim_zzero_35974 = 0 == sizze_35545;
    bool dim_zzero_35975 = 0 == sizze_35546;
    bool old_empty_35976 = dim_zzero_35974 || dim_zzero_35975;
    bool dim_zzero_35977 = 0 == sizze_35516;
    bool new_empty_35978 = dim_zzero_35975 || dim_zzero_35977;
    bool both_empty_35979 = old_empty_35976 && new_empty_35978;
    bool dim_match_35980 = sizze_35516 == sizze_35545;
    bool empty_or_match_35981 = both_empty_35979 || dim_match_35980;
    bool empty_or_match_cert_35982;
    
    if (!empty_or_match_35981) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-136:17 -> validation-benchmark.fut:108:14-112:36",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_35983 = 0 == sizze_35517;
    bool both_empty_35984 = dim_zzero_35975 && dim_zzero_35983;
    bool dim_match_35985 = sizze_35517 == sizze_35546;
    bool empty_or_match_35986 = both_empty_35984 || dim_match_35985;
    bool empty_or_match_cert_35987;
    
    if (!empty_or_match_35986) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-136:17 -> validation-benchmark.fut:108:14-112:36 -> /futlib/soacs.fut:51:3-37 -> /futlib/soacs.fut:51:19-23 -> validation-benchmark.fut:109:25-111:73",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool res_35989;
    bool redout_36091 = 1;
    
    for (int32_t i_36092 = 0; i_36092 < sizze_35516; i_36092++) {
        bool res_35996;
        bool redout_36089 = 1;
        
        for (int32_t i_36090 = 0; i_36090 < sizze_35517; i_36090++) {
            float x_36000 = ((float *) MOp_mem_36167.mem)[i_36092 *
                                                          sizze_35517 +
                                                          i_36090];
            float x_36001 = ((float *) MOpseq_mem_36184.mem)[i_36092 *
                                                             sizze_35546 +
                                                             i_36090];
            bool res_36002;
            
            res_36002 = futrts_isnan32(x_36000);
            
            float abs_arg_36003 = x_36000 - x_36001;
            float res_36004 = (float) fabs(abs_arg_36003);
            bool res_36005 = res_36004 < 0.1F;
            bool x_36006 = !res_36002;
            bool y_36007 = res_36005 && x_36006;
            bool res_36008 = res_36002 || y_36007;
            bool x_35999 = res_36008 && redout_36089;
            bool redout_tmp_36230 = x_35999;
            
            redout_36089 = redout_tmp_36230;
        }
        res_35996 = redout_36089;
        
        bool x_35992 = res_35996 && redout_36091;
        bool redout_tmp_36229 = x_35992;
        
        redout_36091 = redout_tmp_36229;
    }
    res_35989 = redout_36091;
    
    bool dim_zzero_36009 = 0 == sizze_35547;
    bool dim_zzero_36010 = 0 == sizze_35518;
    bool both_empty_36011 = dim_zzero_36009 && dim_zzero_36010;
    bool dim_match_36012 = sizze_35518 == sizze_35547;
    bool empty_or_match_36013 = both_empty_36011 || dim_match_36012;
    bool empty_or_match_cert_36014;
    
    if (!empty_or_match_36013) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-136:17 -> validation-benchmark.fut:116:21-62",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool res_36016;
    bool redout_36093 = 1;
    
    for (int32_t i_36094 = 0; i_36094 < sizze_35518; i_36094++) {
        int32_t x_36020 = ((int32_t *) breaks_mem_36168.mem)[i_36094];
        int32_t x_36021 = ((int32_t *) breaksseq_mem_36185.mem)[i_36094];
        bool res_36022 = x_36020 == x_36021;
        bool x_36019 = res_36022 && redout_36093;
        bool redout_tmp_36231 = x_36019;
        
        redout_36093 = redout_tmp_36231;
    }
    res_36016 = redout_36093;
    
    bool dim_zzero_36023 = 0 == sizze_35548;
    bool dim_zzero_36024 = 0 == sizze_35519;
    bool both_empty_36025 = dim_zzero_36023 && dim_zzero_36024;
    bool dim_match_36026 = sizze_35519 == sizze_35548;
    bool empty_or_match_36027 = both_empty_36025 || dim_match_36026;
    bool empty_or_match_cert_36028;
    
    if (!empty_or_match_36027) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-136:17 -> validation-benchmark.fut:117:20-72",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool res_36030;
    bool redout_36095 = 1;
    
    for (int32_t i_36096 = 0; i_36096 < sizze_35519; i_36096++) {
        float x_36034 = ((float *) means_mem_36169.mem)[i_36096];
        float x_36035 = ((float *) meansseq_mem_36186.mem)[i_36096];
        float abs_arg_36036 = x_36034 - x_36035;
        float res_36037 = (float) fabs(abs_arg_36036);
        bool res_36038 = res_36037 < 1.0F;
        bool x_36033 = res_36038 && redout_36095;
        bool redout_tmp_36232 = x_36033;
        
        redout_36095 = redout_tmp_36232;
    }
    res_36030 = redout_36095;
    scalar_out_36187 = res_35598;
    scalar_out_36188 = res_35637;
    scalar_out_36189 = res_35683;
    scalar_out_36190 = res_35721;
    scalar_out_36191 = res_35752;
    scalar_out_36192 = res_35783;
    scalar_out_36193 = res_35806;
    scalar_out_36194 = res_35830;
    scalar_out_36195 = res_35865;
    scalar_out_36196 = res_35886;
    scalar_out_36197 = res_35900;
    scalar_out_36198 = res_35914;
    scalar_out_36199 = res_35930;
    scalar_out_36200 = res_35954;
    scalar_out_36201 = res_35989;
    scalar_out_36202 = res_36016;
    scalar_out_36203 = res_36030;
    *out_scalar_out_36233 = scalar_out_36187;
    *out_scalar_out_36234 = scalar_out_36188;
    *out_scalar_out_36235 = scalar_out_36189;
    *out_scalar_out_36236 = scalar_out_36190;
    *out_scalar_out_36237 = scalar_out_36191;
    *out_scalar_out_36238 = scalar_out_36192;
    *out_scalar_out_36239 = scalar_out_36193;
    *out_scalar_out_36240 = scalar_out_36194;
    *out_scalar_out_36241 = scalar_out_36195;
    *out_scalar_out_36242 = scalar_out_36196;
    *out_scalar_out_36243 = scalar_out_36197;
    *out_scalar_out_36244 = scalar_out_36198;
    *out_scalar_out_36245 = scalar_out_36199;
    *out_scalar_out_36246 = scalar_out_36200;
    *out_scalar_out_36247 = scalar_out_36201;
    *out_scalar_out_36248 = scalar_out_36202;
    *out_scalar_out_36249 = scalar_out_36203;
    return 0;
}
struct futhark_f32_1d {
    struct memblock mem;
    int64_t shape[1];
} ;
struct futhark_f32_1d *futhark_new_f32_1d(struct futhark_context *ctx,
                                          float *data, int64_t dim0)
{
    struct futhark_f32_1d *bad = NULL;
    struct futhark_f32_1d *arr =
                          (struct futhark_f32_1d *) malloc(sizeof(struct futhark_f32_1d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc(ctx, &arr->mem, dim0 * sizeof(float), "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    memmove(arr->mem.mem + 0, data + 0, dim0 * sizeof(float));
    lock_unlock(&ctx->lock);
    return arr;
}
struct futhark_f32_1d *futhark_new_raw_f32_1d(struct futhark_context *ctx,
                                              char *data, int offset,
                                              int64_t dim0)
{
    struct futhark_f32_1d *bad = NULL;
    struct futhark_f32_1d *arr =
                          (struct futhark_f32_1d *) malloc(sizeof(struct futhark_f32_1d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc(ctx, &arr->mem, dim0 * sizeof(float), "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    memmove(arr->mem.mem + 0, data + offset, dim0 * sizeof(float));
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_f32_1d(struct futhark_context *ctx, struct futhark_f32_1d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_f32_1d(struct futhark_context *ctx,
                          struct futhark_f32_1d *arr, float *data)
{
    lock_lock(&ctx->lock);
    memmove(data + 0, arr->mem.mem + 0, arr->shape[0] * sizeof(float));
    lock_unlock(&ctx->lock);
    return 0;
}
char *futhark_values_raw_f32_1d(struct futhark_context *ctx,
                                struct futhark_f32_1d *arr)
{
    return arr->mem.mem;
}
int64_t *futhark_shape_f32_1d(struct futhark_context *ctx,
                              struct futhark_f32_1d *arr)
{
    return arr->shape;
}
struct futhark_i32_2d {
    struct memblock mem;
    int64_t shape[2];
} ;
struct futhark_i32_2d *futhark_new_i32_2d(struct futhark_context *ctx,
                                          int32_t *data, int64_t dim0,
                                          int64_t dim1)
{
    struct futhark_i32_2d *bad = NULL;
    struct futhark_i32_2d *arr =
                          (struct futhark_i32_2d *) malloc(sizeof(struct futhark_i32_2d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc(ctx, &arr->mem, dim0 * dim1 * sizeof(int32_t),
                       "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    memmove(arr->mem.mem + 0, data + 0, dim0 * dim1 * sizeof(int32_t));
    lock_unlock(&ctx->lock);
    return arr;
}
struct futhark_i32_2d *futhark_new_raw_i32_2d(struct futhark_context *ctx,
                                              char *data, int offset,
                                              int64_t dim0, int64_t dim1)
{
    struct futhark_i32_2d *bad = NULL;
    struct futhark_i32_2d *arr =
                          (struct futhark_i32_2d *) malloc(sizeof(struct futhark_i32_2d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc(ctx, &arr->mem, dim0 * dim1 * sizeof(int32_t),
                       "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    memmove(arr->mem.mem + 0, data + offset, dim0 * dim1 * sizeof(int32_t));
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_i32_2d(struct futhark_context *ctx, struct futhark_i32_2d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_i32_2d(struct futhark_context *ctx,
                          struct futhark_i32_2d *arr, int32_t *data)
{
    lock_lock(&ctx->lock);
    memmove(data + 0, arr->mem.mem + 0, arr->shape[0] * arr->shape[1] *
            sizeof(int32_t));
    lock_unlock(&ctx->lock);
    return 0;
}
char *futhark_values_raw_i32_2d(struct futhark_context *ctx,
                                struct futhark_i32_2d *arr)
{
    return arr->mem.mem;
}
int64_t *futhark_shape_i32_2d(struct futhark_context *ctx,
                              struct futhark_i32_2d *arr)
{
    return arr->shape;
}
struct futhark_i32_1d {
    struct memblock mem;
    int64_t shape[1];
} ;
struct futhark_i32_1d *futhark_new_i32_1d(struct futhark_context *ctx,
                                          int32_t *data, int64_t dim0)
{
    struct futhark_i32_1d *bad = NULL;
    struct futhark_i32_1d *arr =
                          (struct futhark_i32_1d *) malloc(sizeof(struct futhark_i32_1d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc(ctx, &arr->mem, dim0 * sizeof(int32_t), "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    memmove(arr->mem.mem + 0, data + 0, dim0 * sizeof(int32_t));
    lock_unlock(&ctx->lock);
    return arr;
}
struct futhark_i32_1d *futhark_new_raw_i32_1d(struct futhark_context *ctx,
                                              char *data, int offset,
                                              int64_t dim0)
{
    struct futhark_i32_1d *bad = NULL;
    struct futhark_i32_1d *arr =
                          (struct futhark_i32_1d *) malloc(sizeof(struct futhark_i32_1d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc(ctx, &arr->mem, dim0 * sizeof(int32_t), "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    memmove(arr->mem.mem + 0, data + offset, dim0 * sizeof(int32_t));
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_i32_1d(struct futhark_context *ctx, struct futhark_i32_1d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_i32_1d(struct futhark_context *ctx,
                          struct futhark_i32_1d *arr, int32_t *data)
{
    lock_lock(&ctx->lock);
    memmove(data + 0, arr->mem.mem + 0, arr->shape[0] * sizeof(int32_t));
    lock_unlock(&ctx->lock);
    return 0;
}
char *futhark_values_raw_i32_1d(struct futhark_context *ctx,
                                struct futhark_i32_1d *arr)
{
    return arr->mem.mem;
}
int64_t *futhark_shape_i32_1d(struct futhark_context *ctx,
                              struct futhark_i32_1d *arr)
{
    return arr->shape;
}
struct futhark_f32_3d {
    struct memblock mem;
    int64_t shape[3];
} ;
struct futhark_f32_3d *futhark_new_f32_3d(struct futhark_context *ctx,
                                          float *data, int64_t dim0,
                                          int64_t dim1, int64_t dim2)
{
    struct futhark_f32_3d *bad = NULL;
    struct futhark_f32_3d *arr =
                          (struct futhark_f32_3d *) malloc(sizeof(struct futhark_f32_3d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc(ctx, &arr->mem, dim0 * dim1 * dim2 * sizeof(float),
                       "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    arr->shape[2] = dim2;
    memmove(arr->mem.mem + 0, data + 0, dim0 * dim1 * dim2 * sizeof(float));
    lock_unlock(&ctx->lock);
    return arr;
}
struct futhark_f32_3d *futhark_new_raw_f32_3d(struct futhark_context *ctx,
                                              char *data, int offset,
                                              int64_t dim0, int64_t dim1,
                                              int64_t dim2)
{
    struct futhark_f32_3d *bad = NULL;
    struct futhark_f32_3d *arr =
                          (struct futhark_f32_3d *) malloc(sizeof(struct futhark_f32_3d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc(ctx, &arr->mem, dim0 * dim1 * dim2 * sizeof(float),
                       "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    arr->shape[2] = dim2;
    memmove(arr->mem.mem + 0, data + offset, dim0 * dim1 * dim2 *
            sizeof(float));
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_f32_3d(struct futhark_context *ctx, struct futhark_f32_3d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_f32_3d(struct futhark_context *ctx,
                          struct futhark_f32_3d *arr, float *data)
{
    lock_lock(&ctx->lock);
    memmove(data + 0, arr->mem.mem + 0, arr->shape[0] * arr->shape[1] *
            arr->shape[2] * sizeof(float));
    lock_unlock(&ctx->lock);
    return 0;
}
char *futhark_values_raw_f32_3d(struct futhark_context *ctx,
                                struct futhark_f32_3d *arr)
{
    return arr->mem.mem;
}
int64_t *futhark_shape_f32_3d(struct futhark_context *ctx,
                              struct futhark_f32_3d *arr)
{
    return arr->shape;
}
struct futhark_f32_2d {
    struct memblock mem;
    int64_t shape[2];
} ;
struct futhark_f32_2d *futhark_new_f32_2d(struct futhark_context *ctx,
                                          float *data, int64_t dim0,
                                          int64_t dim1)
{
    struct futhark_f32_2d *bad = NULL;
    struct futhark_f32_2d *arr =
                          (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc(ctx, &arr->mem, dim0 * dim1 * sizeof(float), "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    memmove(arr->mem.mem + 0, data + 0, dim0 * dim1 * sizeof(float));
    lock_unlock(&ctx->lock);
    return arr;
}
struct futhark_f32_2d *futhark_new_raw_f32_2d(struct futhark_context *ctx,
                                              char *data, int offset,
                                              int64_t dim0, int64_t dim1)
{
    struct futhark_f32_2d *bad = NULL;
    struct futhark_f32_2d *arr =
                          (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc(ctx, &arr->mem, dim0 * dim1 * sizeof(float), "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    memmove(arr->mem.mem + 0, data + offset, dim0 * dim1 * sizeof(float));
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_f32_2d(struct futhark_context *ctx, struct futhark_f32_2d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_f32_2d(struct futhark_context *ctx,
                          struct futhark_f32_2d *arr, float *data)
{
    lock_lock(&ctx->lock);
    memmove(data + 0, arr->mem.mem + 0, arr->shape[0] * arr->shape[1] *
            sizeof(float));
    lock_unlock(&ctx->lock);
    return 0;
}
char *futhark_values_raw_f32_2d(struct futhark_context *ctx,
                                struct futhark_f32_2d *arr)
{
    return arr->mem.mem;
}
int64_t *futhark_shape_f32_2d(struct futhark_context *ctx,
                              struct futhark_f32_2d *arr)
{
    return arr->shape;
}
int futhark_entry_main(struct futhark_context *ctx, bool *out0, bool *out1,
                       bool *out2, bool *out3, bool *out4, bool *out5,
                       bool *out6, bool *out7, bool *out8, bool *out9,
                       bool *out10, bool *out11, bool *out12, bool *out13,
                       bool *out14, bool *out15, bool *out16, const
                       struct futhark_f32_2d *in0, const
                       struct futhark_f32_3d *in1, const
                       struct futhark_f32_3d *in2, const
                       struct futhark_f32_2d *in3, const
                       struct futhark_f32_2d *in4, const
                       struct futhark_f32_2d *in5, const
                       struct futhark_i32_1d *in6, const
                       struct futhark_f32_2d *in7, const
                       struct futhark_i32_2d *in8, const
                       struct futhark_i32_1d *in9, const
                       struct futhark_i32_1d *in10, const
                       struct futhark_f32_1d *in11, const
                       struct futhark_f32_1d *in12, const
                       struct futhark_f32_2d *in13, const
                       struct futhark_f32_2d *in14, const
                       struct futhark_i32_1d *in15, const
                       struct futhark_f32_1d *in16, const
                       struct futhark_f32_2d *in17, const
                       struct futhark_f32_3d *in18, const
                       struct futhark_f32_3d *in19, const
                       struct futhark_f32_2d *in20, const
                       struct futhark_f32_2d *in21, const
                       struct futhark_f32_2d *in22, const
                       struct futhark_i32_1d *in23, const
                       struct futhark_f32_2d *in24, const
                       struct futhark_i32_2d *in25, const
                       struct futhark_i32_1d *in26, const
                       struct futhark_i32_1d *in27, const
                       struct futhark_f32_1d *in28, const
                       struct futhark_f32_1d *in29, const
                       struct futhark_f32_2d *in30, const
                       struct futhark_f32_2d *in31, const
                       struct futhark_i32_1d *in32, const
                       struct futhark_f32_1d *in33)
{
    struct memblock X_mem_36153;
    
    X_mem_36153.references = NULL;
    
    struct memblock Xsqr_mem_36154;
    
    Xsqr_mem_36154.references = NULL;
    
    struct memblock Xinv_mem_36155;
    
    Xinv_mem_36155.references = NULL;
    
    struct memblock beta0_mem_36156;
    
    beta0_mem_36156.references = NULL;
    
    struct memblock beta_mem_36157;
    
    beta_mem_36157.references = NULL;
    
    struct memblock y_preds_mem_36158;
    
    y_preds_mem_36158.references = NULL;
    
    struct memblock Nss_mem_36159;
    
    Nss_mem_36159.references = NULL;
    
    struct memblock y_errors_mem_36160;
    
    y_errors_mem_36160.references = NULL;
    
    struct memblock val_indss_mem_36161;
    
    val_indss_mem_36161.references = NULL;
    
    struct memblock hs_mem_36162;
    
    hs_mem_36162.references = NULL;
    
    struct memblock nss_mem_36163;
    
    nss_mem_36163.references = NULL;
    
    struct memblock sigmas_mem_36164;
    
    sigmas_mem_36164.references = NULL;
    
    struct memblock MO_fsts_mem_36165;
    
    MO_fsts_mem_36165.references = NULL;
    
    struct memblock MOpp_mem_36166;
    
    MOpp_mem_36166.references = NULL;
    
    struct memblock MOp_mem_36167;
    
    MOp_mem_36167.references = NULL;
    
    struct memblock breaks_mem_36168;
    
    breaks_mem_36168.references = NULL;
    
    struct memblock means_mem_36169;
    
    means_mem_36169.references = NULL;
    
    struct memblock Xseq_mem_36170;
    
    Xseq_mem_36170.references = NULL;
    
    struct memblock Xsqrseq_mem_36171;
    
    Xsqrseq_mem_36171.references = NULL;
    
    struct memblock Xinvseq_mem_36172;
    
    Xinvseq_mem_36172.references = NULL;
    
    struct memblock beta0seq_mem_36173;
    
    beta0seq_mem_36173.references = NULL;
    
    struct memblock betaseq_mem_36174;
    
    betaseq_mem_36174.references = NULL;
    
    struct memblock y_predsseq_mem_36175;
    
    y_predsseq_mem_36175.references = NULL;
    
    struct memblock Nssseq_mem_36176;
    
    Nssseq_mem_36176.references = NULL;
    
    struct memblock y_errorsseq_mem_36177;
    
    y_errorsseq_mem_36177.references = NULL;
    
    struct memblock val_indssseq_mem_36178;
    
    val_indssseq_mem_36178.references = NULL;
    
    struct memblock hsseq_mem_36179;
    
    hsseq_mem_36179.references = NULL;
    
    struct memblock nssseq_mem_36180;
    
    nssseq_mem_36180.references = NULL;
    
    struct memblock sigmasseq_mem_36181;
    
    sigmasseq_mem_36181.references = NULL;
    
    struct memblock MO_fstsseq_mem_36182;
    
    MO_fstsseq_mem_36182.references = NULL;
    
    struct memblock MOppseq_mem_36183;
    
    MOppseq_mem_36183.references = NULL;
    
    struct memblock MOpseq_mem_36184;
    
    MOpseq_mem_36184.references = NULL;
    
    struct memblock breaksseq_mem_36185;
    
    breaksseq_mem_36185.references = NULL;
    
    struct memblock meansseq_mem_36186;
    
    meansseq_mem_36186.references = NULL;
    
    int32_t sizze_35491;
    int32_t sizze_35492;
    int32_t sizze_35493;
    int32_t sizze_35494;
    int32_t sizze_35495;
    int32_t sizze_35496;
    int32_t sizze_35497;
    int32_t sizze_35498;
    int32_t sizze_35499;
    int32_t sizze_35500;
    int32_t sizze_35501;
    int32_t sizze_35502;
    int32_t sizze_35503;
    int32_t sizze_35504;
    int32_t sizze_35505;
    int32_t sizze_35506;
    int32_t sizze_35507;
    int32_t sizze_35508;
    int32_t sizze_35509;
    int32_t sizze_35510;
    int32_t sizze_35511;
    int32_t sizze_35512;
    int32_t sizze_35513;
    int32_t sizze_35514;
    int32_t sizze_35515;
    int32_t sizze_35516;
    int32_t sizze_35517;
    int32_t sizze_35518;
    int32_t sizze_35519;
    int32_t sizze_35520;
    int32_t sizze_35521;
    int32_t sizze_35522;
    int32_t sizze_35523;
    int32_t sizze_35524;
    int32_t sizze_35525;
    int32_t sizze_35526;
    int32_t sizze_35527;
    int32_t sizze_35528;
    int32_t sizze_35529;
    int32_t sizze_35530;
    int32_t sizze_35531;
    int32_t sizze_35532;
    int32_t sizze_35533;
    int32_t sizze_35534;
    int32_t sizze_35535;
    int32_t sizze_35536;
    int32_t sizze_35537;
    int32_t sizze_35538;
    int32_t sizze_35539;
    int32_t sizze_35540;
    int32_t sizze_35541;
    int32_t sizze_35542;
    int32_t sizze_35543;
    int32_t sizze_35544;
    int32_t sizze_35545;
    int32_t sizze_35546;
    int32_t sizze_35547;
    int32_t sizze_35548;
    bool scalar_out_36187;
    bool scalar_out_36188;
    bool scalar_out_36189;
    bool scalar_out_36190;
    bool scalar_out_36191;
    bool scalar_out_36192;
    bool scalar_out_36193;
    bool scalar_out_36194;
    bool scalar_out_36195;
    bool scalar_out_36196;
    bool scalar_out_36197;
    bool scalar_out_36198;
    bool scalar_out_36199;
    bool scalar_out_36200;
    bool scalar_out_36201;
    bool scalar_out_36202;
    bool scalar_out_36203;
    
    lock_lock(&ctx->lock);
    X_mem_36153 = in0->mem;
    sizze_35491 = in0->shape[0];
    sizze_35492 = in0->shape[1];
    Xsqr_mem_36154 = in1->mem;
    sizze_35493 = in1->shape[0];
    sizze_35494 = in1->shape[1];
    sizze_35495 = in1->shape[2];
    Xinv_mem_36155 = in2->mem;
    sizze_35496 = in2->shape[0];
    sizze_35497 = in2->shape[1];
    sizze_35498 = in2->shape[2];
    beta0_mem_36156 = in3->mem;
    sizze_35499 = in3->shape[0];
    sizze_35500 = in3->shape[1];
    beta_mem_36157 = in4->mem;
    sizze_35501 = in4->shape[0];
    sizze_35502 = in4->shape[1];
    y_preds_mem_36158 = in5->mem;
    sizze_35503 = in5->shape[0];
    sizze_35504 = in5->shape[1];
    Nss_mem_36159 = in6->mem;
    sizze_35505 = in6->shape[0];
    y_errors_mem_36160 = in7->mem;
    sizze_35506 = in7->shape[0];
    sizze_35507 = in7->shape[1];
    val_indss_mem_36161 = in8->mem;
    sizze_35508 = in8->shape[0];
    sizze_35509 = in8->shape[1];
    hs_mem_36162 = in9->mem;
    sizze_35510 = in9->shape[0];
    nss_mem_36163 = in10->mem;
    sizze_35511 = in10->shape[0];
    sigmas_mem_36164 = in11->mem;
    sizze_35512 = in11->shape[0];
    MO_fsts_mem_36165 = in12->mem;
    sizze_35513 = in12->shape[0];
    MOpp_mem_36166 = in13->mem;
    sizze_35514 = in13->shape[0];
    sizze_35515 = in13->shape[1];
    MOp_mem_36167 = in14->mem;
    sizze_35516 = in14->shape[0];
    sizze_35517 = in14->shape[1];
    breaks_mem_36168 = in15->mem;
    sizze_35518 = in15->shape[0];
    means_mem_36169 = in16->mem;
    sizze_35519 = in16->shape[0];
    Xseq_mem_36170 = in17->mem;
    sizze_35520 = in17->shape[0];
    sizze_35521 = in17->shape[1];
    Xsqrseq_mem_36171 = in18->mem;
    sizze_35522 = in18->shape[0];
    sizze_35523 = in18->shape[1];
    sizze_35524 = in18->shape[2];
    Xinvseq_mem_36172 = in19->mem;
    sizze_35525 = in19->shape[0];
    sizze_35526 = in19->shape[1];
    sizze_35527 = in19->shape[2];
    beta0seq_mem_36173 = in20->mem;
    sizze_35528 = in20->shape[0];
    sizze_35529 = in20->shape[1];
    betaseq_mem_36174 = in21->mem;
    sizze_35530 = in21->shape[0];
    sizze_35531 = in21->shape[1];
    y_predsseq_mem_36175 = in22->mem;
    sizze_35532 = in22->shape[0];
    sizze_35533 = in22->shape[1];
    Nssseq_mem_36176 = in23->mem;
    sizze_35534 = in23->shape[0];
    y_errorsseq_mem_36177 = in24->mem;
    sizze_35535 = in24->shape[0];
    sizze_35536 = in24->shape[1];
    val_indssseq_mem_36178 = in25->mem;
    sizze_35537 = in25->shape[0];
    sizze_35538 = in25->shape[1];
    hsseq_mem_36179 = in26->mem;
    sizze_35539 = in26->shape[0];
    nssseq_mem_36180 = in27->mem;
    sizze_35540 = in27->shape[0];
    sigmasseq_mem_36181 = in28->mem;
    sizze_35541 = in28->shape[0];
    MO_fstsseq_mem_36182 = in29->mem;
    sizze_35542 = in29->shape[0];
    MOppseq_mem_36183 = in30->mem;
    sizze_35543 = in30->shape[0];
    sizze_35544 = in30->shape[1];
    MOpseq_mem_36184 = in31->mem;
    sizze_35545 = in31->shape[0];
    sizze_35546 = in31->shape[1];
    breaksseq_mem_36185 = in32->mem;
    sizze_35547 = in32->shape[0];
    meansseq_mem_36186 = in33->mem;
    sizze_35548 = in33->shape[0];
    
    int ret = futrts_main(ctx, &scalar_out_36187, &scalar_out_36188,
                          &scalar_out_36189, &scalar_out_36190,
                          &scalar_out_36191, &scalar_out_36192,
                          &scalar_out_36193, &scalar_out_36194,
                          &scalar_out_36195, &scalar_out_36196,
                          &scalar_out_36197, &scalar_out_36198,
                          &scalar_out_36199, &scalar_out_36200,
                          &scalar_out_36201, &scalar_out_36202,
                          &scalar_out_36203, X_mem_36153, Xsqr_mem_36154,
                          Xinv_mem_36155, beta0_mem_36156, beta_mem_36157,
                          y_preds_mem_36158, Nss_mem_36159, y_errors_mem_36160,
                          val_indss_mem_36161, hs_mem_36162, nss_mem_36163,
                          sigmas_mem_36164, MO_fsts_mem_36165, MOpp_mem_36166,
                          MOp_mem_36167, breaks_mem_36168, means_mem_36169,
                          Xseq_mem_36170, Xsqrseq_mem_36171, Xinvseq_mem_36172,
                          beta0seq_mem_36173, betaseq_mem_36174,
                          y_predsseq_mem_36175, Nssseq_mem_36176,
                          y_errorsseq_mem_36177, val_indssseq_mem_36178,
                          hsseq_mem_36179, nssseq_mem_36180,
                          sigmasseq_mem_36181, MO_fstsseq_mem_36182,
                          MOppseq_mem_36183, MOpseq_mem_36184,
                          breaksseq_mem_36185, meansseq_mem_36186, sizze_35491,
                          sizze_35492, sizze_35493, sizze_35494, sizze_35495,
                          sizze_35496, sizze_35497, sizze_35498, sizze_35499,
                          sizze_35500, sizze_35501, sizze_35502, sizze_35503,
                          sizze_35504, sizze_35505, sizze_35506, sizze_35507,
                          sizze_35508, sizze_35509, sizze_35510, sizze_35511,
                          sizze_35512, sizze_35513, sizze_35514, sizze_35515,
                          sizze_35516, sizze_35517, sizze_35518, sizze_35519,
                          sizze_35520, sizze_35521, sizze_35522, sizze_35523,
                          sizze_35524, sizze_35525, sizze_35526, sizze_35527,
                          sizze_35528, sizze_35529, sizze_35530, sizze_35531,
                          sizze_35532, sizze_35533, sizze_35534, sizze_35535,
                          sizze_35536, sizze_35537, sizze_35538, sizze_35539,
                          sizze_35540, sizze_35541, sizze_35542, sizze_35543,
                          sizze_35544, sizze_35545, sizze_35546, sizze_35547,
                          sizze_35548);
    
    if (ret == 0) {
        *out0 = scalar_out_36187;
        *out1 = scalar_out_36188;
        *out2 = scalar_out_36189;
        *out3 = scalar_out_36190;
        *out4 = scalar_out_36191;
        *out5 = scalar_out_36192;
        *out6 = scalar_out_36193;
        *out7 = scalar_out_36194;
        *out8 = scalar_out_36195;
        *out9 = scalar_out_36196;
        *out10 = scalar_out_36197;
        *out11 = scalar_out_36198;
        *out12 = scalar_out_36199;
        *out13 = scalar_out_36200;
        *out14 = scalar_out_36201;
        *out15 = scalar_out_36202;
        *out16 = scalar_out_36203;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
