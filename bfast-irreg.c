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

int futhark_entry_main(struct futhark_context *ctx,
                       struct futhark_f32_2d **out0,
                       struct futhark_f32_2d **out1,
                       struct futhark_i32_1d **out2,
                       struct futhark_f32_1d **out3,
                       struct futhark_f32_2d **out4,
                       struct futhark_f32_3d **out5,
                       struct futhark_f32_3d **out6,
                       struct futhark_f32_2d **out7,
                       struct futhark_f32_2d **out8,
                       struct futhark_f32_2d **out9,
                       struct futhark_f32_2d **out10,
                       struct futhark_i32_1d **out11,
                       struct futhark_i32_2d **out12,
                       struct futhark_i32_1d **out13,
                       struct futhark_i32_1d **out14,
                       struct futhark_f32_1d **out15,
                       struct futhark_f32_1d **out16, const int32_t in0, const
                       int32_t in1, const int32_t in2, const float in3, const
                       float in4, const float in5, const
                       struct futhark_i32_1d *in6, const
                       struct futhark_f32_2d *in7);

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
    
    int32_t read_value_54139;
    
    if (read_scalar(&i32_info, &read_value_54139) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 0,
              i32_info.type_name, strerror(errno));
    
    int32_t read_value_54140;
    
    if (read_scalar(&i32_info, &read_value_54140) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 1,
              i32_info.type_name, strerror(errno));
    
    int32_t read_value_54141;
    
    if (read_scalar(&i32_info, &read_value_54141) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 2,
              i32_info.type_name, strerror(errno));
    
    float read_value_54142;
    
    if (read_scalar(&f32_info, &read_value_54142) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 3,
              f32_info.type_name, strerror(errno));
    
    float read_value_54143;
    
    if (read_scalar(&f32_info, &read_value_54143) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 4,
              f32_info.type_name, strerror(errno));
    
    float read_value_54144;
    
    if (read_scalar(&f32_info, &read_value_54144) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 5,
              f32_info.type_name, strerror(errno));
    
    struct futhark_i32_1d *read_value_54145;
    int64_t read_shape_54146[1];
    int32_t *read_arr_54147 = NULL;
    
    errno = 0;
    if (read_array(&i32_info, (void **) &read_arr_54147, read_shape_54146, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 6, "[]",
              i32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_54148;
    int64_t read_shape_54149[2];
    float *read_arr_54150 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_54150, read_shape_54149, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 7, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *result_54151;
    struct futhark_f32_2d *result_54152;
    struct futhark_i32_1d *result_54153;
    struct futhark_f32_1d *result_54154;
    struct futhark_f32_2d *result_54155;
    struct futhark_f32_3d *result_54156;
    struct futhark_f32_3d *result_54157;
    struct futhark_f32_2d *result_54158;
    struct futhark_f32_2d *result_54159;
    struct futhark_f32_2d *result_54160;
    struct futhark_f32_2d *result_54161;
    struct futhark_i32_1d *result_54162;
    struct futhark_i32_2d *result_54163;
    struct futhark_i32_1d *result_54164;
    struct futhark_i32_1d *result_54165;
    struct futhark_f32_1d *result_54166;
    struct futhark_f32_1d *result_54167;
    
    if (perform_warmup) {
        time_runs = 0;
        
        int r;
        
        ;
        ;
        ;
        ;
        ;
        ;
        assert((read_value_54145 = futhark_new_i32_1d(ctx, read_arr_54147,
                                                      read_shape_54146[0])) !=
            0);
        assert((read_value_54148 = futhark_new_f32_2d(ctx, read_arr_54150,
                                                      read_shape_54149[0],
                                                      read_shape_54149[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_main(ctx, &result_54151, &result_54152, &result_54153,
                               &result_54154, &result_54155, &result_54156,
                               &result_54157, &result_54158, &result_54159,
                               &result_54160, &result_54161, &result_54162,
                               &result_54163, &result_54164, &result_54165,
                               &result_54166, &result_54167, read_value_54139,
                               read_value_54140, read_value_54141,
                               read_value_54142, read_value_54143,
                               read_value_54144, read_value_54145,
                               read_value_54148);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        ;
        ;
        ;
        ;
        ;
        ;
        assert(futhark_free_i32_1d(ctx, read_value_54145) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_54148) == 0);
        assert(futhark_free_f32_2d(ctx, result_54151) == 0);
        assert(futhark_free_f32_2d(ctx, result_54152) == 0);
        assert(futhark_free_i32_1d(ctx, result_54153) == 0);
        assert(futhark_free_f32_1d(ctx, result_54154) == 0);
        assert(futhark_free_f32_2d(ctx, result_54155) == 0);
        assert(futhark_free_f32_3d(ctx, result_54156) == 0);
        assert(futhark_free_f32_3d(ctx, result_54157) == 0);
        assert(futhark_free_f32_2d(ctx, result_54158) == 0);
        assert(futhark_free_f32_2d(ctx, result_54159) == 0);
        assert(futhark_free_f32_2d(ctx, result_54160) == 0);
        assert(futhark_free_f32_2d(ctx, result_54161) == 0);
        assert(futhark_free_i32_1d(ctx, result_54162) == 0);
        assert(futhark_free_i32_2d(ctx, result_54163) == 0);
        assert(futhark_free_i32_1d(ctx, result_54164) == 0);
        assert(futhark_free_i32_1d(ctx, result_54165) == 0);
        assert(futhark_free_f32_1d(ctx, result_54166) == 0);
        assert(futhark_free_f32_1d(ctx, result_54167) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        int r;
        
        ;
        ;
        ;
        ;
        ;
        ;
        assert((read_value_54145 = futhark_new_i32_1d(ctx, read_arr_54147,
                                                      read_shape_54146[0])) !=
            0);
        assert((read_value_54148 = futhark_new_f32_2d(ctx, read_arr_54150,
                                                      read_shape_54149[0],
                                                      read_shape_54149[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_main(ctx, &result_54151, &result_54152, &result_54153,
                               &result_54154, &result_54155, &result_54156,
                               &result_54157, &result_54158, &result_54159,
                               &result_54160, &result_54161, &result_54162,
                               &result_54163, &result_54164, &result_54165,
                               &result_54166, &result_54167, read_value_54139,
                               read_value_54140, read_value_54141,
                               read_value_54142, read_value_54143,
                               read_value_54144, read_value_54145,
                               read_value_54148);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        ;
        ;
        ;
        ;
        ;
        ;
        assert(futhark_free_i32_1d(ctx, read_value_54145) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_54148) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_2d(ctx, result_54151) == 0);
            assert(futhark_free_f32_2d(ctx, result_54152) == 0);
            assert(futhark_free_i32_1d(ctx, result_54153) == 0);
            assert(futhark_free_f32_1d(ctx, result_54154) == 0);
            assert(futhark_free_f32_2d(ctx, result_54155) == 0);
            assert(futhark_free_f32_3d(ctx, result_54156) == 0);
            assert(futhark_free_f32_3d(ctx, result_54157) == 0);
            assert(futhark_free_f32_2d(ctx, result_54158) == 0);
            assert(futhark_free_f32_2d(ctx, result_54159) == 0);
            assert(futhark_free_f32_2d(ctx, result_54160) == 0);
            assert(futhark_free_f32_2d(ctx, result_54161) == 0);
            assert(futhark_free_i32_1d(ctx, result_54162) == 0);
            assert(futhark_free_i32_2d(ctx, result_54163) == 0);
            assert(futhark_free_i32_1d(ctx, result_54164) == 0);
            assert(futhark_free_i32_1d(ctx, result_54165) == 0);
            assert(futhark_free_f32_1d(ctx, result_54166) == 0);
            assert(futhark_free_f32_1d(ctx, result_54167) == 0);
        }
    }
    ;
    ;
    ;
    ;
    ;
    ;
    free(read_arr_54147);
    free(read_arr_54150);
    if (binary_output)
        set_binary_mode(stdout);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_2d(ctx,
                                                                result_54151)[0] *
                            futhark_shape_f32_2d(ctx, result_54151)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_2d(ctx, result_54151, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_2d(ctx, result_54151), 2);
        free(arr);
    }
    printf("\n");
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_2d(ctx,
                                                                result_54152)[0] *
                            futhark_shape_f32_2d(ctx, result_54152)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_2d(ctx, result_54152, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_2d(ctx, result_54152), 2);
        free(arr);
    }
    printf("\n");
    {
        int32_t *arr = calloc(sizeof(int32_t), futhark_shape_i32_1d(ctx,
                                                                    result_54153)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_i32_1d(ctx, result_54153, arr) == 0);
        write_array(stdout, binary_output, &i32_info, arr,
                    futhark_shape_i32_1d(ctx, result_54153), 1);
        free(arr);
    }
    printf("\n");
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_1d(ctx,
                                                                result_54154)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_1d(ctx, result_54154, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_1d(ctx, result_54154), 1);
        free(arr);
    }
    printf("\n");
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_2d(ctx,
                                                                result_54155)[0] *
                            futhark_shape_f32_2d(ctx, result_54155)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_2d(ctx, result_54155, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_2d(ctx, result_54155), 2);
        free(arr);
    }
    printf("\n");
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_3d(ctx,
                                                                result_54156)[0] *
                            futhark_shape_f32_3d(ctx, result_54156)[1] *
                            futhark_shape_f32_3d(ctx, result_54156)[2]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_3d(ctx, result_54156, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_3d(ctx, result_54156), 3);
        free(arr);
    }
    printf("\n");
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_3d(ctx,
                                                                result_54157)[0] *
                            futhark_shape_f32_3d(ctx, result_54157)[1] *
                            futhark_shape_f32_3d(ctx, result_54157)[2]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_3d(ctx, result_54157, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_3d(ctx, result_54157), 3);
        free(arr);
    }
    printf("\n");
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_2d(ctx,
                                                                result_54158)[0] *
                            futhark_shape_f32_2d(ctx, result_54158)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_2d(ctx, result_54158, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_2d(ctx, result_54158), 2);
        free(arr);
    }
    printf("\n");
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_2d(ctx,
                                                                result_54159)[0] *
                            futhark_shape_f32_2d(ctx, result_54159)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_2d(ctx, result_54159, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_2d(ctx, result_54159), 2);
        free(arr);
    }
    printf("\n");
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_2d(ctx,
                                                                result_54160)[0] *
                            futhark_shape_f32_2d(ctx, result_54160)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_2d(ctx, result_54160, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_2d(ctx, result_54160), 2);
        free(arr);
    }
    printf("\n");
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_2d(ctx,
                                                                result_54161)[0] *
                            futhark_shape_f32_2d(ctx, result_54161)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_2d(ctx, result_54161, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_2d(ctx, result_54161), 2);
        free(arr);
    }
    printf("\n");
    {
        int32_t *arr = calloc(sizeof(int32_t), futhark_shape_i32_1d(ctx,
                                                                    result_54162)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_i32_1d(ctx, result_54162, arr) == 0);
        write_array(stdout, binary_output, &i32_info, arr,
                    futhark_shape_i32_1d(ctx, result_54162), 1);
        free(arr);
    }
    printf("\n");
    {
        int32_t *arr = calloc(sizeof(int32_t), futhark_shape_i32_2d(ctx,
                                                                    result_54163)[0] *
                              futhark_shape_i32_2d(ctx, result_54163)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_i32_2d(ctx, result_54163, arr) == 0);
        write_array(stdout, binary_output, &i32_info, arr,
                    futhark_shape_i32_2d(ctx, result_54163), 2);
        free(arr);
    }
    printf("\n");
    {
        int32_t *arr = calloc(sizeof(int32_t), futhark_shape_i32_1d(ctx,
                                                                    result_54164)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_i32_1d(ctx, result_54164, arr) == 0);
        write_array(stdout, binary_output, &i32_info, arr,
                    futhark_shape_i32_1d(ctx, result_54164), 1);
        free(arr);
    }
    printf("\n");
    {
        int32_t *arr = calloc(sizeof(int32_t), futhark_shape_i32_1d(ctx,
                                                                    result_54165)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_i32_1d(ctx, result_54165, arr) == 0);
        write_array(stdout, binary_output, &i32_info, arr,
                    futhark_shape_i32_1d(ctx, result_54165), 1);
        free(arr);
    }
    printf("\n");
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_1d(ctx,
                                                                result_54166)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_1d(ctx, result_54166, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_1d(ctx, result_54166), 1);
        free(arr);
    }
    printf("\n");
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_1d(ctx,
                                                                result_54167)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_1d(ctx, result_54167, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_1d(ctx, result_54167), 1);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_2d(ctx, result_54151) == 0);
    assert(futhark_free_f32_2d(ctx, result_54152) == 0);
    assert(futhark_free_i32_1d(ctx, result_54153) == 0);
    assert(futhark_free_f32_1d(ctx, result_54154) == 0);
    assert(futhark_free_f32_2d(ctx, result_54155) == 0);
    assert(futhark_free_f32_3d(ctx, result_54156) == 0);
    assert(futhark_free_f32_3d(ctx, result_54157) == 0);
    assert(futhark_free_f32_2d(ctx, result_54158) == 0);
    assert(futhark_free_f32_2d(ctx, result_54159) == 0);
    assert(futhark_free_f32_2d(ctx, result_54160) == 0);
    assert(futhark_free_f32_2d(ctx, result_54161) == 0);
    assert(futhark_free_i32_1d(ctx, result_54162) == 0);
    assert(futhark_free_i32_2d(ctx, result_54163) == 0);
    assert(futhark_free_i32_1d(ctx, result_54164) == 0);
    assert(futhark_free_i32_1d(ctx, result_54165) == 0);
    assert(futhark_free_f32_1d(ctx, result_54166) == 0);
    assert(futhark_free_f32_1d(ctx, result_54167) == 0);
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
static int futrts_main(struct futhark_context *ctx,
                       struct memblock *out_mem_p_54093,
                       int32_t *out_out_arrsizze_54094,
                       int32_t *out_out_arrsizze_54095,
                       struct memblock *out_mem_p_54096,
                       int32_t *out_out_arrsizze_54097,
                       int32_t *out_out_arrsizze_54098,
                       struct memblock *out_mem_p_54099,
                       int32_t *out_out_arrsizze_54100,
                       struct memblock *out_mem_p_54101,
                       int32_t *out_out_arrsizze_54102,
                       struct memblock *out_mem_p_54103,
                       int32_t *out_out_arrsizze_54104,
                       int32_t *out_out_arrsizze_54105,
                       struct memblock *out_mem_p_54106,
                       int32_t *out_out_arrsizze_54107,
                       int32_t *out_out_arrsizze_54108,
                       int32_t *out_out_arrsizze_54109,
                       struct memblock *out_mem_p_54110,
                       int32_t *out_out_arrsizze_54111,
                       int32_t *out_out_arrsizze_54112,
                       int32_t *out_out_arrsizze_54113,
                       struct memblock *out_mem_p_54114,
                       int32_t *out_out_arrsizze_54115,
                       int32_t *out_out_arrsizze_54116,
                       struct memblock *out_mem_p_54117,
                       int32_t *out_out_arrsizze_54118,
                       int32_t *out_out_arrsizze_54119,
                       struct memblock *out_mem_p_54120,
                       int32_t *out_out_arrsizze_54121,
                       int32_t *out_out_arrsizze_54122,
                       struct memblock *out_mem_p_54123,
                       int32_t *out_out_arrsizze_54124,
                       int32_t *out_out_arrsizze_54125,
                       struct memblock *out_mem_p_54126,
                       int32_t *out_out_arrsizze_54127,
                       struct memblock *out_mem_p_54128,
                       int32_t *out_out_arrsizze_54129,
                       int32_t *out_out_arrsizze_54130,
                       struct memblock *out_mem_p_54131,
                       int32_t *out_out_arrsizze_54132,
                       struct memblock *out_mem_p_54133,
                       int32_t *out_out_arrsizze_54134,
                       struct memblock *out_mem_p_54135,
                       int32_t *out_out_arrsizze_54136,
                       struct memblock *out_mem_p_54137,
                       int32_t *out_out_arrsizze_54138,
                       struct memblock mappingindices_mem_53689,
                       struct memblock images_mem_53690, int32_t N_52750,
                       int32_t m_52751, int32_t N_52752, int32_t trend_52753,
                       int32_t k_52754, int32_t n_52755, float freq_52756,
                       float hfrac_52757, float lam_52758);
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
static int futrts_main(struct futhark_context *ctx,
                       struct memblock *out_mem_p_54093,
                       int32_t *out_out_arrsizze_54094,
                       int32_t *out_out_arrsizze_54095,
                       struct memblock *out_mem_p_54096,
                       int32_t *out_out_arrsizze_54097,
                       int32_t *out_out_arrsizze_54098,
                       struct memblock *out_mem_p_54099,
                       int32_t *out_out_arrsizze_54100,
                       struct memblock *out_mem_p_54101,
                       int32_t *out_out_arrsizze_54102,
                       struct memblock *out_mem_p_54103,
                       int32_t *out_out_arrsizze_54104,
                       int32_t *out_out_arrsizze_54105,
                       struct memblock *out_mem_p_54106,
                       int32_t *out_out_arrsizze_54107,
                       int32_t *out_out_arrsizze_54108,
                       int32_t *out_out_arrsizze_54109,
                       struct memblock *out_mem_p_54110,
                       int32_t *out_out_arrsizze_54111,
                       int32_t *out_out_arrsizze_54112,
                       int32_t *out_out_arrsizze_54113,
                       struct memblock *out_mem_p_54114,
                       int32_t *out_out_arrsizze_54115,
                       int32_t *out_out_arrsizze_54116,
                       struct memblock *out_mem_p_54117,
                       int32_t *out_out_arrsizze_54118,
                       int32_t *out_out_arrsizze_54119,
                       struct memblock *out_mem_p_54120,
                       int32_t *out_out_arrsizze_54121,
                       int32_t *out_out_arrsizze_54122,
                       struct memblock *out_mem_p_54123,
                       int32_t *out_out_arrsizze_54124,
                       int32_t *out_out_arrsizze_54125,
                       struct memblock *out_mem_p_54126,
                       int32_t *out_out_arrsizze_54127,
                       struct memblock *out_mem_p_54128,
                       int32_t *out_out_arrsizze_54129,
                       int32_t *out_out_arrsizze_54130,
                       struct memblock *out_mem_p_54131,
                       int32_t *out_out_arrsizze_54132,
                       struct memblock *out_mem_p_54133,
                       int32_t *out_out_arrsizze_54134,
                       struct memblock *out_mem_p_54135,
                       int32_t *out_out_arrsizze_54136,
                       struct memblock *out_mem_p_54137,
                       int32_t *out_out_arrsizze_54138,
                       struct memblock mappingindices_mem_53689,
                       struct memblock images_mem_53690, int32_t N_52750,
                       int32_t m_52751, int32_t N_52752, int32_t trend_52753,
                       int32_t k_52754, int32_t n_52755, float freq_52756,
                       float hfrac_52757, float lam_52758)
{
    struct memblock out_mem_53988;
    
    out_mem_53988.references = NULL;
    
    int32_t out_arrsizze_53989;
    int32_t out_arrsizze_53990;
    struct memblock out_mem_53991;
    
    out_mem_53991.references = NULL;
    
    int32_t out_arrsizze_53992;
    int32_t out_arrsizze_53993;
    struct memblock out_mem_53994;
    
    out_mem_53994.references = NULL;
    
    int32_t out_arrsizze_53995;
    struct memblock out_mem_53996;
    
    out_mem_53996.references = NULL;
    
    int32_t out_arrsizze_53997;
    struct memblock out_mem_53998;
    
    out_mem_53998.references = NULL;
    
    int32_t out_arrsizze_53999;
    int32_t out_arrsizze_54000;
    struct memblock out_mem_54001;
    
    out_mem_54001.references = NULL;
    
    int32_t out_arrsizze_54002;
    int32_t out_arrsizze_54003;
    int32_t out_arrsizze_54004;
    struct memblock out_mem_54005;
    
    out_mem_54005.references = NULL;
    
    int32_t out_arrsizze_54006;
    int32_t out_arrsizze_54007;
    int32_t out_arrsizze_54008;
    struct memblock out_mem_54009;
    
    out_mem_54009.references = NULL;
    
    int32_t out_arrsizze_54010;
    int32_t out_arrsizze_54011;
    struct memblock out_mem_54012;
    
    out_mem_54012.references = NULL;
    
    int32_t out_arrsizze_54013;
    int32_t out_arrsizze_54014;
    struct memblock out_mem_54015;
    
    out_mem_54015.references = NULL;
    
    int32_t out_arrsizze_54016;
    int32_t out_arrsizze_54017;
    struct memblock out_mem_54018;
    
    out_mem_54018.references = NULL;
    
    int32_t out_arrsizze_54019;
    int32_t out_arrsizze_54020;
    struct memblock out_mem_54021;
    
    out_mem_54021.references = NULL;
    
    int32_t out_arrsizze_54022;
    struct memblock out_mem_54023;
    
    out_mem_54023.references = NULL;
    
    int32_t out_arrsizze_54024;
    int32_t out_arrsizze_54025;
    struct memblock out_mem_54026;
    
    out_mem_54026.references = NULL;
    
    int32_t out_arrsizze_54027;
    struct memblock out_mem_54028;
    
    out_mem_54028.references = NULL;
    
    int32_t out_arrsizze_54029;
    struct memblock out_mem_54030;
    
    out_mem_54030.references = NULL;
    
    int32_t out_arrsizze_54031;
    struct memblock out_mem_54032;
    
    out_mem_54032.references = NULL;
    
    int32_t out_arrsizze_54033;
    bool dim_zzero_52761 = 0 == m_52751;
    bool dim_zzero_52762 = 0 == N_52752;
    bool old_empty_52763 = dim_zzero_52761 || dim_zzero_52762;
    bool dim_zzero_52764 = 0 == N_52750;
    bool new_empty_52765 = dim_zzero_52761 || dim_zzero_52764;
    bool both_empty_52766 = old_empty_52763 && new_empty_52765;
    bool dim_match_52767 = N_52750 == N_52752;
    bool empty_or_match_52768 = both_empty_52766 || dim_match_52767;
    bool empty_or_match_cert_52769;
    
    if (!empty_or_match_52768) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "bfast-irreg.fut:133:1-314:123",
                               "function arguments of wrong shape");
        if (memblock_unref(ctx, &out_mem_54032, "out_mem_54032") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54030, "out_mem_54030") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54028, "out_mem_54028") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54026, "out_mem_54026") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54023, "out_mem_54023") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54021, "out_mem_54021") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54018, "out_mem_54018") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54015, "out_mem_54015") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54012, "out_mem_54012") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54009, "out_mem_54009") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54005, "out_mem_54005") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54001, "out_mem_54001") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53998, "out_mem_53998") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53996, "out_mem_53996") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53994, "out_mem_53994") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53991, "out_mem_53991") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53988, "out_mem_53988") != 0)
            return 1;
        return 1;
    }
    
    int32_t x_52771 = 2 * k_52754;
    int32_t k2p2_52772 = 2 + x_52771;
    bool cond_52773 = slt32(0, trend_52753);
    int32_t k2p2zq_52774;
    
    if (cond_52773) {
        k2p2zq_52774 = k2p2_52772;
    } else {
        int32_t res_52775 = k2p2_52772 - 1;
        
        k2p2zq_52774 = res_52775;
    }
    
    int64_t binop_x_53692 = sext_i32_i64(k2p2zq_52774);
    int64_t binop_y_53693 = sext_i32_i64(N_52750);
    int64_t binop_x_53694 = binop_x_53692 * binop_y_53693;
    int64_t bytes_53691 = 4 * binop_x_53694;
    int64_t binop_x_53707 = sext_i32_i64(k2p2zq_52774);
    int64_t binop_y_53708 = sext_i32_i64(N_52750);
    int64_t binop_x_53709 = binop_x_53707 * binop_y_53708;
    int64_t bytes_53706 = 4 * binop_x_53709;
    struct memblock lifted_1_zlzb_arg_mem_53721;
    
    lifted_1_zlzb_arg_mem_53721.references = NULL;
    if (cond_52773) {
        bool bounds_invalid_upwards_52777 = slt32(k2p2zq_52774, 0);
        bool eq_x_zz_52778 = 0 == k2p2zq_52774;
        bool not_p_52779 = !bounds_invalid_upwards_52777;
        bool p_and_eq_x_y_52780 = eq_x_zz_52778 && not_p_52779;
        bool dim_zzero_52781 = bounds_invalid_upwards_52777 ||
             p_and_eq_x_y_52780;
        bool both_empty_52782 = eq_x_zz_52778 && dim_zzero_52781;
        bool empty_or_match_52786 = not_p_52779 || both_empty_52782;
        bool empty_or_match_cert_52787;
        
        if (!empty_or_match_52786) {
            ctx->error = msgprintf("Error at %s:\n%s%s%s%d%s%s\n",
                                   "bfast-irreg.fut:133:1-314:123 -> bfast-irreg.fut:144:16-55 -> bfast-irreg.fut:64:10-18 -> /futlib/array.fut:61:1-62:12",
                                   "Function return value does not match shape of type ",
                                   "*", "[", k2p2zq_52774, "]",
                                   "intrinsics.i32");
            if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_53721,
                               "lifted_1_zlzb_arg_mem_53721") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_54032, "out_mem_54032") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_54030, "out_mem_54030") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_54028, "out_mem_54028") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_54026, "out_mem_54026") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_54023, "out_mem_54023") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_54021, "out_mem_54021") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_54018, "out_mem_54018") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_54015, "out_mem_54015") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_54012, "out_mem_54012") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_54009, "out_mem_54009") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_54005, "out_mem_54005") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_54001, "out_mem_54001") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53998, "out_mem_53998") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53996, "out_mem_53996") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53994, "out_mem_53994") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53991, "out_mem_53991") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53988, "out_mem_53988") != 0)
                return 1;
            return 1;
        }
        
        struct memblock mem_53695;
        
        mem_53695.references = NULL;
        if (memblock_alloc(ctx, &mem_53695, bytes_53691, "mem_53695"))
            return 1;
        for (int32_t i_53346 = 0; i_53346 < k2p2zq_52774; i_53346++) {
            bool cond_52791 = i_53346 == 0;
            bool cond_52792 = i_53346 == 1;
            int32_t r32_arg_52793 = sdiv32(i_53346, 2);
            int32_t x_52794 = smod32(i_53346, 2);
            float res_52795 = sitofp_i32_f32(r32_arg_52793);
            bool cond_52796 = x_52794 == 0;
            float x_52797 = 6.2831855F * res_52795;
            
            for (int32_t i_53342 = 0; i_53342 < N_52750; i_53342++) {
                int32_t x_52799 =
                        ((int32_t *) mappingindices_mem_53689.mem)[i_53342];
                float res_52800;
                
                if (cond_52791) {
                    res_52800 = 1.0F;
                } else {
                    float res_52801;
                    
                    if (cond_52792) {
                        float res_52802 = sitofp_i32_f32(x_52799);
                        
                        res_52801 = res_52802;
                    } else {
                        float res_52803 = sitofp_i32_f32(x_52799);
                        float x_52804 = x_52797 * res_52803;
                        float angle_52805 = x_52804 / freq_52756;
                        float res_52806;
                        
                        if (cond_52796) {
                            float res_52807;
                            
                            res_52807 = futrts_sin32(angle_52805);
                            res_52806 = res_52807;
                        } else {
                            float res_52808;
                            
                            res_52808 = futrts_cos32(angle_52805);
                            res_52806 = res_52808;
                        }
                        res_52801 = res_52806;
                    }
                    res_52800 = res_52801;
                }
                ((float *) mem_53695.mem)[i_53346 * N_52750 + i_53342] =
                    res_52800;
            }
        }
        if (memblock_set(ctx, &lifted_1_zlzb_arg_mem_53721, &mem_53695,
                         "mem_53695") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53695, "mem_53695") != 0)
            return 1;
    } else {
        bool bounds_invalid_upwards_52809 = slt32(k2p2zq_52774, 0);
        bool eq_x_zz_52810 = 0 == k2p2zq_52774;
        bool not_p_52811 = !bounds_invalid_upwards_52809;
        bool p_and_eq_x_y_52812 = eq_x_zz_52810 && not_p_52811;
        bool dim_zzero_52813 = bounds_invalid_upwards_52809 ||
             p_and_eq_x_y_52812;
        bool both_empty_52814 = eq_x_zz_52810 && dim_zzero_52813;
        bool empty_or_match_52818 = not_p_52811 || both_empty_52814;
        bool empty_or_match_cert_52819;
        
        if (!empty_or_match_52818) {
            ctx->error = msgprintf("Error at %s:\n%s%s%s%d%s%s\n",
                                   "bfast-irreg.fut:133:1-314:123 -> bfast-irreg.fut:145:16-55 -> bfast-irreg.fut:76:10-20 -> /futlib/array.fut:61:1-62:12",
                                   "Function return value does not match shape of type ",
                                   "*", "[", k2p2zq_52774, "]",
                                   "intrinsics.i32");
            if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_53721,
                               "lifted_1_zlzb_arg_mem_53721") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_54032, "out_mem_54032") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_54030, "out_mem_54030") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_54028, "out_mem_54028") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_54026, "out_mem_54026") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_54023, "out_mem_54023") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_54021, "out_mem_54021") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_54018, "out_mem_54018") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_54015, "out_mem_54015") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_54012, "out_mem_54012") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_54009, "out_mem_54009") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_54005, "out_mem_54005") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_54001, "out_mem_54001") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53998, "out_mem_53998") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53996, "out_mem_53996") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53994, "out_mem_53994") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53991, "out_mem_53991") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53988, "out_mem_53988") != 0)
                return 1;
            return 1;
        }
        
        struct memblock mem_53710;
        
        mem_53710.references = NULL;
        if (memblock_alloc(ctx, &mem_53710, bytes_53706, "mem_53710"))
            return 1;
        for (int32_t i_53354 = 0; i_53354 < k2p2zq_52774; i_53354++) {
            bool cond_52823 = i_53354 == 0;
            int32_t i_52824 = 1 + i_53354;
            int32_t r32_arg_52825 = sdiv32(i_52824, 2);
            int32_t x_52826 = smod32(i_52824, 2);
            float res_52827 = sitofp_i32_f32(r32_arg_52825);
            bool cond_52828 = x_52826 == 0;
            float x_52829 = 6.2831855F * res_52827;
            
            for (int32_t i_53350 = 0; i_53350 < N_52750; i_53350++) {
                int32_t x_52831 =
                        ((int32_t *) mappingindices_mem_53689.mem)[i_53350];
                float res_52832;
                
                if (cond_52823) {
                    res_52832 = 1.0F;
                } else {
                    float res_52833 = sitofp_i32_f32(x_52831);
                    float x_52834 = x_52829 * res_52833;
                    float angle_52835 = x_52834 / freq_52756;
                    float res_52836;
                    
                    if (cond_52828) {
                        float res_52837;
                        
                        res_52837 = futrts_sin32(angle_52835);
                        res_52836 = res_52837;
                    } else {
                        float res_52838;
                        
                        res_52838 = futrts_cos32(angle_52835);
                        res_52836 = res_52838;
                    }
                    res_52832 = res_52836;
                }
                ((float *) mem_53710.mem)[i_53354 * N_52750 + i_53350] =
                    res_52832;
            }
        }
        if (memblock_set(ctx, &lifted_1_zlzb_arg_mem_53721, &mem_53710,
                         "mem_53710") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53710, "mem_53710") != 0)
            return 1;
    }
    
    int32_t x_52840 = N_52750 * N_52750;
    int32_t y_52841 = 2 * N_52750;
    int32_t x_52842 = x_52840 + y_52841;
    int32_t x_52843 = 1 + x_52842;
    int32_t y_52844 = 1 + N_52750;
    int32_t x_52845 = sdiv32(x_52843, y_52844);
    int32_t x_52846 = x_52845 - N_52750;
    int32_t lifted_1_zlzb_arg_52847 = x_52846 - 1;
    float res_52848 = sitofp_i32_f32(lifted_1_zlzb_arg_52847);
    int64_t binop_x_53723 = sext_i32_i64(N_52750);
    int64_t binop_y_53724 = sext_i32_i64(k2p2zq_52774);
    int64_t binop_x_53725 = binop_x_53723 * binop_y_53724;
    int64_t bytes_53722 = 4 * binop_x_53725;
    struct memblock mem_53726;
    
    mem_53726.references = NULL;
    if (memblock_alloc(ctx, &mem_53726, bytes_53722, "mem_53726"))
        return 1;
    for (int32_t i_53362 = 0; i_53362 < N_52750; i_53362++) {
        for (int32_t i_53358 = 0; i_53358 < k2p2zq_52774; i_53358++) {
            float x_52853 =
                  ((float *) lifted_1_zlzb_arg_mem_53721.mem)[i_53358 *
                                                              N_52750 +
                                                              i_53362];
            float res_52854 = res_52848 + x_52853;
            
            ((float *) mem_53726.mem)[i_53362 * k2p2zq_52774 + i_53358] =
                res_52854;
        }
    }
    
    int32_t m_52857 = k2p2zq_52774 - 1;
    bool empty_slice_52864 = n_52755 == 0;
    int32_t m_52865 = n_52755 - 1;
    bool zzero_leq_i_p_m_t_s_52866 = sle32(0, m_52865);
    bool i_p_m_t_s_leq_w_52867 = slt32(m_52865, N_52750);
    bool i_lte_j_52868 = sle32(0, n_52755);
    bool y_52869 = zzero_leq_i_p_m_t_s_52866 && i_p_m_t_s_leq_w_52867;
    bool y_52870 = i_lte_j_52868 && y_52869;
    bool ok_or_empty_52871 = empty_slice_52864 || y_52870;
    bool index_certs_52873;
    
    if (!ok_or_empty_52871) {
        ctx->error = msgprintf("Error at %s:\n%s%d%s%s%s%d%s%d%s%d%s\n",
                               "bfast-irreg.fut:133:1-314:123 -> bfast-irreg.fut:154:15-21",
                               "Index [", 0, ", ", "", ":", n_52755,
                               "] out of bounds for array of shape [",
                               k2p2zq_52774, "][", N_52750, "].");
        if (memblock_unref(ctx, &mem_53726, "mem_53726") != 0)
            return 1;
        if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_53721,
                           "lifted_1_zlzb_arg_mem_53721") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54032, "out_mem_54032") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54030, "out_mem_54030") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54028, "out_mem_54028") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54026, "out_mem_54026") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54023, "out_mem_54023") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54021, "out_mem_54021") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54018, "out_mem_54018") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54015, "out_mem_54015") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54012, "out_mem_54012") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54009, "out_mem_54009") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54005, "out_mem_54005") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54001, "out_mem_54001") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53998, "out_mem_53998") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53996, "out_mem_53996") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53994, "out_mem_53994") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53991, "out_mem_53991") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53988, "out_mem_53988") != 0)
            return 1;
        return 1;
    }
    
    bool index_certs_52875;
    
    if (!ok_or_empty_52871) {
        ctx->error = msgprintf("Error at %s:\n%s%s%s%d%s%d%s%d%s%d%s\n",
                               "bfast-irreg.fut:133:1-314:123 -> bfast-irreg.fut:155:15-22",
                               "Index [", "", ":", n_52755, ", ", 0,
                               "] out of bounds for array of shape [", N_52750,
                               "][", k2p2zq_52774, "].");
        if (memblock_unref(ctx, &mem_53726, "mem_53726") != 0)
            return 1;
        if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_53721,
                           "lifted_1_zlzb_arg_mem_53721") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54032, "out_mem_54032") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54030, "out_mem_54030") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54028, "out_mem_54028") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54026, "out_mem_54026") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54023, "out_mem_54023") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54021, "out_mem_54021") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54018, "out_mem_54018") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54015, "out_mem_54015") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54012, "out_mem_54012") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54009, "out_mem_54009") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54005, "out_mem_54005") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54001, "out_mem_54001") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53998, "out_mem_53998") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53996, "out_mem_53996") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53994, "out_mem_53994") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53991, "out_mem_53991") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53988, "out_mem_53988") != 0)
            return 1;
        return 1;
    }
    
    bool index_certs_52886;
    
    if (!ok_or_empty_52871) {
        ctx->error = msgprintf("Error at %s:\n%s%d%s%s%s%d%s%d%s%d%s\n",
                               "bfast-irreg.fut:133:1-314:123 -> bfast-irreg.fut:156:15-26",
                               "Index [", 0, ", ", "", ":", n_52755,
                               "] out of bounds for array of shape [", m_52751,
                               "][", N_52750, "].");
        if (memblock_unref(ctx, &mem_53726, "mem_53726") != 0)
            return 1;
        if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_53721,
                           "lifted_1_zlzb_arg_mem_53721") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54032, "out_mem_54032") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54030, "out_mem_54030") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54028, "out_mem_54028") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54026, "out_mem_54026") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54023, "out_mem_54023") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54021, "out_mem_54021") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54018, "out_mem_54018") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54015, "out_mem_54015") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54012, "out_mem_54012") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54009, "out_mem_54009") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54005, "out_mem_54005") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54001, "out_mem_54001") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53998, "out_mem_53998") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53996, "out_mem_53996") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53994, "out_mem_53994") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53991, "out_mem_53991") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53988, "out_mem_53988") != 0)
            return 1;
        return 1;
    }
    
    int64_t binop_x_53738 = sext_i32_i64(m_52751);
    int64_t binop_x_53740 = binop_y_53724 * binop_x_53738;
    int64_t binop_x_53742 = binop_y_53724 * binop_x_53740;
    int64_t bytes_53737 = 4 * binop_x_53742;
    struct memblock mem_53743;
    
    mem_53743.references = NULL;
    if (memblock_alloc(ctx, &mem_53743, bytes_53737, "mem_53743"))
        return 1;
    for (int32_t i_53376 = 0; i_53376 < m_52751; i_53376++) {
        for (int32_t i_53372 = 0; i_53372 < k2p2zq_52774; i_53372++) {
            for (int32_t i_53368 = 0; i_53368 < k2p2zq_52774; i_53368++) {
                float res_52895;
                float redout_53364 = 0.0F;
                
                for (int32_t i_53365 = 0; i_53365 < n_52755; i_53365++) {
                    float x_52899 = ((float *) images_mem_53690.mem)[i_53376 *
                                                                     N_52752 +
                                                                     i_53365];
                    float x_52900 =
                          ((float *) lifted_1_zlzb_arg_mem_53721.mem)[i_53372 *
                                                                      N_52750 +
                                                                      i_53365];
                    float x_52901 = ((float *) mem_53726.mem)[i_53365 *
                                                              k2p2zq_52774 +
                                                              i_53368];
                    float x_52902 = x_52900 * x_52901;
                    bool res_52903;
                    
                    res_52903 = futrts_isnan32(x_52899);
                    
                    float y_52904;
                    
                    if (res_52903) {
                        y_52904 = 0.0F;
                    } else {
                        y_52904 = 1.0F;
                    }
                    
                    float res_52905 = x_52902 * y_52904;
                    float res_52898 = res_52905 + redout_53364;
                    float redout_tmp_54043 = res_52898;
                    
                    redout_53364 = redout_tmp_54043;
                }
                res_52895 = redout_53364;
                ((float *) mem_53743.mem)[i_53376 * (k2p2zq_52774 *
                                                     k2p2zq_52774) + i_53372 *
                                          k2p2zq_52774 + i_53368] = res_52895;
            }
        }
    }
    
    int32_t j_52907 = 2 * k2p2zq_52774;
    int32_t j_m_i_52908 = j_52907 - k2p2zq_52774;
    int32_t nm_52911 = k2p2zq_52774 * j_52907;
    bool empty_slice_52924 = j_m_i_52908 == 0;
    int32_t m_52925 = j_m_i_52908 - 1;
    int32_t i_p_m_t_s_52926 = k2p2zq_52774 + m_52925;
    bool zzero_leq_i_p_m_t_s_52927 = sle32(0, i_p_m_t_s_52926);
    bool ok_or_empty_52934 = empty_slice_52924 || zzero_leq_i_p_m_t_s_52927;
    bool index_certs_52936;
    
    if (!ok_or_empty_52934) {
        ctx->error = msgprintf("Error at %s:\n%s%d%s%d%s%d%s%d%s%d%s%d%s\n",
                               "bfast-irreg.fut:133:1-314:123 -> bfast-irreg.fut:168:14-29 -> bfast-irreg.fut:109:8-37",
                               "Index [", 0, ":", k2p2zq_52774, ", ",
                               k2p2zq_52774, ":", j_52907,
                               "] out of bounds for array of shape [",
                               k2p2zq_52774, "][", j_52907, "].");
        if (memblock_unref(ctx, &mem_53743, "mem_53743") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53726, "mem_53726") != 0)
            return 1;
        if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_53721,
                           "lifted_1_zlzb_arg_mem_53721") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54032, "out_mem_54032") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54030, "out_mem_54030") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54028, "out_mem_54028") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54026, "out_mem_54026") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54023, "out_mem_54023") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54021, "out_mem_54021") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54018, "out_mem_54018") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54015, "out_mem_54015") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54012, "out_mem_54012") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54009, "out_mem_54009") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54005, "out_mem_54005") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54001, "out_mem_54001") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53998, "out_mem_53998") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53996, "out_mem_53996") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53994, "out_mem_53994") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53991, "out_mem_53991") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53988, "out_mem_53988") != 0)
            return 1;
        return 1;
    }
    
    int64_t binop_y_53775 = sext_i32_i64(j_m_i_52908);
    int64_t binop_x_53776 = binop_x_53740 * binop_y_53775;
    int64_t bytes_53771 = 4 * binop_x_53776;
    struct memblock mem_53777;
    
    mem_53777.references = NULL;
    if (memblock_alloc(ctx, &mem_53777, bytes_53771, "mem_53777"))
        return 1;
    
    int64_t binop_x_53780 = sext_i32_i64(nm_52911);
    int64_t bytes_53779 = 4 * binop_x_53780;
    struct memblock mem_53781;
    
    mem_53781.references = NULL;
    if (memblock_alloc(ctx, &mem_53781, bytes_53779, "mem_53781"))
        return 1;
    
    struct memblock mem_53787;
    
    mem_53787.references = NULL;
    if (memblock_alloc(ctx, &mem_53787, bytes_53779, "mem_53787"))
        return 1;
    for (int32_t i_53398 = 0; i_53398 < m_52751; i_53398++) {
        for (int32_t i_53380 = 0; i_53380 < nm_52911; i_53380++) {
            int32_t res_52941 = sdiv32(i_53380, j_52907);
            int32_t res_52942 = smod32(i_53380, j_52907);
            bool cond_52943 = slt32(res_52942, k2p2zq_52774);
            float res_52944;
            
            if (cond_52943) {
                float res_52945 = ((float *) mem_53743.mem)[i_53398 *
                                                            (k2p2zq_52774 *
                                                             k2p2zq_52774) +
                                                            res_52941 *
                                                            k2p2zq_52774 +
                                                            res_52942];
                
                res_52944 = res_52945;
            } else {
                int32_t y_52946 = k2p2zq_52774 + res_52941;
                bool cond_52947 = res_52942 == y_52946;
                float res_52948;
                
                if (cond_52947) {
                    res_52948 = 1.0F;
                } else {
                    res_52948 = 0.0F;
                }
                res_52944 = res_52948;
            }
            ((float *) mem_53781.mem)[i_53380] = res_52944;
        }
        for (int32_t i_52951 = 0; i_52951 < k2p2zq_52774; i_52951++) {
            float v1_52956 = ((float *) mem_53781.mem)[i_52951];
            bool cond_52957 = v1_52956 == 0.0F;
            
            for (int32_t i_53384 = 0; i_53384 < nm_52911; i_53384++) {
                int32_t res_52960 = sdiv32(i_53384, j_52907);
                int32_t res_52961 = smod32(i_53384, j_52907);
                float res_52962;
                
                if (cond_52957) {
                    int32_t x_52963 = j_52907 * res_52960;
                    int32_t i_52964 = res_52961 + x_52963;
                    float res_52965 = ((float *) mem_53781.mem)[i_52964];
                    
                    res_52962 = res_52965;
                } else {
                    float x_52966 = ((float *) mem_53781.mem)[res_52961];
                    float x_52967 = x_52966 / v1_52956;
                    bool cond_52968 = slt32(res_52960, m_52857);
                    float res_52969;
                    
                    if (cond_52968) {
                        int32_t x_52970 = 1 + res_52960;
                        int32_t x_52971 = j_52907 * x_52970;
                        int32_t i_52972 = res_52961 + x_52971;
                        float x_52973 = ((float *) mem_53781.mem)[i_52972];
                        int32_t i_52974 = i_52951 + x_52971;
                        float x_52975 = ((float *) mem_53781.mem)[i_52974];
                        float y_52976 = x_52967 * x_52975;
                        float res_52977 = x_52973 - y_52976;
                        
                        res_52969 = res_52977;
                    } else {
                        res_52969 = x_52967;
                    }
                    res_52962 = res_52969;
                }
                ((float *) mem_53787.mem)[i_53384] = res_52962;
            }
            for (int32_t write_iter_53386 = 0; write_iter_53386 < nm_52911;
                 write_iter_53386++) {
                bool less_than_zzero_53390 = slt32(write_iter_53386, 0);
                bool greater_than_sizze_53391 = sle32(nm_52911,
                                                      write_iter_53386);
                bool outside_bounds_dim_53392 = less_than_zzero_53390 ||
                     greater_than_sizze_53391;
                
                if (!outside_bounds_dim_53392) {
                    memmove(mem_53781.mem + write_iter_53386 * 4,
                            mem_53787.mem + write_iter_53386 * 4,
                            sizeof(float));
                }
            }
        }
        for (int32_t i_54049 = 0; i_54049 < k2p2zq_52774; i_54049++) {
            for (int32_t i_54050 = 0; i_54050 < j_m_i_52908; i_54050++) {
                ((float *) mem_53777.mem)[i_53398 * (j_m_i_52908 *
                                                     k2p2zq_52774) + (i_54049 *
                                                                      j_m_i_52908 +
                                                                      i_54050)] =
                    ((float *) mem_53781.mem)[k2p2zq_52774 + (i_54049 *
                                                              j_52907 +
                                                              i_54050)];
            }
        }
    }
    if (memblock_unref(ctx, &mem_53781, "mem_53781") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53787, "mem_53787") != 0)
        return 1;
    
    int64_t bytes_53794 = 4 * binop_x_53740;
    struct memblock mem_53798;
    
    mem_53798.references = NULL;
    if (memblock_alloc(ctx, &mem_53798, bytes_53794, "mem_53798"))
        return 1;
    for (int32_t i_53408 = 0; i_53408 < m_52751; i_53408++) {
        for (int32_t i_53404 = 0; i_53404 < k2p2zq_52774; i_53404++) {
            float res_52988;
            float redout_53400 = 0.0F;
            
            for (int32_t i_53401 = 0; i_53401 < n_52755; i_53401++) {
                float x_52992 =
                      ((float *) lifted_1_zlzb_arg_mem_53721.mem)[i_53404 *
                                                                  N_52750 +
                                                                  i_53401];
                float x_52993 = ((float *) images_mem_53690.mem)[i_53408 *
                                                                 N_52752 +
                                                                 i_53401];
                bool res_52994;
                
                res_52994 = futrts_isnan32(x_52993);
                
                float res_52995;
                
                if (res_52994) {
                    res_52995 = 0.0F;
                } else {
                    float res_52996 = x_52992 * x_52993;
                    
                    res_52995 = res_52996;
                }
                
                float res_52991 = res_52995 + redout_53400;
                float redout_tmp_54053 = res_52991;
                
                redout_53400 = redout_tmp_54053;
            }
            res_52988 = redout_53400;
            ((float *) mem_53798.mem)[i_53408 * k2p2zq_52774 + i_53404] =
                res_52988;
        }
    }
    if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_53721,
                       "lifted_1_zlzb_arg_mem_53721") != 0)
        return 1;
    
    struct memblock mem_53813;
    
    mem_53813.references = NULL;
    if (memblock_alloc(ctx, &mem_53813, bytes_53794, "mem_53813"))
        return 1;
    for (int32_t i_53418 = 0; i_53418 < m_52751; i_53418++) {
        for (int32_t i_53414 = 0; i_53414 < k2p2zq_52774; i_53414++) {
            float res_53008;
            float redout_53410 = 0.0F;
            
            for (int32_t i_53411 = 0; i_53411 < j_m_i_52908; i_53411++) {
                float x_53012 = ((float *) mem_53798.mem)[i_53418 *
                                                          k2p2zq_52774 +
                                                          i_53411];
                float x_53013 = ((float *) mem_53777.mem)[i_53418 *
                                                          (j_m_i_52908 *
                                                           k2p2zq_52774) +
                                                          i_53414 *
                                                          j_m_i_52908 +
                                                          i_53411];
                float res_53014 = x_53012 * x_53013;
                float res_53011 = res_53014 + redout_53410;
                float redout_tmp_54056 = res_53011;
                
                redout_53410 = redout_tmp_54056;
            }
            res_53008 = redout_53410;
            ((float *) mem_53813.mem)[i_53418 * k2p2zq_52774 + i_53414] =
                res_53008;
        }
    }
    
    int64_t binop_x_53827 = binop_x_53723 * binop_x_53738;
    int64_t bytes_53824 = 4 * binop_x_53827;
    struct memblock mem_53828;
    
    mem_53828.references = NULL;
    if (memblock_alloc(ctx, &mem_53828, bytes_53824, "mem_53828"))
        return 1;
    for (int32_t i_53428 = 0; i_53428 < m_52751; i_53428++) {
        for (int32_t i_53424 = 0; i_53424 < N_52750; i_53424++) {
            float res_53020;
            float redout_53420 = 0.0F;
            
            for (int32_t i_53421 = 0; i_53421 < k2p2zq_52774; i_53421++) {
                float x_53024 = ((float *) mem_53813.mem)[i_53428 *
                                                          k2p2zq_52774 +
                                                          i_53421];
                float x_53025 = ((float *) mem_53726.mem)[i_53424 *
                                                          k2p2zq_52774 +
                                                          i_53421];
                float res_53026 = x_53024 * x_53025;
                float res_53023 = res_53026 + redout_53420;
                float redout_tmp_54059 = res_53023;
                
                redout_53420 = redout_tmp_54059;
            }
            res_53020 = redout_53420;
            ((float *) mem_53828.mem)[i_53428 * N_52750 + i_53424] = res_53020;
        }
    }
    
    int32_t i_53028 = N_52750 - 1;
    bool x_53029 = sle32(0, i_53028);
    bool index_certs_53032;
    
    if (!x_53029) {
        ctx->error = msgprintf("Error at %s:\n%s%d%s%d%s\n",
                               "bfast-irreg.fut:133:1-314:123 -> bfast-irreg.fut:189:5-198:25 -> /futlib/soacs.fut:51:3-37 -> /futlib/soacs.fut:51:19-23 -> bfast-irreg.fut:194:30-91 -> bfast-irreg.fut:37:13-20 -> /futlib/array.fut:18:29-34",
                               "Index [", i_53028,
                               "] out of bounds for array of shape [", N_52750,
                               "].");
        if (memblock_unref(ctx, &mem_53828, "mem_53828") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53813, "mem_53813") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53798, "mem_53798") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53787, "mem_53787") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53781, "mem_53781") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53777, "mem_53777") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53743, "mem_53743") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53726, "mem_53726") != 0)
            return 1;
        if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_53721,
                           "lifted_1_zlzb_arg_mem_53721") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54032, "out_mem_54032") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54030, "out_mem_54030") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54028, "out_mem_54028") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54026, "out_mem_54026") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54023, "out_mem_54023") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54021, "out_mem_54021") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54018, "out_mem_54018") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54015, "out_mem_54015") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54012, "out_mem_54012") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54009, "out_mem_54009") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54005, "out_mem_54005") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54001, "out_mem_54001") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53998, "out_mem_53998") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53996, "out_mem_53996") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53994, "out_mem_53994") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53991, "out_mem_53991") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53988, "out_mem_53988") != 0)
            return 1;
        return 1;
    }
    
    int64_t bytes_53839 = 4 * binop_x_53738;
    struct memblock mem_53841;
    
    mem_53841.references = NULL;
    if (memblock_alloc(ctx, &mem_53841, bytes_53839, "mem_53841"))
        return 1;
    
    struct memblock mem_53846;
    
    mem_53846.references = NULL;
    if (memblock_alloc(ctx, &mem_53846, bytes_53824, "mem_53846"))
        return 1;
    
    struct memblock mem_53851;
    
    mem_53851.references = NULL;
    if (memblock_alloc(ctx, &mem_53851, bytes_53824, "mem_53851"))
        return 1;
    
    int64_t bytes_53855 = 4 * binop_x_53723;
    struct memblock mem_53857;
    
    mem_53857.references = NULL;
    if (memblock_alloc(ctx, &mem_53857, bytes_53855, "mem_53857"))
        return 1;
    
    struct memblock mem_53860;
    
    mem_53860.references = NULL;
    if (memblock_alloc(ctx, &mem_53860, bytes_53855, "mem_53860"))
        return 1;
    
    struct memblock mem_53867;
    
    mem_53867.references = NULL;
    if (memblock_alloc(ctx, &mem_53867, bytes_53855, "mem_53867"))
        return 1;
    for (int32_t i_54060 = 0; i_54060 < N_52750; i_54060++) {
        ((float *) mem_53867.mem)[i_54060] = NAN;
    }
    
    struct memblock mem_53870;
    
    mem_53870.references = NULL;
    if (memblock_alloc(ctx, &mem_53870, bytes_53855, "mem_53870"))
        return 1;
    for (int32_t i_54061 = 0; i_54061 < N_52750; i_54061++) {
        ((int32_t *) mem_53870.mem)[i_54061] = 0;
    }
    
    struct memblock mem_53875;
    
    mem_53875.references = NULL;
    if (memblock_alloc(ctx, &mem_53875, bytes_53855, "mem_53875"))
        return 1;
    
    struct memblock mem_53885;
    
    mem_53885.references = NULL;
    if (memblock_alloc(ctx, &mem_53885, bytes_53855, "mem_53885"))
        return 1;
    for (int32_t i_53463 = 0; i_53463 < m_52751; i_53463++) {
        int32_t discard_53438;
        int32_t scanacc_53432 = 0;
        
        for (int32_t i_53435 = 0; i_53435 < N_52750; i_53435++) {
            float x_53062 = ((float *) images_mem_53690.mem)[i_53463 * N_52752 +
                                                             i_53435];
            float x_53063 = ((float *) mem_53828.mem)[i_53463 * N_52750 +
                                                      i_53435];
            bool res_53064;
            
            res_53064 = futrts_isnan32(x_53062);
            
            bool cond_53065 = !res_53064;
            float res_53066;
            
            if (cond_53065) {
                float res_53067 = x_53062 - x_53063;
                
                res_53066 = res_53067;
            } else {
                res_53066 = NAN;
            }
            
            bool res_53068;
            
            res_53068 = futrts_isnan32(res_53066);
            
            bool res_53069 = !res_53068;
            int32_t res_53070;
            
            if (res_53069) {
                res_53070 = 1;
            } else {
                res_53070 = 0;
            }
            
            int32_t res_53061 = res_53070 + scanacc_53432;
            
            ((int32_t *) mem_53857.mem)[i_53435] = res_53061;
            ((float *) mem_53860.mem)[i_53435] = res_53066;
            
            int32_t scanacc_tmp_54065 = res_53061;
            
            scanacc_53432 = scanacc_tmp_54065;
        }
        discard_53438 = scanacc_53432;
        memmove(mem_53846.mem + i_53463 * N_52750 * 4, mem_53867.mem + 0,
                N_52750 * sizeof(float));
        memmove(mem_53851.mem + i_53463 * N_52750 * 4, mem_53870.mem + 0,
                N_52750 * sizeof(int32_t));
        for (int32_t write_iter_53439 = 0; write_iter_53439 < N_52750;
             write_iter_53439++) {
            float write_iv_53442 = ((float *) mem_53860.mem)[write_iter_53439];
            int32_t write_iv_53443 =
                    ((int32_t *) mem_53857.mem)[write_iter_53439];
            bool res_53081;
            
            res_53081 = futrts_isnan32(write_iv_53442);
            
            bool res_53082 = !res_53081;
            int32_t res_53083;
            
            if (res_53082) {
                int32_t res_53084 = write_iv_53443 - 1;
                
                res_53083 = res_53084;
            } else {
                res_53083 = -1;
            }
            
            bool less_than_zzero_53445 = slt32(res_53083, 0);
            bool greater_than_sizze_53446 = sle32(N_52750, res_53083);
            bool outside_bounds_dim_53447 = less_than_zzero_53445 ||
                 greater_than_sizze_53446;
            
            memmove(mem_53875.mem + 0, mem_53851.mem + i_53463 * N_52750 * 4,
                    N_52750 * sizeof(int32_t));
            
            struct memblock write_out_mem_53882;
            
            write_out_mem_53882.references = NULL;
            if (outside_bounds_dim_53447) {
                if (memblock_set(ctx, &write_out_mem_53882, &mem_53875,
                                 "mem_53875") != 0)
                    return 1;
            } else {
                struct memblock mem_53878;
                
                mem_53878.references = NULL;
                if (memblock_alloc(ctx, &mem_53878, 4, "mem_53878"))
                    return 1;
                
                int32_t x_54071;
                
                for (int32_t i_54070 = 0; i_54070 < 1; i_54070++) {
                    x_54071 = write_iter_53439 + sext_i32_i32(i_54070);
                    ((int32_t *) mem_53878.mem)[i_54070] = x_54071;
                }
                
                struct memblock mem_53881;
                
                mem_53881.references = NULL;
                if (memblock_alloc(ctx, &mem_53881, bytes_53855, "mem_53881"))
                    return 1;
                memmove(mem_53881.mem + 0, mem_53851.mem + i_53463 * N_52750 *
                        4, N_52750 * sizeof(int32_t));
                memmove(mem_53881.mem + res_53083 * 4, mem_53878.mem + 0,
                        sizeof(int32_t));
                if (memblock_unref(ctx, &mem_53878, "mem_53878") != 0)
                    return 1;
                if (memblock_set(ctx, &write_out_mem_53882, &mem_53881,
                                 "mem_53881") != 0)
                    return 1;
                if (memblock_unref(ctx, &mem_53881, "mem_53881") != 0)
                    return 1;
                if (memblock_unref(ctx, &mem_53878, "mem_53878") != 0)
                    return 1;
            }
            memmove(mem_53851.mem + i_53463 * N_52750 * 4,
                    write_out_mem_53882.mem + 0, N_52750 * sizeof(int32_t));
            if (memblock_unref(ctx, &write_out_mem_53882,
                               "write_out_mem_53882") != 0)
                return 1;
            memmove(mem_53885.mem + 0, mem_53846.mem + i_53463 * N_52750 * 4,
                    N_52750 * sizeof(float));
            
            struct memblock write_out_mem_53889;
            
            write_out_mem_53889.references = NULL;
            if (outside_bounds_dim_53447) {
                if (memblock_set(ctx, &write_out_mem_53889, &mem_53885,
                                 "mem_53885") != 0)
                    return 1;
            } else {
                struct memblock mem_53888;
                
                mem_53888.references = NULL;
                if (memblock_alloc(ctx, &mem_53888, bytes_53855, "mem_53888"))
                    return 1;
                memmove(mem_53888.mem + 0, mem_53846.mem + i_53463 * N_52750 *
                        4, N_52750 * sizeof(float));
                memmove(mem_53888.mem + res_53083 * 4, mem_53860.mem +
                        write_iter_53439 * 4, sizeof(float));
                if (memblock_set(ctx, &write_out_mem_53889, &mem_53888,
                                 "mem_53888") != 0)
                    return 1;
                if (memblock_unref(ctx, &mem_53888, "mem_53888") != 0)
                    return 1;
            }
            memmove(mem_53846.mem + i_53463 * N_52750 * 4,
                    write_out_mem_53889.mem + 0, N_52750 * sizeof(float));
            if (memblock_unref(ctx, &write_out_mem_53889,
                               "write_out_mem_53889") != 0)
                return 1;
            if (memblock_unref(ctx, &write_out_mem_53889,
                               "write_out_mem_53889") != 0)
                return 1;
            if (memblock_unref(ctx, &write_out_mem_53882,
                               "write_out_mem_53882") != 0)
                return 1;
        }
        memmove(mem_53841.mem + i_53463 * 4, mem_53857.mem + i_53028 * 4,
                sizeof(int32_t));
    }
    if (memblock_unref(ctx, &mem_53857, "mem_53857") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53860, "mem_53860") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53867, "mem_53867") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53870, "mem_53870") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53875, "mem_53875") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53885, "mem_53885") != 0)
        return 1;
    
    struct memblock mem_53897;
    
    mem_53897.references = NULL;
    if (memblock_alloc(ctx, &mem_53897, bytes_53839, "mem_53897"))
        return 1;
    
    struct memblock mem_53900;
    
    mem_53900.references = NULL;
    if (memblock_alloc(ctx, &mem_53900, bytes_53839, "mem_53900"))
        return 1;
    
    struct memblock mem_53903;
    
    mem_53903.references = NULL;
    if (memblock_alloc(ctx, &mem_53903, bytes_53839, "mem_53903"))
        return 1;
    for (int32_t i_53477 = 0; i_53477 < m_52751; i_53477++) {
        int32_t res_53105;
        int32_t redout_53467 = 0;
        
        for (int32_t i_53468 = 0; i_53468 < n_52755; i_53468++) {
            float x_53109 = ((float *) images_mem_53690.mem)[i_53477 * N_52752 +
                                                             i_53468];
            bool res_53110;
            
            res_53110 = futrts_isnan32(x_53109);
            
            bool cond_53111 = !res_53110;
            int32_t res_53112;
            
            if (cond_53111) {
                res_53112 = 1;
            } else {
                res_53112 = 0;
            }
            
            int32_t res_53108 = res_53112 + redout_53467;
            int32_t redout_tmp_54075 = res_53108;
            
            redout_53467 = redout_tmp_54075;
        }
        res_53105 = redout_53467;
        
        float res_53113;
        float redout_53469 = 0.0F;
        
        for (int32_t i_53470 = 0; i_53470 < n_52755; i_53470++) {
            float y_error_elem_53118 = ((float *) mem_53846.mem)[i_53477 *
                                                                 N_52750 +
                                                                 i_53470];
            bool cond_53119 = slt32(i_53470, res_53105);
            float res_53120;
            
            if (cond_53119) {
                res_53120 = y_error_elem_53118;
            } else {
                res_53120 = 0.0F;
            }
            
            float res_53121 = res_53120 * res_53120;
            float res_53116 = res_53121 + redout_53469;
            float redout_tmp_54076 = res_53116;
            
            redout_53469 = redout_tmp_54076;
        }
        res_53113 = redout_53469;
        
        int32_t r32_arg_53122 = res_53105 - k2p2_52772;
        float res_53123 = sitofp_i32_f32(r32_arg_53122);
        float sqrt_arg_53124 = res_53113 / res_53123;
        float res_53125;
        
        res_53125 = futrts_sqrt32(sqrt_arg_53124);
        
        float res_53126 = sitofp_i32_f32(res_53105);
        float t32_arg_53127 = hfrac_52757 * res_53126;
        int32_t res_53128 = fptosi_f32_i32(t32_arg_53127);
        
        ((int32_t *) mem_53897.mem)[i_53477] = res_53128;
        ((int32_t *) mem_53900.mem)[i_53477] = res_53105;
        ((float *) mem_53903.mem)[i_53477] = res_53125;
    }
    
    int32_t res_53132;
    int32_t redout_53481 = 0;
    
    for (int32_t i_53482 = 0; i_53482 < m_52751; i_53482++) {
        int32_t x_53136 = ((int32_t *) mem_53897.mem)[i_53482];
        int32_t res_53135 = smax32(x_53136, redout_53481);
        int32_t redout_tmp_54077 = res_53135;
        
        redout_53481 = redout_tmp_54077;
    }
    res_53132 = redout_53481;
    
    struct memblock mem_53912;
    
    mem_53912.references = NULL;
    if (memblock_alloc(ctx, &mem_53912, bytes_53839, "mem_53912"))
        return 1;
    for (int32_t i_53487 = 0; i_53487 < m_52751; i_53487++) {
        int32_t x_53140 = ((int32_t *) mem_53900.mem)[i_53487];
        int32_t x_53141 = ((int32_t *) mem_53897.mem)[i_53487];
        float res_53142;
        float redout_53483 = 0.0F;
        
        for (int32_t i_53484 = 0; i_53484 < res_53132; i_53484++) {
            bool cond_53147 = slt32(i_53484, x_53141);
            float res_53148;
            
            if (cond_53147) {
                int32_t x_53149 = x_53140 + i_53484;
                int32_t x_53150 = x_53149 - x_53141;
                int32_t i_53151 = 1 + x_53150;
                float res_53152 = ((float *) mem_53846.mem)[i_53487 * N_52750 +
                                                            i_53151];
                
                res_53148 = res_53152;
            } else {
                res_53148 = 0.0F;
            }
            
            float res_53145 = res_53148 + redout_53483;
            float redout_tmp_54079 = res_53145;
            
            redout_53483 = redout_tmp_54079;
        }
        res_53142 = redout_53483;
        ((float *) mem_53912.mem)[i_53487] = res_53142;
    }
    
    int32_t iota_arg_53154 = N_52750 - n_52755;
    bool bounds_invalid_upwards_53155 = slt32(iota_arg_53154, 0);
    bool eq_x_zz_53156 = 0 == iota_arg_53154;
    bool not_p_53157 = !bounds_invalid_upwards_53155;
    bool p_and_eq_x_y_53158 = eq_x_zz_53156 && not_p_53157;
    bool dim_zzero_53159 = bounds_invalid_upwards_53155 || p_and_eq_x_y_53158;
    bool both_empty_53160 = eq_x_zz_53156 && dim_zzero_53159;
    bool eq_x_y_53161 = iota_arg_53154 == 0;
    bool p_and_eq_x_y_53162 = bounds_invalid_upwards_53155 && eq_x_y_53161;
    bool dim_match_53163 = not_p_53157 || p_and_eq_x_y_53162;
    bool empty_or_match_53164 = both_empty_53160 || dim_match_53163;
    bool empty_or_match_cert_53165;
    
    if (!empty_or_match_53164) {
        ctx->error = msgprintf("Error at %s:\n%s%s%s%d%s%s\n",
                               "bfast-irreg.fut:133:1-314:123 -> bfast-irreg.fut:242:22-31 -> /futlib/array.fut:61:1-62:12",
                               "Function return value does not match shape of type ",
                               "*", "[", iota_arg_53154, "]", "intrinsics.i32");
        if (memblock_unref(ctx, &mem_53912, "mem_53912") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53903, "mem_53903") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53900, "mem_53900") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53897, "mem_53897") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53885, "mem_53885") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53875, "mem_53875") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53870, "mem_53870") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53867, "mem_53867") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53860, "mem_53860") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53857, "mem_53857") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53851, "mem_53851") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53846, "mem_53846") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53841, "mem_53841") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53828, "mem_53828") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53813, "mem_53813") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53798, "mem_53798") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53787, "mem_53787") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53781, "mem_53781") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53777, "mem_53777") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53743, "mem_53743") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53726, "mem_53726") != 0)
            return 1;
        if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_53721,
                           "lifted_1_zlzb_arg_mem_53721") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54032, "out_mem_54032") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54030, "out_mem_54030") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54028, "out_mem_54028") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54026, "out_mem_54026") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54023, "out_mem_54023") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54021, "out_mem_54021") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54018, "out_mem_54018") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54015, "out_mem_54015") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54012, "out_mem_54012") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54009, "out_mem_54009") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54005, "out_mem_54005") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54001, "out_mem_54001") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53998, "out_mem_53998") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53996, "out_mem_53996") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53994, "out_mem_53994") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53991, "out_mem_53991") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53988, "out_mem_53988") != 0)
            return 1;
        return 1;
    }
    
    int32_t x_53167 = 1 + n_52755;
    bool index_certs_53168;
    
    if (!x_53029) {
        ctx->error = msgprintf("Error at %s:\n%s%d%s%d%s\n",
                               "bfast-irreg.fut:133:1-314:123 -> bfast-irreg.fut:238:15-242:32 -> bfast-irreg.fut:240:63-81",
                               "Index [", i_53028,
                               "] out of bounds for array of shape [", N_52750,
                               "].");
        if (memblock_unref(ctx, &mem_53912, "mem_53912") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53903, "mem_53903") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53900, "mem_53900") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53897, "mem_53897") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53885, "mem_53885") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53875, "mem_53875") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53870, "mem_53870") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53867, "mem_53867") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53860, "mem_53860") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53857, "mem_53857") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53851, "mem_53851") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53846, "mem_53846") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53841, "mem_53841") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53828, "mem_53828") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53813, "mem_53813") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53798, "mem_53798") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53787, "mem_53787") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53781, "mem_53781") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53777, "mem_53777") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53743, "mem_53743") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53726, "mem_53726") != 0)
            return 1;
        if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_53721,
                           "lifted_1_zlzb_arg_mem_53721") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54032, "out_mem_54032") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54030, "out_mem_54030") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54028, "out_mem_54028") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54026, "out_mem_54026") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54023, "out_mem_54023") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54021, "out_mem_54021") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54018, "out_mem_54018") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54015, "out_mem_54015") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54012, "out_mem_54012") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54009, "out_mem_54009") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54005, "out_mem_54005") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54001, "out_mem_54001") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53998, "out_mem_53998") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53996, "out_mem_53996") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53994, "out_mem_53994") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53991, "out_mem_53991") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53988, "out_mem_53988") != 0)
            return 1;
        return 1;
    }
    
    int32_t r32_arg_53169 = ((int32_t *) mappingindices_mem_53689.mem)[i_53028];
    float res_53170 = sitofp_i32_f32(r32_arg_53169);
    int64_t binop_x_53916 = sext_i32_i64(iota_arg_53154);
    int64_t bytes_53915 = 4 * binop_x_53916;
    struct memblock mem_53917;
    
    mem_53917.references = NULL;
    if (memblock_alloc(ctx, &mem_53917, bytes_53915, "mem_53917"))
        return 1;
    for (int32_t i_53491 = 0; i_53491 < iota_arg_53154; i_53491++) {
        int32_t t_53173 = x_53167 + i_53491;
        int32_t i_53174 = t_53173 - 1;
        int32_t time_53175 =
                ((int32_t *) mappingindices_mem_53689.mem)[i_53174];
        float res_53176 = sitofp_i32_f32(time_53175);
        float logplus_arg_53177 = res_53176 / res_53170;
        bool cond_53178 = 2.7182817F < logplus_arg_53177;
        float res_53179;
        
        if (cond_53178) {
            float res_53180;
            
            res_53180 = futrts_log32(logplus_arg_53177);
            res_53179 = res_53180;
        } else {
            res_53179 = 1.0F;
        }
        
        float res_53181;
        
        res_53181 = futrts_sqrt32(res_53179);
        
        float res_53182 = lam_52758 * res_53181;
        
        ((float *) mem_53917.mem)[i_53491] = res_53182;
    }
    
    bool empty_or_match_cert_53183;
    
    if (!empty_or_match_53164) {
        ctx->error = msgprintf("Error at %s:\n%s%s%s%d%s%s\n",
                               "bfast-irreg.fut:133:1-314:123 -> bfast-irreg.fut:252:38-308:9 -> /futlib/functional.fut:7:42-44 -> bfast-irreg.fut:299:33-53 -> /futlib/array.fut:66:1-67:19",
                               "Function return value does not match shape of type ",
                               "*", "[", iota_arg_53154, "]", "t");
        if (memblock_unref(ctx, &mem_53917, "mem_53917") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53912, "mem_53912") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53903, "mem_53903") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53900, "mem_53900") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53897, "mem_53897") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53885, "mem_53885") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53875, "mem_53875") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53870, "mem_53870") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53867, "mem_53867") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53860, "mem_53860") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53857, "mem_53857") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53851, "mem_53851") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53846, "mem_53846") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53841, "mem_53841") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53828, "mem_53828") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53813, "mem_53813") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53798, "mem_53798") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53787, "mem_53787") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53781, "mem_53781") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53777, "mem_53777") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53743, "mem_53743") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53726, "mem_53726") != 0)
            return 1;
        if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_53721,
                           "lifted_1_zlzb_arg_mem_53721") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54032, "out_mem_54032") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54030, "out_mem_54030") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54028, "out_mem_54028") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54026, "out_mem_54026") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54023, "out_mem_54023") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54021, "out_mem_54021") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54018, "out_mem_54018") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54015, "out_mem_54015") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54012, "out_mem_54012") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54009, "out_mem_54009") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54005, "out_mem_54005") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_54001, "out_mem_54001") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53998, "out_mem_53998") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53996, "out_mem_53996") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53994, "out_mem_53994") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53991, "out_mem_53991") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53988, "out_mem_53988") != 0)
            return 1;
        return 1;
    }
    
    int64_t binop_x_53923 = binop_x_53738 * binop_x_53916;
    int64_t bytes_53920 = 4 * binop_x_53923;
    struct memblock mem_53924;
    
    mem_53924.references = NULL;
    if (memblock_alloc(ctx, &mem_53924, bytes_53920, "mem_53924"))
        return 1;
    
    struct memblock mem_53929;
    
    mem_53929.references = NULL;
    if (memblock_alloc(ctx, &mem_53929, bytes_53920, "mem_53929"))
        return 1;
    
    struct memblock mem_53932;
    
    mem_53932.references = NULL;
    if (memblock_alloc(ctx, &mem_53932, bytes_53839, "mem_53932"))
        return 1;
    
    struct memblock mem_53935;
    
    mem_53935.references = NULL;
    if (memblock_alloc(ctx, &mem_53935, bytes_53839, "mem_53935"))
        return 1;
    
    struct memblock mem_53942;
    
    mem_53942.references = NULL;
    if (memblock_alloc(ctx, &mem_53942, bytes_53915, "mem_53942"))
        return 1;
    
    struct memblock mem_53947;
    
    mem_53947.references = NULL;
    if (memblock_alloc(ctx, &mem_53947, bytes_53915, "mem_53947"))
        return 1;
    
    struct memblock mem_53952;
    
    mem_53952.references = NULL;
    if (memblock_alloc(ctx, &mem_53952, bytes_53915, "mem_53952"))
        return 1;
    for (int32_t i_54081 = 0; i_54081 < iota_arg_53154; i_54081++) {
        ((float *) mem_53952.mem)[i_54081] = NAN;
    }
    
    struct memblock mem_53956;
    
    mem_53956.references = NULL;
    if (memblock_alloc(ctx, &mem_53956, bytes_53915, "mem_53956"))
        return 1;
    for (int32_t i_53524 = 0; i_53524 < m_52751; i_53524++) {
        int32_t x_53188 = ((int32_t *) mem_53841.mem)[i_53524];
        int32_t x_53189 = ((int32_t *) mem_53900.mem)[i_53524];
        float x_53190 = ((float *) mem_53903.mem)[i_53524];
        int32_t x_53191 = ((int32_t *) mem_53897.mem)[i_53524];
        float x_53192 = ((float *) mem_53912.mem)[i_53524];
        int32_t y_53195 = x_53188 - x_53189;
        float res_53196 = sitofp_i32_f32(x_53189);
        float res_53197;
        
        res_53197 = futrts_sqrt32(res_53196);
        
        float y_53198 = x_53190 * res_53197;
        float discard_53498;
        float scanacc_53494 = 0.0F;
        
        for (int32_t i_53496 = 0; i_53496 < iota_arg_53154; i_53496++) {
            bool cond_53246 = sle32(y_53195, i_53496);
            float res_53247;
            
            if (cond_53246) {
                res_53247 = 0.0F;
            } else {
                bool cond_53248 = i_53496 == 0;
                float res_53249;
                
                if (cond_53248) {
                    res_53249 = x_53192;
                } else {
                    int32_t x_53250 = x_53189 - x_53191;
                    int32_t i_53251 = x_53250 + i_53496;
                    float negate_arg_53252 = ((float *) mem_53846.mem)[i_53524 *
                                                                       N_52750 +
                                                                       i_53251];
                    float x_53253 = 0.0F - negate_arg_53252;
                    int32_t i_53254 = x_53189 + i_53496;
                    float y_53255 = ((float *) mem_53846.mem)[i_53524 *
                                                              N_52750 +
                                                              i_53254];
                    float res_53256 = x_53253 + y_53255;
                    
                    res_53249 = res_53256;
                }
                res_53247 = res_53249;
            }
            
            float res_53244 = res_53247 + scanacc_53494;
            
            ((float *) mem_53942.mem)[i_53496] = res_53244;
            
            float scanacc_tmp_54086 = res_53244;
            
            scanacc_53494 = scanacc_tmp_54086;
        }
        discard_53498 = scanacc_53494;
        
        bool acc0_53266;
        int32_t acc0_53267;
        float acc0_53268;
        bool redout_53500;
        int32_t redout_53501;
        float redout_53502;
        
        redout_53500 = 0;
        redout_53501 = -1;
        redout_53502 = 0.0F;
        for (int32_t i_53504 = 0; i_53504 < iota_arg_53154; i_53504++) {
            float x_53284 = ((float *) mem_53942.mem)[i_53504];
            float x_53285 = ((float *) mem_53917.mem)[i_53504];
            int32_t x_53286 = i_53504;
            float res_53288 = x_53284 / y_53198;
            bool cond_53289 = slt32(i_53504, y_53195);
            bool res_53290;
            
            res_53290 = futrts_isnan32(res_53288);
            
            bool res_53291 = !res_53290;
            bool x_53292 = cond_53289 && res_53291;
            float res_53293 = (float) fabs(res_53288);
            bool res_53294 = x_53285 < res_53293;
            bool x_53295 = x_53292 && res_53294;
            float res_53296;
            
            if (cond_53289) {
                res_53296 = res_53288;
            } else {
                res_53296 = 0.0F;
            }
            
            bool res_53276;
            int32_t res_53277;
            
            if (redout_53500) {
                res_53276 = redout_53500;
                res_53277 = redout_53501;
            } else {
                bool x_53279 = !x_53295;
                bool y_53280 = x_53279 && redout_53500;
                bool res_53281 = y_53280 || x_53295;
                int32_t res_53282;
                
                if (x_53295) {
                    res_53282 = x_53286;
                } else {
                    res_53282 = redout_53501;
                }
                res_53276 = res_53281;
                res_53277 = res_53282;
            }
            
            float res_53283 = res_53296 + redout_53502;
            
            ((float *) mem_53947.mem)[i_53504] = res_53288;
            
            bool redout_tmp_54088 = res_53276;
            int32_t redout_tmp_54089 = res_53277;
            float redout_tmp_54090;
            
            redout_tmp_54090 = res_53283;
            redout_53500 = redout_tmp_54088;
            redout_53501 = redout_tmp_54089;
            redout_53502 = redout_tmp_54090;
        }
        acc0_53266 = redout_53500;
        acc0_53267 = redout_53501;
        acc0_53268 = redout_53502;
        memmove(mem_53929.mem + i_53524 * iota_arg_53154 * 4, mem_53947.mem + 0,
                iota_arg_53154 * sizeof(float));
        
        int32_t res_53307;
        
        if (acc0_53266) {
            res_53307 = acc0_53267;
        } else {
            res_53307 = -1;
        }
        
        bool cond_53314 = !acc0_53266;
        int32_t fst_breakzq_53315;
        
        if (cond_53314) {
            fst_breakzq_53315 = -1;
        } else {
            bool cond_53316 = slt32(res_53307, y_53195);
            int32_t res_53317;
            
            if (cond_53316) {
                int32_t i_53318 = x_53189 + res_53307;
                int32_t x_53319 = ((int32_t *) mem_53851.mem)[i_53524 *
                                                              N_52750 +
                                                              i_53318];
                int32_t res_53320 = x_53319 - n_52755;
                
                res_53317 = res_53320;
            } else {
                res_53317 = -1;
            }
            
            int32_t x_53321 = res_53317 - 1;
            int32_t x_53322 = sdiv32(x_53321, 2);
            int32_t x_53323 = 2 * x_53322;
            int32_t res_53324 = 1 + x_53323;
            
            fst_breakzq_53315 = res_53324;
        }
        
        bool cond_53325 = sle32(x_53189, 5);
        bool res_53326 = sle32(y_53195, 5);
        bool x_53327 = !cond_53325;
        bool y_53328 = res_53326 && x_53327;
        bool cond_53329 = cond_53325 || y_53328;
        int32_t fst_breakzq_53330;
        
        if (cond_53329) {
            fst_breakzq_53330 = -2;
        } else {
            fst_breakzq_53330 = fst_breakzq_53315;
        }
        memmove(mem_53924.mem + i_53524 * iota_arg_53154 * 4, mem_53952.mem + 0,
                iota_arg_53154 * sizeof(float));
        for (int32_t write_iter_53506 = 0; write_iter_53506 < iota_arg_53154;
             write_iter_53506++) {
            bool cond_53335 = slt32(write_iter_53506, y_53195);
            int32_t res_53336;
            
            if (cond_53335) {
                int32_t i_53337 = x_53189 + write_iter_53506;
                int32_t x_53338 = ((int32_t *) mem_53851.mem)[i_53524 *
                                                              N_52750 +
                                                              i_53337];
                int32_t res_53339 = x_53338 - n_52755;
                
                res_53336 = res_53339;
            } else {
                res_53336 = -1;
            }
            
            bool less_than_zzero_53510 = slt32(res_53336, 0);
            bool greater_than_sizze_53511 = sle32(iota_arg_53154, res_53336);
            bool outside_bounds_dim_53512 = less_than_zzero_53510 ||
                 greater_than_sizze_53511;
            
            memmove(mem_53956.mem + 0, mem_53924.mem + i_53524 *
                    iota_arg_53154 * 4, iota_arg_53154 * sizeof(float));
            
            struct memblock write_out_mem_53960;
            
            write_out_mem_53960.references = NULL;
            if (outside_bounds_dim_53512) {
                if (memblock_set(ctx, &write_out_mem_53960, &mem_53956,
                                 "mem_53956") != 0)
                    return 1;
            } else {
                struct memblock mem_53959;
                
                mem_53959.references = NULL;
                if (memblock_alloc(ctx, &mem_53959, bytes_53915, "mem_53959"))
                    return 1;
                memmove(mem_53959.mem + 0, mem_53924.mem + i_53524 *
                        iota_arg_53154 * 4, iota_arg_53154 * sizeof(float));
                memmove(mem_53959.mem + res_53336 * 4, mem_53947.mem +
                        write_iter_53506 * 4, sizeof(float));
                if (memblock_set(ctx, &write_out_mem_53960, &mem_53959,
                                 "mem_53959") != 0)
                    return 1;
                if (memblock_unref(ctx, &mem_53959, "mem_53959") != 0)
                    return 1;
            }
            memmove(mem_53924.mem + i_53524 * iota_arg_53154 * 4,
                    write_out_mem_53960.mem + 0, iota_arg_53154 *
                    sizeof(float));
            if (memblock_unref(ctx, &write_out_mem_53960,
                               "write_out_mem_53960") != 0)
                return 1;
            if (memblock_unref(ctx, &write_out_mem_53960,
                               "write_out_mem_53960") != 0)
                return 1;
        }
        ((int32_t *) mem_53932.mem)[i_53524] = fst_breakzq_53330;
        ((float *) mem_53935.mem)[i_53524] = acc0_53268;
    }
    if (memblock_unref(ctx, &mem_53917, "mem_53917") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53942, "mem_53942") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53947, "mem_53947") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53952, "mem_53952") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53956, "mem_53956") != 0)
        return 1;
    out_arrsizze_53989 = m_52751;
    out_arrsizze_53990 = iota_arg_53154;
    out_arrsizze_53992 = m_52751;
    out_arrsizze_53993 = iota_arg_53154;
    out_arrsizze_53995 = m_52751;
    out_arrsizze_53997 = m_52751;
    out_arrsizze_53999 = N_52750;
    out_arrsizze_54000 = k2p2zq_52774;
    out_arrsizze_54002 = m_52751;
    out_arrsizze_54003 = k2p2zq_52774;
    out_arrsizze_54004 = k2p2zq_52774;
    out_arrsizze_54006 = m_52751;
    out_arrsizze_54007 = k2p2zq_52774;
    out_arrsizze_54008 = j_m_i_52908;
    out_arrsizze_54010 = m_52751;
    out_arrsizze_54011 = k2p2zq_52774;
    out_arrsizze_54013 = m_52751;
    out_arrsizze_54014 = k2p2zq_52774;
    out_arrsizze_54016 = m_52751;
    out_arrsizze_54017 = N_52750;
    out_arrsizze_54019 = m_52751;
    out_arrsizze_54020 = N_52750;
    out_arrsizze_54022 = m_52751;
    out_arrsizze_54024 = m_52751;
    out_arrsizze_54025 = N_52750;
    out_arrsizze_54027 = m_52751;
    out_arrsizze_54029 = m_52751;
    out_arrsizze_54031 = m_52751;
    out_arrsizze_54033 = m_52751;
    if (memblock_set(ctx, &out_mem_53988, &mem_53924, "mem_53924") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_53991, &mem_53929, "mem_53929") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_53994, &mem_53932, "mem_53932") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_53996, &mem_53935, "mem_53935") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_53998, &mem_53726, "mem_53726") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_54001, &mem_53743, "mem_53743") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_54005, &mem_53777, "mem_53777") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_54009, &mem_53798, "mem_53798") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_54012, &mem_53813, "mem_53813") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_54015, &mem_53828, "mem_53828") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_54018, &mem_53846, "mem_53846") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_54021, &mem_53841, "mem_53841") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_54023, &mem_53851, "mem_53851") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_54026, &mem_53897, "mem_53897") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_54028, &mem_53900, "mem_53900") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_54030, &mem_53903, "mem_53903") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_54032, &mem_53912, "mem_53912") != 0)
        return 1;
    (*out_mem_p_54093).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_54093, &out_mem_53988, "out_mem_53988") !=
        0)
        return 1;
    *out_out_arrsizze_54094 = out_arrsizze_53989;
    *out_out_arrsizze_54095 = out_arrsizze_53990;
    (*out_mem_p_54096).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_54096, &out_mem_53991, "out_mem_53991") !=
        0)
        return 1;
    *out_out_arrsizze_54097 = out_arrsizze_53992;
    *out_out_arrsizze_54098 = out_arrsizze_53993;
    (*out_mem_p_54099).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_54099, &out_mem_53994, "out_mem_53994") !=
        0)
        return 1;
    *out_out_arrsizze_54100 = out_arrsizze_53995;
    (*out_mem_p_54101).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_54101, &out_mem_53996, "out_mem_53996") !=
        0)
        return 1;
    *out_out_arrsizze_54102 = out_arrsizze_53997;
    (*out_mem_p_54103).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_54103, &out_mem_53998, "out_mem_53998") !=
        0)
        return 1;
    *out_out_arrsizze_54104 = out_arrsizze_53999;
    *out_out_arrsizze_54105 = out_arrsizze_54000;
    (*out_mem_p_54106).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_54106, &out_mem_54001, "out_mem_54001") !=
        0)
        return 1;
    *out_out_arrsizze_54107 = out_arrsizze_54002;
    *out_out_arrsizze_54108 = out_arrsizze_54003;
    *out_out_arrsizze_54109 = out_arrsizze_54004;
    (*out_mem_p_54110).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_54110, &out_mem_54005, "out_mem_54005") !=
        0)
        return 1;
    *out_out_arrsizze_54111 = out_arrsizze_54006;
    *out_out_arrsizze_54112 = out_arrsizze_54007;
    *out_out_arrsizze_54113 = out_arrsizze_54008;
    (*out_mem_p_54114).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_54114, &out_mem_54009, "out_mem_54009") !=
        0)
        return 1;
    *out_out_arrsizze_54115 = out_arrsizze_54010;
    *out_out_arrsizze_54116 = out_arrsizze_54011;
    (*out_mem_p_54117).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_54117, &out_mem_54012, "out_mem_54012") !=
        0)
        return 1;
    *out_out_arrsizze_54118 = out_arrsizze_54013;
    *out_out_arrsizze_54119 = out_arrsizze_54014;
    (*out_mem_p_54120).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_54120, &out_mem_54015, "out_mem_54015") !=
        0)
        return 1;
    *out_out_arrsizze_54121 = out_arrsizze_54016;
    *out_out_arrsizze_54122 = out_arrsizze_54017;
    (*out_mem_p_54123).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_54123, &out_mem_54018, "out_mem_54018") !=
        0)
        return 1;
    *out_out_arrsizze_54124 = out_arrsizze_54019;
    *out_out_arrsizze_54125 = out_arrsizze_54020;
    (*out_mem_p_54126).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_54126, &out_mem_54021, "out_mem_54021") !=
        0)
        return 1;
    *out_out_arrsizze_54127 = out_arrsizze_54022;
    (*out_mem_p_54128).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_54128, &out_mem_54023, "out_mem_54023") !=
        0)
        return 1;
    *out_out_arrsizze_54129 = out_arrsizze_54024;
    *out_out_arrsizze_54130 = out_arrsizze_54025;
    (*out_mem_p_54131).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_54131, &out_mem_54026, "out_mem_54026") !=
        0)
        return 1;
    *out_out_arrsizze_54132 = out_arrsizze_54027;
    (*out_mem_p_54133).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_54133, &out_mem_54028, "out_mem_54028") !=
        0)
        return 1;
    *out_out_arrsizze_54134 = out_arrsizze_54029;
    (*out_mem_p_54135).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_54135, &out_mem_54030, "out_mem_54030") !=
        0)
        return 1;
    *out_out_arrsizze_54136 = out_arrsizze_54031;
    (*out_mem_p_54137).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_54137, &out_mem_54032, "out_mem_54032") !=
        0)
        return 1;
    *out_out_arrsizze_54138 = out_arrsizze_54033;
    if (memblock_unref(ctx, &mem_53956, "mem_53956") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53952, "mem_53952") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53947, "mem_53947") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53942, "mem_53942") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53935, "mem_53935") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53932, "mem_53932") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53929, "mem_53929") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53924, "mem_53924") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53917, "mem_53917") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53912, "mem_53912") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53903, "mem_53903") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53900, "mem_53900") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53897, "mem_53897") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53885, "mem_53885") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53875, "mem_53875") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53870, "mem_53870") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53867, "mem_53867") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53860, "mem_53860") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53857, "mem_53857") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53851, "mem_53851") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53846, "mem_53846") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53841, "mem_53841") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53828, "mem_53828") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53813, "mem_53813") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53798, "mem_53798") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53787, "mem_53787") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53781, "mem_53781") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53777, "mem_53777") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53743, "mem_53743") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53726, "mem_53726") != 0)
        return 1;
    if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_53721,
                       "lifted_1_zlzb_arg_mem_53721") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_54032, "out_mem_54032") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_54030, "out_mem_54030") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_54028, "out_mem_54028") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_54026, "out_mem_54026") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_54023, "out_mem_54023") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_54021, "out_mem_54021") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_54018, "out_mem_54018") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_54015, "out_mem_54015") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_54012, "out_mem_54012") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_54009, "out_mem_54009") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_54005, "out_mem_54005") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_54001, "out_mem_54001") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_53998, "out_mem_53998") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_53996, "out_mem_53996") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_53994, "out_mem_53994") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_53991, "out_mem_53991") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_53988, "out_mem_53988") != 0)
        return 1;
    return 0;
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
int futhark_entry_main(struct futhark_context *ctx,
                       struct futhark_f32_2d **out0,
                       struct futhark_f32_2d **out1,
                       struct futhark_i32_1d **out2,
                       struct futhark_f32_1d **out3,
                       struct futhark_f32_2d **out4,
                       struct futhark_f32_3d **out5,
                       struct futhark_f32_3d **out6,
                       struct futhark_f32_2d **out7,
                       struct futhark_f32_2d **out8,
                       struct futhark_f32_2d **out9,
                       struct futhark_f32_2d **out10,
                       struct futhark_i32_1d **out11,
                       struct futhark_i32_2d **out12,
                       struct futhark_i32_1d **out13,
                       struct futhark_i32_1d **out14,
                       struct futhark_f32_1d **out15,
                       struct futhark_f32_1d **out16, const int32_t in0, const
                       int32_t in1, const int32_t in2, const float in3, const
                       float in4, const float in5, const
                       struct futhark_i32_1d *in6, const
                       struct futhark_f32_2d *in7)
{
    struct memblock mappingindices_mem_53689;
    
    mappingindices_mem_53689.references = NULL;
    
    struct memblock images_mem_53690;
    
    images_mem_53690.references = NULL;
    
    int32_t N_52750;
    int32_t m_52751;
    int32_t N_52752;
    int32_t trend_52753;
    int32_t k_52754;
    int32_t n_52755;
    float freq_52756;
    float hfrac_52757;
    float lam_52758;
    struct memblock out_mem_53988;
    
    out_mem_53988.references = NULL;
    
    int32_t out_arrsizze_53989;
    int32_t out_arrsizze_53990;
    struct memblock out_mem_53991;
    
    out_mem_53991.references = NULL;
    
    int32_t out_arrsizze_53992;
    int32_t out_arrsizze_53993;
    struct memblock out_mem_53994;
    
    out_mem_53994.references = NULL;
    
    int32_t out_arrsizze_53995;
    struct memblock out_mem_53996;
    
    out_mem_53996.references = NULL;
    
    int32_t out_arrsizze_53997;
    struct memblock out_mem_53998;
    
    out_mem_53998.references = NULL;
    
    int32_t out_arrsizze_53999;
    int32_t out_arrsizze_54000;
    struct memblock out_mem_54001;
    
    out_mem_54001.references = NULL;
    
    int32_t out_arrsizze_54002;
    int32_t out_arrsizze_54003;
    int32_t out_arrsizze_54004;
    struct memblock out_mem_54005;
    
    out_mem_54005.references = NULL;
    
    int32_t out_arrsizze_54006;
    int32_t out_arrsizze_54007;
    int32_t out_arrsizze_54008;
    struct memblock out_mem_54009;
    
    out_mem_54009.references = NULL;
    
    int32_t out_arrsizze_54010;
    int32_t out_arrsizze_54011;
    struct memblock out_mem_54012;
    
    out_mem_54012.references = NULL;
    
    int32_t out_arrsizze_54013;
    int32_t out_arrsizze_54014;
    struct memblock out_mem_54015;
    
    out_mem_54015.references = NULL;
    
    int32_t out_arrsizze_54016;
    int32_t out_arrsizze_54017;
    struct memblock out_mem_54018;
    
    out_mem_54018.references = NULL;
    
    int32_t out_arrsizze_54019;
    int32_t out_arrsizze_54020;
    struct memblock out_mem_54021;
    
    out_mem_54021.references = NULL;
    
    int32_t out_arrsizze_54022;
    struct memblock out_mem_54023;
    
    out_mem_54023.references = NULL;
    
    int32_t out_arrsizze_54024;
    int32_t out_arrsizze_54025;
    struct memblock out_mem_54026;
    
    out_mem_54026.references = NULL;
    
    int32_t out_arrsizze_54027;
    struct memblock out_mem_54028;
    
    out_mem_54028.references = NULL;
    
    int32_t out_arrsizze_54029;
    struct memblock out_mem_54030;
    
    out_mem_54030.references = NULL;
    
    int32_t out_arrsizze_54031;
    struct memblock out_mem_54032;
    
    out_mem_54032.references = NULL;
    
    int32_t out_arrsizze_54033;
    
    lock_lock(&ctx->lock);
    trend_52753 = in0;
    k_52754 = in1;
    n_52755 = in2;
    freq_52756 = in3;
    hfrac_52757 = in4;
    lam_52758 = in5;
    mappingindices_mem_53689 = in6->mem;
    N_52750 = in6->shape[0];
    images_mem_53690 = in7->mem;
    m_52751 = in7->shape[0];
    N_52752 = in7->shape[1];
    
    int ret = futrts_main(ctx, &out_mem_53988, &out_arrsizze_53989,
                          &out_arrsizze_53990, &out_mem_53991,
                          &out_arrsizze_53992, &out_arrsizze_53993,
                          &out_mem_53994, &out_arrsizze_53995, &out_mem_53996,
                          &out_arrsizze_53997, &out_mem_53998,
                          &out_arrsizze_53999, &out_arrsizze_54000,
                          &out_mem_54001, &out_arrsizze_54002,
                          &out_arrsizze_54003, &out_arrsizze_54004,
                          &out_mem_54005, &out_arrsizze_54006,
                          &out_arrsizze_54007, &out_arrsizze_54008,
                          &out_mem_54009, &out_arrsizze_54010,
                          &out_arrsizze_54011, &out_mem_54012,
                          &out_arrsizze_54013, &out_arrsizze_54014,
                          &out_mem_54015, &out_arrsizze_54016,
                          &out_arrsizze_54017, &out_mem_54018,
                          &out_arrsizze_54019, &out_arrsizze_54020,
                          &out_mem_54021, &out_arrsizze_54022, &out_mem_54023,
                          &out_arrsizze_54024, &out_arrsizze_54025,
                          &out_mem_54026, &out_arrsizze_54027, &out_mem_54028,
                          &out_arrsizze_54029, &out_mem_54030,
                          &out_arrsizze_54031, &out_mem_54032,
                          &out_arrsizze_54033, mappingindices_mem_53689,
                          images_mem_53690, N_52750, m_52751, N_52752,
                          trend_52753, k_52754, n_52755, freq_52756,
                          hfrac_52757, lam_52758);
    
    if (ret == 0) {
        assert((*out0 =
                (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) !=
            NULL);
        (*out0)->mem = out_mem_53988;
        (*out0)->shape[0] = out_arrsizze_53989;
        (*out0)->shape[1] = out_arrsizze_53990;
        assert((*out1 =
                (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) !=
            NULL);
        (*out1)->mem = out_mem_53991;
        (*out1)->shape[0] = out_arrsizze_53992;
        (*out1)->shape[1] = out_arrsizze_53993;
        assert((*out2 =
                (struct futhark_i32_1d *) malloc(sizeof(struct futhark_i32_1d))) !=
            NULL);
        (*out2)->mem = out_mem_53994;
        (*out2)->shape[0] = out_arrsizze_53995;
        assert((*out3 =
                (struct futhark_f32_1d *) malloc(sizeof(struct futhark_f32_1d))) !=
            NULL);
        (*out3)->mem = out_mem_53996;
        (*out3)->shape[0] = out_arrsizze_53997;
        assert((*out4 =
                (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) !=
            NULL);
        (*out4)->mem = out_mem_53998;
        (*out4)->shape[0] = out_arrsizze_53999;
        (*out4)->shape[1] = out_arrsizze_54000;
        assert((*out5 =
                (struct futhark_f32_3d *) malloc(sizeof(struct futhark_f32_3d))) !=
            NULL);
        (*out5)->mem = out_mem_54001;
        (*out5)->shape[0] = out_arrsizze_54002;
        (*out5)->shape[1] = out_arrsizze_54003;
        (*out5)->shape[2] = out_arrsizze_54004;
        assert((*out6 =
                (struct futhark_f32_3d *) malloc(sizeof(struct futhark_f32_3d))) !=
            NULL);
        (*out6)->mem = out_mem_54005;
        (*out6)->shape[0] = out_arrsizze_54006;
        (*out6)->shape[1] = out_arrsizze_54007;
        (*out6)->shape[2] = out_arrsizze_54008;
        assert((*out7 =
                (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) !=
            NULL);
        (*out7)->mem = out_mem_54009;
        (*out7)->shape[0] = out_arrsizze_54010;
        (*out7)->shape[1] = out_arrsizze_54011;
        assert((*out8 =
                (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) !=
            NULL);
        (*out8)->mem = out_mem_54012;
        (*out8)->shape[0] = out_arrsizze_54013;
        (*out8)->shape[1] = out_arrsizze_54014;
        assert((*out9 =
                (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) !=
            NULL);
        (*out9)->mem = out_mem_54015;
        (*out9)->shape[0] = out_arrsizze_54016;
        (*out9)->shape[1] = out_arrsizze_54017;
        assert((*out10 =
                (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) !=
            NULL);
        (*out10)->mem = out_mem_54018;
        (*out10)->shape[0] = out_arrsizze_54019;
        (*out10)->shape[1] = out_arrsizze_54020;
        assert((*out11 =
                (struct futhark_i32_1d *) malloc(sizeof(struct futhark_i32_1d))) !=
            NULL);
        (*out11)->mem = out_mem_54021;
        (*out11)->shape[0] = out_arrsizze_54022;
        assert((*out12 =
                (struct futhark_i32_2d *) malloc(sizeof(struct futhark_i32_2d))) !=
            NULL);
        (*out12)->mem = out_mem_54023;
        (*out12)->shape[0] = out_arrsizze_54024;
        (*out12)->shape[1] = out_arrsizze_54025;
        assert((*out13 =
                (struct futhark_i32_1d *) malloc(sizeof(struct futhark_i32_1d))) !=
            NULL);
        (*out13)->mem = out_mem_54026;
        (*out13)->shape[0] = out_arrsizze_54027;
        assert((*out14 =
                (struct futhark_i32_1d *) malloc(sizeof(struct futhark_i32_1d))) !=
            NULL);
        (*out14)->mem = out_mem_54028;
        (*out14)->shape[0] = out_arrsizze_54029;
        assert((*out15 =
                (struct futhark_f32_1d *) malloc(sizeof(struct futhark_f32_1d))) !=
            NULL);
        (*out15)->mem = out_mem_54030;
        (*out15)->shape[0] = out_arrsizze_54031;
        assert((*out16 =
                (struct futhark_f32_1d *) malloc(sizeof(struct futhark_f32_1d))) !=
            NULL);
        (*out16)->mem = out_mem_54032;
        (*out16)->shape[0] = out_arrsizze_54033;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
