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
                       struct futhark_i32_1d **out0,
                       struct futhark_f32_1d **out1,
                       struct futhark_f32_2d **out2,
                       struct futhark_f32_3d **out3,
                       struct futhark_f32_3d **out4,
                       struct futhark_f32_2d **out5,
                       struct futhark_f32_2d **out6,
                       struct futhark_f32_2d **out7,
                       struct futhark_f32_2d **out8,
                       struct futhark_i32_1d **out9,
                       struct futhark_i32_2d **out10,
                       struct futhark_i32_1d **out11,
                       struct futhark_i32_1d **out12,
                       struct futhark_f32_1d **out13,
                       struct futhark_f32_1d **out14, const int32_t in0, const
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
    
    int32_t read_value_53712;
    
    if (read_scalar(&i32_info, &read_value_53712) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 0,
              i32_info.type_name, strerror(errno));
    
    int32_t read_value_53713;
    
    if (read_scalar(&i32_info, &read_value_53713) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 1,
              i32_info.type_name, strerror(errno));
    
    int32_t read_value_53714;
    
    if (read_scalar(&i32_info, &read_value_53714) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 2,
              i32_info.type_name, strerror(errno));
    
    float read_value_53715;
    
    if (read_scalar(&f32_info, &read_value_53715) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 3,
              f32_info.type_name, strerror(errno));
    
    float read_value_53716;
    
    if (read_scalar(&f32_info, &read_value_53716) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 4,
              f32_info.type_name, strerror(errno));
    
    float read_value_53717;
    
    if (read_scalar(&f32_info, &read_value_53717) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 5,
              f32_info.type_name, strerror(errno));
    
    struct futhark_i32_1d *read_value_53718;
    int64_t read_shape_53719[1];
    int32_t *read_arr_53720 = NULL;
    
    errno = 0;
    if (read_array(&i32_info, (void **) &read_arr_53720, read_shape_53719, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 6, "[]",
              i32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_53721;
    int64_t read_shape_53722[2];
    float *read_arr_53723 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_53723, read_shape_53722, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 7, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_i32_1d *result_53724;
    struct futhark_f32_1d *result_53725;
    struct futhark_f32_2d *result_53726;
    struct futhark_f32_3d *result_53727;
    struct futhark_f32_3d *result_53728;
    struct futhark_f32_2d *result_53729;
    struct futhark_f32_2d *result_53730;
    struct futhark_f32_2d *result_53731;
    struct futhark_f32_2d *result_53732;
    struct futhark_i32_1d *result_53733;
    struct futhark_i32_2d *result_53734;
    struct futhark_i32_1d *result_53735;
    struct futhark_i32_1d *result_53736;
    struct futhark_f32_1d *result_53737;
    struct futhark_f32_1d *result_53738;
    
    if (perform_warmup) {
        time_runs = 0;
        
        int r;
        
        ;
        ;
        ;
        ;
        ;
        ;
        assert((read_value_53718 = futhark_new_i32_1d(ctx, read_arr_53720,
                                                      read_shape_53719[0])) !=
            0);
        assert((read_value_53721 = futhark_new_f32_2d(ctx, read_arr_53723,
                                                      read_shape_53722[0],
                                                      read_shape_53722[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_main(ctx, &result_53724, &result_53725, &result_53726,
                               &result_53727, &result_53728, &result_53729,
                               &result_53730, &result_53731, &result_53732,
                               &result_53733, &result_53734, &result_53735,
                               &result_53736, &result_53737, &result_53738,
                               read_value_53712, read_value_53713,
                               read_value_53714, read_value_53715,
                               read_value_53716, read_value_53717,
                               read_value_53718, read_value_53721);
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
        assert(futhark_free_i32_1d(ctx, read_value_53718) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_53721) == 0);
        assert(futhark_free_i32_1d(ctx, result_53724) == 0);
        assert(futhark_free_f32_1d(ctx, result_53725) == 0);
        assert(futhark_free_f32_2d(ctx, result_53726) == 0);
        assert(futhark_free_f32_3d(ctx, result_53727) == 0);
        assert(futhark_free_f32_3d(ctx, result_53728) == 0);
        assert(futhark_free_f32_2d(ctx, result_53729) == 0);
        assert(futhark_free_f32_2d(ctx, result_53730) == 0);
        assert(futhark_free_f32_2d(ctx, result_53731) == 0);
        assert(futhark_free_f32_2d(ctx, result_53732) == 0);
        assert(futhark_free_i32_1d(ctx, result_53733) == 0);
        assert(futhark_free_i32_2d(ctx, result_53734) == 0);
        assert(futhark_free_i32_1d(ctx, result_53735) == 0);
        assert(futhark_free_i32_1d(ctx, result_53736) == 0);
        assert(futhark_free_f32_1d(ctx, result_53737) == 0);
        assert(futhark_free_f32_1d(ctx, result_53738) == 0);
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
        assert((read_value_53718 = futhark_new_i32_1d(ctx, read_arr_53720,
                                                      read_shape_53719[0])) !=
            0);
        assert((read_value_53721 = futhark_new_f32_2d(ctx, read_arr_53723,
                                                      read_shape_53722[0],
                                                      read_shape_53722[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_main(ctx, &result_53724, &result_53725, &result_53726,
                               &result_53727, &result_53728, &result_53729,
                               &result_53730, &result_53731, &result_53732,
                               &result_53733, &result_53734, &result_53735,
                               &result_53736, &result_53737, &result_53738,
                               read_value_53712, read_value_53713,
                               read_value_53714, read_value_53715,
                               read_value_53716, read_value_53717,
                               read_value_53718, read_value_53721);
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
        assert(futhark_free_i32_1d(ctx, read_value_53718) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_53721) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_i32_1d(ctx, result_53724) == 0);
            assert(futhark_free_f32_1d(ctx, result_53725) == 0);
            assert(futhark_free_f32_2d(ctx, result_53726) == 0);
            assert(futhark_free_f32_3d(ctx, result_53727) == 0);
            assert(futhark_free_f32_3d(ctx, result_53728) == 0);
            assert(futhark_free_f32_2d(ctx, result_53729) == 0);
            assert(futhark_free_f32_2d(ctx, result_53730) == 0);
            assert(futhark_free_f32_2d(ctx, result_53731) == 0);
            assert(futhark_free_f32_2d(ctx, result_53732) == 0);
            assert(futhark_free_i32_1d(ctx, result_53733) == 0);
            assert(futhark_free_i32_2d(ctx, result_53734) == 0);
            assert(futhark_free_i32_1d(ctx, result_53735) == 0);
            assert(futhark_free_i32_1d(ctx, result_53736) == 0);
            assert(futhark_free_f32_1d(ctx, result_53737) == 0);
            assert(futhark_free_f32_1d(ctx, result_53738) == 0);
        }
    }
    ;
    ;
    ;
    ;
    ;
    ;
    free(read_arr_53720);
    free(read_arr_53723);
    if (binary_output)
        set_binary_mode(stdout);
    {
        int32_t *arr = calloc(sizeof(int32_t), futhark_shape_i32_1d(ctx,
                                                                    result_53724)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_i32_1d(ctx, result_53724, arr) == 0);
        write_array(stdout, binary_output, &i32_info, arr,
                    futhark_shape_i32_1d(ctx, result_53724), 1);
        free(arr);
    }
    printf("\n");
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_1d(ctx,
                                                                result_53725)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_1d(ctx, result_53725, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_1d(ctx, result_53725), 1);
        free(arr);
    }
    printf("\n");
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_2d(ctx,
                                                                result_53726)[0] *
                            futhark_shape_f32_2d(ctx, result_53726)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_2d(ctx, result_53726, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_2d(ctx, result_53726), 2);
        free(arr);
    }
    printf("\n");
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_3d(ctx,
                                                                result_53727)[0] *
                            futhark_shape_f32_3d(ctx, result_53727)[1] *
                            futhark_shape_f32_3d(ctx, result_53727)[2]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_3d(ctx, result_53727, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_3d(ctx, result_53727), 3);
        free(arr);
    }
    printf("\n");
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_3d(ctx,
                                                                result_53728)[0] *
                            futhark_shape_f32_3d(ctx, result_53728)[1] *
                            futhark_shape_f32_3d(ctx, result_53728)[2]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_3d(ctx, result_53728, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_3d(ctx, result_53728), 3);
        free(arr);
    }
    printf("\n");
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_2d(ctx,
                                                                result_53729)[0] *
                            futhark_shape_f32_2d(ctx, result_53729)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_2d(ctx, result_53729, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_2d(ctx, result_53729), 2);
        free(arr);
    }
    printf("\n");
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_2d(ctx,
                                                                result_53730)[0] *
                            futhark_shape_f32_2d(ctx, result_53730)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_2d(ctx, result_53730, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_2d(ctx, result_53730), 2);
        free(arr);
    }
    printf("\n");
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_2d(ctx,
                                                                result_53731)[0] *
                            futhark_shape_f32_2d(ctx, result_53731)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_2d(ctx, result_53731, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_2d(ctx, result_53731), 2);
        free(arr);
    }
    printf("\n");
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_2d(ctx,
                                                                result_53732)[0] *
                            futhark_shape_f32_2d(ctx, result_53732)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_2d(ctx, result_53732, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_2d(ctx, result_53732), 2);
        free(arr);
    }
    printf("\n");
    {
        int32_t *arr = calloc(sizeof(int32_t), futhark_shape_i32_1d(ctx,
                                                                    result_53733)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_i32_1d(ctx, result_53733, arr) == 0);
        write_array(stdout, binary_output, &i32_info, arr,
                    futhark_shape_i32_1d(ctx, result_53733), 1);
        free(arr);
    }
    printf("\n");
    {
        int32_t *arr = calloc(sizeof(int32_t), futhark_shape_i32_2d(ctx,
                                                                    result_53734)[0] *
                              futhark_shape_i32_2d(ctx, result_53734)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_i32_2d(ctx, result_53734, arr) == 0);
        write_array(stdout, binary_output, &i32_info, arr,
                    futhark_shape_i32_2d(ctx, result_53734), 2);
        free(arr);
    }
    printf("\n");
    {
        int32_t *arr = calloc(sizeof(int32_t), futhark_shape_i32_1d(ctx,
                                                                    result_53735)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_i32_1d(ctx, result_53735, arr) == 0);
        write_array(stdout, binary_output, &i32_info, arr,
                    futhark_shape_i32_1d(ctx, result_53735), 1);
        free(arr);
    }
    printf("\n");
    {
        int32_t *arr = calloc(sizeof(int32_t), futhark_shape_i32_1d(ctx,
                                                                    result_53736)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_i32_1d(ctx, result_53736, arr) == 0);
        write_array(stdout, binary_output, &i32_info, arr,
                    futhark_shape_i32_1d(ctx, result_53736), 1);
        free(arr);
    }
    printf("\n");
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_1d(ctx,
                                                                result_53737)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_1d(ctx, result_53737, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_1d(ctx, result_53737), 1);
        free(arr);
    }
    printf("\n");
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_1d(ctx,
                                                                result_53738)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_1d(ctx, result_53738, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_1d(ctx, result_53738), 1);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_i32_1d(ctx, result_53724) == 0);
    assert(futhark_free_f32_1d(ctx, result_53725) == 0);
    assert(futhark_free_f32_2d(ctx, result_53726) == 0);
    assert(futhark_free_f32_3d(ctx, result_53727) == 0);
    assert(futhark_free_f32_3d(ctx, result_53728) == 0);
    assert(futhark_free_f32_2d(ctx, result_53729) == 0);
    assert(futhark_free_f32_2d(ctx, result_53730) == 0);
    assert(futhark_free_f32_2d(ctx, result_53731) == 0);
    assert(futhark_free_f32_2d(ctx, result_53732) == 0);
    assert(futhark_free_i32_1d(ctx, result_53733) == 0);
    assert(futhark_free_i32_2d(ctx, result_53734) == 0);
    assert(futhark_free_i32_1d(ctx, result_53735) == 0);
    assert(futhark_free_i32_1d(ctx, result_53736) == 0);
    assert(futhark_free_f32_1d(ctx, result_53737) == 0);
    assert(futhark_free_f32_1d(ctx, result_53738) == 0);
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
                       struct memblock *out_mem_p_53672,
                       int32_t *out_out_arrsizze_53673,
                       struct memblock *out_mem_p_53674,
                       int32_t *out_out_arrsizze_53675,
                       struct memblock *out_mem_p_53676,
                       int32_t *out_out_arrsizze_53677,
                       int32_t *out_out_arrsizze_53678,
                       struct memblock *out_mem_p_53679,
                       int32_t *out_out_arrsizze_53680,
                       int32_t *out_out_arrsizze_53681,
                       int32_t *out_out_arrsizze_53682,
                       struct memblock *out_mem_p_53683,
                       int32_t *out_out_arrsizze_53684,
                       int32_t *out_out_arrsizze_53685,
                       int32_t *out_out_arrsizze_53686,
                       struct memblock *out_mem_p_53687,
                       int32_t *out_out_arrsizze_53688,
                       int32_t *out_out_arrsizze_53689,
                       struct memblock *out_mem_p_53690,
                       int32_t *out_out_arrsizze_53691,
                       int32_t *out_out_arrsizze_53692,
                       struct memblock *out_mem_p_53693,
                       int32_t *out_out_arrsizze_53694,
                       int32_t *out_out_arrsizze_53695,
                       struct memblock *out_mem_p_53696,
                       int32_t *out_out_arrsizze_53697,
                       int32_t *out_out_arrsizze_53698,
                       struct memblock *out_mem_p_53699,
                       int32_t *out_out_arrsizze_53700,
                       struct memblock *out_mem_p_53701,
                       int32_t *out_out_arrsizze_53702,
                       int32_t *out_out_arrsizze_53703,
                       struct memblock *out_mem_p_53704,
                       int32_t *out_out_arrsizze_53705,
                       struct memblock *out_mem_p_53706,
                       int32_t *out_out_arrsizze_53707,
                       struct memblock *out_mem_p_53708,
                       int32_t *out_out_arrsizze_53709,
                       struct memblock *out_mem_p_53710,
                       int32_t *out_out_arrsizze_53711,
                       struct memblock mappingindices_mem_53310,
                       struct memblock images_mem_53311, int32_t N_52458,
                       int32_t m_52459, int32_t N_52460, int32_t trend_52461,
                       int32_t k_52462, int32_t n_52463, float freq_52464,
                       float hfrac_52465, float lam_52466);
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
                       struct memblock *out_mem_p_53672,
                       int32_t *out_out_arrsizze_53673,
                       struct memblock *out_mem_p_53674,
                       int32_t *out_out_arrsizze_53675,
                       struct memblock *out_mem_p_53676,
                       int32_t *out_out_arrsizze_53677,
                       int32_t *out_out_arrsizze_53678,
                       struct memblock *out_mem_p_53679,
                       int32_t *out_out_arrsizze_53680,
                       int32_t *out_out_arrsizze_53681,
                       int32_t *out_out_arrsizze_53682,
                       struct memblock *out_mem_p_53683,
                       int32_t *out_out_arrsizze_53684,
                       int32_t *out_out_arrsizze_53685,
                       int32_t *out_out_arrsizze_53686,
                       struct memblock *out_mem_p_53687,
                       int32_t *out_out_arrsizze_53688,
                       int32_t *out_out_arrsizze_53689,
                       struct memblock *out_mem_p_53690,
                       int32_t *out_out_arrsizze_53691,
                       int32_t *out_out_arrsizze_53692,
                       struct memblock *out_mem_p_53693,
                       int32_t *out_out_arrsizze_53694,
                       int32_t *out_out_arrsizze_53695,
                       struct memblock *out_mem_p_53696,
                       int32_t *out_out_arrsizze_53697,
                       int32_t *out_out_arrsizze_53698,
                       struct memblock *out_mem_p_53699,
                       int32_t *out_out_arrsizze_53700,
                       struct memblock *out_mem_p_53701,
                       int32_t *out_out_arrsizze_53702,
                       int32_t *out_out_arrsizze_53703,
                       struct memblock *out_mem_p_53704,
                       int32_t *out_out_arrsizze_53705,
                       struct memblock *out_mem_p_53706,
                       int32_t *out_out_arrsizze_53707,
                       struct memblock *out_mem_p_53708,
                       int32_t *out_out_arrsizze_53709,
                       struct memblock *out_mem_p_53710,
                       int32_t *out_out_arrsizze_53711,
                       struct memblock mappingindices_mem_53310,
                       struct memblock images_mem_53311, int32_t N_52458,
                       int32_t m_52459, int32_t N_52460, int32_t trend_52461,
                       int32_t k_52462, int32_t n_52463, float freq_52464,
                       float hfrac_52465, float lam_52466)
{
    struct memblock out_mem_53578;
    
    out_mem_53578.references = NULL;
    
    int32_t out_arrsizze_53579;
    struct memblock out_mem_53580;
    
    out_mem_53580.references = NULL;
    
    int32_t out_arrsizze_53581;
    struct memblock out_mem_53582;
    
    out_mem_53582.references = NULL;
    
    int32_t out_arrsizze_53583;
    int32_t out_arrsizze_53584;
    struct memblock out_mem_53585;
    
    out_mem_53585.references = NULL;
    
    int32_t out_arrsizze_53586;
    int32_t out_arrsizze_53587;
    int32_t out_arrsizze_53588;
    struct memblock out_mem_53589;
    
    out_mem_53589.references = NULL;
    
    int32_t out_arrsizze_53590;
    int32_t out_arrsizze_53591;
    int32_t out_arrsizze_53592;
    struct memblock out_mem_53593;
    
    out_mem_53593.references = NULL;
    
    int32_t out_arrsizze_53594;
    int32_t out_arrsizze_53595;
    struct memblock out_mem_53596;
    
    out_mem_53596.references = NULL;
    
    int32_t out_arrsizze_53597;
    int32_t out_arrsizze_53598;
    struct memblock out_mem_53599;
    
    out_mem_53599.references = NULL;
    
    int32_t out_arrsizze_53600;
    int32_t out_arrsizze_53601;
    struct memblock out_mem_53602;
    
    out_mem_53602.references = NULL;
    
    int32_t out_arrsizze_53603;
    int32_t out_arrsizze_53604;
    struct memblock out_mem_53605;
    
    out_mem_53605.references = NULL;
    
    int32_t out_arrsizze_53606;
    struct memblock out_mem_53607;
    
    out_mem_53607.references = NULL;
    
    int32_t out_arrsizze_53608;
    int32_t out_arrsizze_53609;
    struct memblock out_mem_53610;
    
    out_mem_53610.references = NULL;
    
    int32_t out_arrsizze_53611;
    struct memblock out_mem_53612;
    
    out_mem_53612.references = NULL;
    
    int32_t out_arrsizze_53613;
    struct memblock out_mem_53614;
    
    out_mem_53614.references = NULL;
    
    int32_t out_arrsizze_53615;
    struct memblock out_mem_53616;
    
    out_mem_53616.references = NULL;
    
    int32_t out_arrsizze_53617;
    bool dim_zzero_52469 = 0 == m_52459;
    bool dim_zzero_52470 = 0 == N_52460;
    bool old_empty_52471 = dim_zzero_52469 || dim_zzero_52470;
    bool dim_zzero_52472 = 0 == N_52458;
    bool new_empty_52473 = dim_zzero_52469 || dim_zzero_52472;
    bool both_empty_52474 = old_empty_52471 && new_empty_52473;
    bool dim_match_52475 = N_52458 == N_52460;
    bool empty_or_match_52476 = both_empty_52474 || dim_match_52475;
    bool empty_or_match_cert_52477;
    
    if (!empty_or_match_52476) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "bfast-irreg.fut:133:1-314:110",
                               "function arguments of wrong shape");
        if (memblock_unref(ctx, &out_mem_53616, "out_mem_53616") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53614, "out_mem_53614") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53612, "out_mem_53612") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53610, "out_mem_53610") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53607, "out_mem_53607") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53605, "out_mem_53605") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53602, "out_mem_53602") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53599, "out_mem_53599") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53596, "out_mem_53596") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53593, "out_mem_53593") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53589, "out_mem_53589") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53585, "out_mem_53585") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53582, "out_mem_53582") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53580, "out_mem_53580") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53578, "out_mem_53578") != 0)
            return 1;
        return 1;
    }
    
    int32_t x_52479 = 2 * k_52462;
    int32_t k2p2_52480 = 2 + x_52479;
    bool cond_52481 = slt32(0, trend_52461);
    int32_t k2p2zq_52482;
    
    if (cond_52481) {
        k2p2zq_52482 = k2p2_52480;
    } else {
        int32_t res_52483 = k2p2_52480 - 1;
        
        k2p2zq_52482 = res_52483;
    }
    
    int64_t binop_x_53313 = sext_i32_i64(k2p2zq_52482);
    int64_t binop_y_53314 = sext_i32_i64(N_52458);
    int64_t binop_x_53315 = binop_x_53313 * binop_y_53314;
    int64_t bytes_53312 = 4 * binop_x_53315;
    int64_t binop_x_53328 = sext_i32_i64(k2p2zq_52482);
    int64_t binop_y_53329 = sext_i32_i64(N_52458);
    int64_t binop_x_53330 = binop_x_53328 * binop_y_53329;
    int64_t bytes_53327 = 4 * binop_x_53330;
    struct memblock lifted_1_zlzb_arg_mem_53342;
    
    lifted_1_zlzb_arg_mem_53342.references = NULL;
    if (cond_52481) {
        bool bounds_invalid_upwards_52485 = slt32(k2p2zq_52482, 0);
        bool eq_x_zz_52486 = 0 == k2p2zq_52482;
        bool not_p_52487 = !bounds_invalid_upwards_52485;
        bool p_and_eq_x_y_52488 = eq_x_zz_52486 && not_p_52487;
        bool dim_zzero_52489 = bounds_invalid_upwards_52485 ||
             p_and_eq_x_y_52488;
        bool both_empty_52490 = eq_x_zz_52486 && dim_zzero_52489;
        bool empty_or_match_52494 = not_p_52487 || both_empty_52490;
        bool empty_or_match_cert_52495;
        
        if (!empty_or_match_52494) {
            ctx->error = msgprintf("Error at %s:\n%s%s%s%d%s%s\n",
                                   "bfast-irreg.fut:133:1-314:110 -> bfast-irreg.fut:144:16-55 -> bfast-irreg.fut:64:10-18 -> /futlib/array.fut:61:1-62:12",
                                   "Function return value does not match shape of type ",
                                   "*", "[", k2p2zq_52482, "]",
                                   "intrinsics.i32");
            if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_53342,
                               "lifted_1_zlzb_arg_mem_53342") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53616, "out_mem_53616") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53614, "out_mem_53614") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53612, "out_mem_53612") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53610, "out_mem_53610") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53607, "out_mem_53607") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53605, "out_mem_53605") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53602, "out_mem_53602") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53599, "out_mem_53599") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53596, "out_mem_53596") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53593, "out_mem_53593") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53589, "out_mem_53589") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53585, "out_mem_53585") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53582, "out_mem_53582") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53580, "out_mem_53580") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53578, "out_mem_53578") != 0)
                return 1;
            return 1;
        }
        
        struct memblock mem_53316;
        
        mem_53316.references = NULL;
        if (memblock_alloc(ctx, &mem_53316, bytes_53312, "mem_53316"))
            return 1;
        for (int32_t i_52997 = 0; i_52997 < k2p2zq_52482; i_52997++) {
            bool cond_52499 = i_52997 == 0;
            bool cond_52500 = i_52997 == 1;
            int32_t r32_arg_52501 = sdiv32(i_52997, 2);
            int32_t x_52502 = smod32(i_52997, 2);
            float res_52503 = sitofp_i32_f32(r32_arg_52501);
            bool cond_52504 = x_52502 == 0;
            float x_52505 = 6.2831855F * res_52503;
            
            for (int32_t i_52993 = 0; i_52993 < N_52458; i_52993++) {
                int32_t x_52507 =
                        ((int32_t *) mappingindices_mem_53310.mem)[i_52993];
                float res_52508;
                
                if (cond_52499) {
                    res_52508 = 1.0F;
                } else {
                    float res_52509;
                    
                    if (cond_52500) {
                        float res_52510 = sitofp_i32_f32(x_52507);
                        
                        res_52509 = res_52510;
                    } else {
                        float res_52511 = sitofp_i32_f32(x_52507);
                        float x_52512 = x_52505 * res_52511;
                        float angle_52513 = x_52512 / freq_52464;
                        float res_52514;
                        
                        if (cond_52504) {
                            float res_52515;
                            
                            res_52515 = futrts_sin32(angle_52513);
                            res_52514 = res_52515;
                        } else {
                            float res_52516;
                            
                            res_52516 = futrts_cos32(angle_52513);
                            res_52514 = res_52516;
                        }
                        res_52509 = res_52514;
                    }
                    res_52508 = res_52509;
                }
                ((float *) mem_53316.mem)[i_52997 * N_52458 + i_52993] =
                    res_52508;
            }
        }
        if (memblock_set(ctx, &lifted_1_zlzb_arg_mem_53342, &mem_53316,
                         "mem_53316") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53316, "mem_53316") != 0)
            return 1;
    } else {
        bool bounds_invalid_upwards_52517 = slt32(k2p2zq_52482, 0);
        bool eq_x_zz_52518 = 0 == k2p2zq_52482;
        bool not_p_52519 = !bounds_invalid_upwards_52517;
        bool p_and_eq_x_y_52520 = eq_x_zz_52518 && not_p_52519;
        bool dim_zzero_52521 = bounds_invalid_upwards_52517 ||
             p_and_eq_x_y_52520;
        bool both_empty_52522 = eq_x_zz_52518 && dim_zzero_52521;
        bool empty_or_match_52526 = not_p_52519 || both_empty_52522;
        bool empty_or_match_cert_52527;
        
        if (!empty_or_match_52526) {
            ctx->error = msgprintf("Error at %s:\n%s%s%s%d%s%s\n",
                                   "bfast-irreg.fut:133:1-314:110 -> bfast-irreg.fut:145:16-55 -> bfast-irreg.fut:76:10-20 -> /futlib/array.fut:61:1-62:12",
                                   "Function return value does not match shape of type ",
                                   "*", "[", k2p2zq_52482, "]",
                                   "intrinsics.i32");
            if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_53342,
                               "lifted_1_zlzb_arg_mem_53342") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53616, "out_mem_53616") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53614, "out_mem_53614") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53612, "out_mem_53612") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53610, "out_mem_53610") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53607, "out_mem_53607") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53605, "out_mem_53605") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53602, "out_mem_53602") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53599, "out_mem_53599") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53596, "out_mem_53596") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53593, "out_mem_53593") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53589, "out_mem_53589") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53585, "out_mem_53585") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53582, "out_mem_53582") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53580, "out_mem_53580") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53578, "out_mem_53578") != 0)
                return 1;
            return 1;
        }
        
        struct memblock mem_53331;
        
        mem_53331.references = NULL;
        if (memblock_alloc(ctx, &mem_53331, bytes_53327, "mem_53331"))
            return 1;
        for (int32_t i_53005 = 0; i_53005 < k2p2zq_52482; i_53005++) {
            bool cond_52531 = i_53005 == 0;
            int32_t i_52532 = 1 + i_53005;
            int32_t r32_arg_52533 = sdiv32(i_52532, 2);
            int32_t x_52534 = smod32(i_52532, 2);
            float res_52535 = sitofp_i32_f32(r32_arg_52533);
            bool cond_52536 = x_52534 == 0;
            float x_52537 = 6.2831855F * res_52535;
            
            for (int32_t i_53001 = 0; i_53001 < N_52458; i_53001++) {
                int32_t x_52539 =
                        ((int32_t *) mappingindices_mem_53310.mem)[i_53001];
                float res_52540;
                
                if (cond_52531) {
                    res_52540 = 1.0F;
                } else {
                    float res_52541 = sitofp_i32_f32(x_52539);
                    float x_52542 = x_52537 * res_52541;
                    float angle_52543 = x_52542 / freq_52464;
                    float res_52544;
                    
                    if (cond_52536) {
                        float res_52545;
                        
                        res_52545 = futrts_sin32(angle_52543);
                        res_52544 = res_52545;
                    } else {
                        float res_52546;
                        
                        res_52546 = futrts_cos32(angle_52543);
                        res_52544 = res_52546;
                    }
                    res_52540 = res_52544;
                }
                ((float *) mem_53331.mem)[i_53005 * N_52458 + i_53001] =
                    res_52540;
            }
        }
        if (memblock_set(ctx, &lifted_1_zlzb_arg_mem_53342, &mem_53331,
                         "mem_53331") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53331, "mem_53331") != 0)
            return 1;
    }
    
    int32_t x_52548 = N_52458 * N_52458;
    int32_t y_52549 = 2 * N_52458;
    int32_t x_52550 = x_52548 + y_52549;
    int32_t x_52551 = 1 + x_52550;
    int32_t y_52552 = 1 + N_52458;
    int32_t x_52553 = sdiv32(x_52551, y_52552);
    int32_t x_52554 = x_52553 - N_52458;
    int32_t lifted_1_zlzb_arg_52555 = x_52554 - 1;
    float res_52556 = sitofp_i32_f32(lifted_1_zlzb_arg_52555);
    int64_t binop_x_53344 = sext_i32_i64(N_52458);
    int64_t binop_y_53345 = sext_i32_i64(k2p2zq_52482);
    int64_t binop_x_53346 = binop_x_53344 * binop_y_53345;
    int64_t bytes_53343 = 4 * binop_x_53346;
    struct memblock mem_53347;
    
    mem_53347.references = NULL;
    if (memblock_alloc(ctx, &mem_53347, bytes_53343, "mem_53347"))
        return 1;
    for (int32_t i_53013 = 0; i_53013 < N_52458; i_53013++) {
        for (int32_t i_53009 = 0; i_53009 < k2p2zq_52482; i_53009++) {
            float x_52561 =
                  ((float *) lifted_1_zlzb_arg_mem_53342.mem)[i_53009 *
                                                              N_52458 +
                                                              i_53013];
            float res_52562 = res_52556 + x_52561;
            
            ((float *) mem_53347.mem)[i_53013 * k2p2zq_52482 + i_53009] =
                res_52562;
        }
    }
    
    int32_t m_52565 = k2p2zq_52482 - 1;
    bool empty_slice_52572 = n_52463 == 0;
    int32_t m_52573 = n_52463 - 1;
    bool zzero_leq_i_p_m_t_s_52574 = sle32(0, m_52573);
    bool i_p_m_t_s_leq_w_52575 = slt32(m_52573, N_52458);
    bool i_lte_j_52576 = sle32(0, n_52463);
    bool y_52577 = zzero_leq_i_p_m_t_s_52574 && i_p_m_t_s_leq_w_52575;
    bool y_52578 = i_lte_j_52576 && y_52577;
    bool ok_or_empty_52579 = empty_slice_52572 || y_52578;
    bool index_certs_52581;
    
    if (!ok_or_empty_52579) {
        ctx->error = msgprintf("Error at %s:\n%s%d%s%s%s%d%s%d%s%d%s\n",
                               "bfast-irreg.fut:133:1-314:110 -> bfast-irreg.fut:154:15-21",
                               "Index [", 0, ", ", "", ":", n_52463,
                               "] out of bounds for array of shape [",
                               k2p2zq_52482, "][", N_52458, "].");
        if (memblock_unref(ctx, &mem_53347, "mem_53347") != 0)
            return 1;
        if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_53342,
                           "lifted_1_zlzb_arg_mem_53342") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53616, "out_mem_53616") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53614, "out_mem_53614") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53612, "out_mem_53612") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53610, "out_mem_53610") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53607, "out_mem_53607") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53605, "out_mem_53605") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53602, "out_mem_53602") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53599, "out_mem_53599") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53596, "out_mem_53596") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53593, "out_mem_53593") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53589, "out_mem_53589") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53585, "out_mem_53585") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53582, "out_mem_53582") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53580, "out_mem_53580") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53578, "out_mem_53578") != 0)
            return 1;
        return 1;
    }
    
    bool index_certs_52583;
    
    if (!ok_or_empty_52579) {
        ctx->error = msgprintf("Error at %s:\n%s%s%s%d%s%d%s%d%s%d%s\n",
                               "bfast-irreg.fut:133:1-314:110 -> bfast-irreg.fut:155:15-22",
                               "Index [", "", ":", n_52463, ", ", 0,
                               "] out of bounds for array of shape [", N_52458,
                               "][", k2p2zq_52482, "].");
        if (memblock_unref(ctx, &mem_53347, "mem_53347") != 0)
            return 1;
        if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_53342,
                           "lifted_1_zlzb_arg_mem_53342") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53616, "out_mem_53616") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53614, "out_mem_53614") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53612, "out_mem_53612") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53610, "out_mem_53610") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53607, "out_mem_53607") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53605, "out_mem_53605") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53602, "out_mem_53602") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53599, "out_mem_53599") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53596, "out_mem_53596") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53593, "out_mem_53593") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53589, "out_mem_53589") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53585, "out_mem_53585") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53582, "out_mem_53582") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53580, "out_mem_53580") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53578, "out_mem_53578") != 0)
            return 1;
        return 1;
    }
    
    bool index_certs_52594;
    
    if (!ok_or_empty_52579) {
        ctx->error = msgprintf("Error at %s:\n%s%d%s%s%s%d%s%d%s%d%s\n",
                               "bfast-irreg.fut:133:1-314:110 -> bfast-irreg.fut:156:15-26",
                               "Index [", 0, ", ", "", ":", n_52463,
                               "] out of bounds for array of shape [", m_52459,
                               "][", N_52458, "].");
        if (memblock_unref(ctx, &mem_53347, "mem_53347") != 0)
            return 1;
        if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_53342,
                           "lifted_1_zlzb_arg_mem_53342") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53616, "out_mem_53616") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53614, "out_mem_53614") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53612, "out_mem_53612") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53610, "out_mem_53610") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53607, "out_mem_53607") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53605, "out_mem_53605") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53602, "out_mem_53602") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53599, "out_mem_53599") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53596, "out_mem_53596") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53593, "out_mem_53593") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53589, "out_mem_53589") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53585, "out_mem_53585") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53582, "out_mem_53582") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53580, "out_mem_53580") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53578, "out_mem_53578") != 0)
            return 1;
        return 1;
    }
    
    int64_t binop_x_53359 = sext_i32_i64(m_52459);
    int64_t binop_x_53361 = binop_y_53345 * binop_x_53359;
    int64_t binop_x_53363 = binop_y_53345 * binop_x_53361;
    int64_t bytes_53358 = 4 * binop_x_53363;
    struct memblock mem_53364;
    
    mem_53364.references = NULL;
    if (memblock_alloc(ctx, &mem_53364, bytes_53358, "mem_53364"))
        return 1;
    for (int32_t i_53027 = 0; i_53027 < m_52459; i_53027++) {
        for (int32_t i_53023 = 0; i_53023 < k2p2zq_52482; i_53023++) {
            for (int32_t i_53019 = 0; i_53019 < k2p2zq_52482; i_53019++) {
                float res_52603;
                float redout_53015 = 0.0F;
                
                for (int32_t i_53016 = 0; i_53016 < n_52463; i_53016++) {
                    float x_52607 = ((float *) images_mem_53311.mem)[i_53027 *
                                                                     N_52460 +
                                                                     i_53016];
                    float x_52608 =
                          ((float *) lifted_1_zlzb_arg_mem_53342.mem)[i_53023 *
                                                                      N_52458 +
                                                                      i_53016];
                    float x_52609 = ((float *) mem_53347.mem)[i_53016 *
                                                              k2p2zq_52482 +
                                                              i_53019];
                    float x_52610 = x_52608 * x_52609;
                    bool res_52611;
                    
                    res_52611 = futrts_isnan32(x_52607);
                    
                    float y_52612;
                    
                    if (res_52611) {
                        y_52612 = 0.0F;
                    } else {
                        y_52612 = 1.0F;
                    }
                    
                    float res_52613 = x_52610 * y_52612;
                    float res_52606 = res_52613 + redout_53015;
                    float redout_tmp_53627 = res_52606;
                    
                    redout_53015 = redout_tmp_53627;
                }
                res_52603 = redout_53015;
                ((float *) mem_53364.mem)[i_53027 * (k2p2zq_52482 *
                                                     k2p2zq_52482) + i_53023 *
                                          k2p2zq_52482 + i_53019] = res_52603;
            }
        }
    }
    
    int32_t j_52615 = 2 * k2p2zq_52482;
    int32_t j_m_i_52616 = j_52615 - k2p2zq_52482;
    int32_t nm_52619 = k2p2zq_52482 * j_52615;
    bool empty_slice_52632 = j_m_i_52616 == 0;
    int32_t m_52633 = j_m_i_52616 - 1;
    int32_t i_p_m_t_s_52634 = k2p2zq_52482 + m_52633;
    bool zzero_leq_i_p_m_t_s_52635 = sle32(0, i_p_m_t_s_52634);
    bool ok_or_empty_52642 = empty_slice_52632 || zzero_leq_i_p_m_t_s_52635;
    bool index_certs_52644;
    
    if (!ok_or_empty_52642) {
        ctx->error = msgprintf("Error at %s:\n%s%d%s%d%s%d%s%d%s%d%s%d%s\n",
                               "bfast-irreg.fut:133:1-314:110 -> bfast-irreg.fut:168:14-29 -> bfast-irreg.fut:109:8-37",
                               "Index [", 0, ":", k2p2zq_52482, ", ",
                               k2p2zq_52482, ":", j_52615,
                               "] out of bounds for array of shape [",
                               k2p2zq_52482, "][", j_52615, "].");
        if (memblock_unref(ctx, &mem_53364, "mem_53364") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53347, "mem_53347") != 0)
            return 1;
        if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_53342,
                           "lifted_1_zlzb_arg_mem_53342") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53616, "out_mem_53616") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53614, "out_mem_53614") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53612, "out_mem_53612") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53610, "out_mem_53610") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53607, "out_mem_53607") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53605, "out_mem_53605") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53602, "out_mem_53602") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53599, "out_mem_53599") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53596, "out_mem_53596") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53593, "out_mem_53593") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53589, "out_mem_53589") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53585, "out_mem_53585") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53582, "out_mem_53582") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53580, "out_mem_53580") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53578, "out_mem_53578") != 0)
            return 1;
        return 1;
    }
    
    int64_t binop_y_53396 = sext_i32_i64(j_m_i_52616);
    int64_t binop_x_53397 = binop_x_53361 * binop_y_53396;
    int64_t bytes_53392 = 4 * binop_x_53397;
    struct memblock mem_53398;
    
    mem_53398.references = NULL;
    if (memblock_alloc(ctx, &mem_53398, bytes_53392, "mem_53398"))
        return 1;
    
    int64_t binop_x_53401 = sext_i32_i64(nm_52619);
    int64_t bytes_53400 = 4 * binop_x_53401;
    struct memblock mem_53402;
    
    mem_53402.references = NULL;
    if (memblock_alloc(ctx, &mem_53402, bytes_53400, "mem_53402"))
        return 1;
    
    struct memblock mem_53408;
    
    mem_53408.references = NULL;
    if (memblock_alloc(ctx, &mem_53408, bytes_53400, "mem_53408"))
        return 1;
    for (int32_t i_53049 = 0; i_53049 < m_52459; i_53049++) {
        for (int32_t i_53031 = 0; i_53031 < nm_52619; i_53031++) {
            int32_t res_52649 = sdiv32(i_53031, j_52615);
            int32_t res_52650 = smod32(i_53031, j_52615);
            bool cond_52651 = slt32(res_52650, k2p2zq_52482);
            float res_52652;
            
            if (cond_52651) {
                float res_52653 = ((float *) mem_53364.mem)[i_53049 *
                                                            (k2p2zq_52482 *
                                                             k2p2zq_52482) +
                                                            res_52649 *
                                                            k2p2zq_52482 +
                                                            res_52650];
                
                res_52652 = res_52653;
            } else {
                int32_t y_52654 = k2p2zq_52482 + res_52649;
                bool cond_52655 = res_52650 == y_52654;
                float res_52656;
                
                if (cond_52655) {
                    res_52656 = 1.0F;
                } else {
                    res_52656 = 0.0F;
                }
                res_52652 = res_52656;
            }
            ((float *) mem_53402.mem)[i_53031] = res_52652;
        }
        for (int32_t i_52659 = 0; i_52659 < k2p2zq_52482; i_52659++) {
            float v1_52664 = ((float *) mem_53402.mem)[i_52659];
            bool cond_52665 = v1_52664 == 0.0F;
            
            for (int32_t i_53035 = 0; i_53035 < nm_52619; i_53035++) {
                int32_t res_52668 = sdiv32(i_53035, j_52615);
                int32_t res_52669 = smod32(i_53035, j_52615);
                float res_52670;
                
                if (cond_52665) {
                    int32_t x_52671 = j_52615 * res_52668;
                    int32_t i_52672 = res_52669 + x_52671;
                    float res_52673 = ((float *) mem_53402.mem)[i_52672];
                    
                    res_52670 = res_52673;
                } else {
                    float x_52674 = ((float *) mem_53402.mem)[res_52669];
                    float x_52675 = x_52674 / v1_52664;
                    bool cond_52676 = slt32(res_52668, m_52565);
                    float res_52677;
                    
                    if (cond_52676) {
                        int32_t x_52678 = 1 + res_52668;
                        int32_t x_52679 = j_52615 * x_52678;
                        int32_t i_52680 = res_52669 + x_52679;
                        float x_52681 = ((float *) mem_53402.mem)[i_52680];
                        int32_t i_52682 = i_52659 + x_52679;
                        float x_52683 = ((float *) mem_53402.mem)[i_52682];
                        float y_52684 = x_52675 * x_52683;
                        float res_52685 = x_52681 - y_52684;
                        
                        res_52677 = res_52685;
                    } else {
                        res_52677 = x_52675;
                    }
                    res_52670 = res_52677;
                }
                ((float *) mem_53408.mem)[i_53035] = res_52670;
            }
            for (int32_t write_iter_53037 = 0; write_iter_53037 < nm_52619;
                 write_iter_53037++) {
                bool less_than_zzero_53041 = slt32(write_iter_53037, 0);
                bool greater_than_sizze_53042 = sle32(nm_52619,
                                                      write_iter_53037);
                bool outside_bounds_dim_53043 = less_than_zzero_53041 ||
                     greater_than_sizze_53042;
                
                if (!outside_bounds_dim_53043) {
                    memmove(mem_53402.mem + write_iter_53037 * 4,
                            mem_53408.mem + write_iter_53037 * 4,
                            sizeof(float));
                }
            }
        }
        for (int32_t i_53633 = 0; i_53633 < k2p2zq_52482; i_53633++) {
            for (int32_t i_53634 = 0; i_53634 < j_m_i_52616; i_53634++) {
                ((float *) mem_53398.mem)[i_53049 * (j_m_i_52616 *
                                                     k2p2zq_52482) + (i_53633 *
                                                                      j_m_i_52616 +
                                                                      i_53634)] =
                    ((float *) mem_53402.mem)[k2p2zq_52482 + (i_53633 *
                                                              j_52615 +
                                                              i_53634)];
            }
        }
    }
    if (memblock_unref(ctx, &mem_53402, "mem_53402") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53408, "mem_53408") != 0)
        return 1;
    
    int64_t bytes_53415 = 4 * binop_x_53361;
    struct memblock mem_53419;
    
    mem_53419.references = NULL;
    if (memblock_alloc(ctx, &mem_53419, bytes_53415, "mem_53419"))
        return 1;
    for (int32_t i_53059 = 0; i_53059 < m_52459; i_53059++) {
        for (int32_t i_53055 = 0; i_53055 < k2p2zq_52482; i_53055++) {
            float res_52696;
            float redout_53051 = 0.0F;
            
            for (int32_t i_53052 = 0; i_53052 < n_52463; i_53052++) {
                float x_52700 =
                      ((float *) lifted_1_zlzb_arg_mem_53342.mem)[i_53055 *
                                                                  N_52458 +
                                                                  i_53052];
                float x_52701 = ((float *) images_mem_53311.mem)[i_53059 *
                                                                 N_52460 +
                                                                 i_53052];
                bool res_52702;
                
                res_52702 = futrts_isnan32(x_52701);
                
                float res_52703;
                
                if (res_52702) {
                    res_52703 = 0.0F;
                } else {
                    float res_52704 = x_52700 * x_52701;
                    
                    res_52703 = res_52704;
                }
                
                float res_52699 = res_52703 + redout_53051;
                float redout_tmp_53637 = res_52699;
                
                redout_53051 = redout_tmp_53637;
            }
            res_52696 = redout_53051;
            ((float *) mem_53419.mem)[i_53059 * k2p2zq_52482 + i_53055] =
                res_52696;
        }
    }
    if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_53342,
                       "lifted_1_zlzb_arg_mem_53342") != 0)
        return 1;
    
    struct memblock mem_53434;
    
    mem_53434.references = NULL;
    if (memblock_alloc(ctx, &mem_53434, bytes_53415, "mem_53434"))
        return 1;
    for (int32_t i_53069 = 0; i_53069 < m_52459; i_53069++) {
        for (int32_t i_53065 = 0; i_53065 < k2p2zq_52482; i_53065++) {
            float res_52716;
            float redout_53061 = 0.0F;
            
            for (int32_t i_53062 = 0; i_53062 < j_m_i_52616; i_53062++) {
                float x_52720 = ((float *) mem_53419.mem)[i_53069 *
                                                          k2p2zq_52482 +
                                                          i_53062];
                float x_52721 = ((float *) mem_53398.mem)[i_53069 *
                                                          (j_m_i_52616 *
                                                           k2p2zq_52482) +
                                                          i_53065 *
                                                          j_m_i_52616 +
                                                          i_53062];
                float res_52722 = x_52720 * x_52721;
                float res_52719 = res_52722 + redout_53061;
                float redout_tmp_53640 = res_52719;
                
                redout_53061 = redout_tmp_53640;
            }
            res_52716 = redout_53061;
            ((float *) mem_53434.mem)[i_53069 * k2p2zq_52482 + i_53065] =
                res_52716;
        }
    }
    
    int64_t binop_x_53448 = binop_x_53344 * binop_x_53359;
    int64_t bytes_53445 = 4 * binop_x_53448;
    struct memblock mem_53449;
    
    mem_53449.references = NULL;
    if (memblock_alloc(ctx, &mem_53449, bytes_53445, "mem_53449"))
        return 1;
    for (int32_t i_53079 = 0; i_53079 < m_52459; i_53079++) {
        for (int32_t i_53075 = 0; i_53075 < N_52458; i_53075++) {
            float res_52728;
            float redout_53071 = 0.0F;
            
            for (int32_t i_53072 = 0; i_53072 < k2p2zq_52482; i_53072++) {
                float x_52732 = ((float *) mem_53434.mem)[i_53079 *
                                                          k2p2zq_52482 +
                                                          i_53072];
                float x_52733 = ((float *) mem_53347.mem)[i_53075 *
                                                          k2p2zq_52482 +
                                                          i_53072];
                float res_52734 = x_52732 * x_52733;
                float res_52731 = res_52734 + redout_53071;
                float redout_tmp_53643 = res_52731;
                
                redout_53071 = redout_tmp_53643;
            }
            res_52728 = redout_53071;
            ((float *) mem_53449.mem)[i_53079 * N_52458 + i_53075] = res_52728;
        }
    }
    
    int32_t i_52736 = N_52458 - 1;
    bool x_52737 = sle32(0, i_52736);
    bool index_certs_52740;
    
    if (!x_52737) {
        ctx->error = msgprintf("Error at %s:\n%s%d%s%d%s\n",
                               "bfast-irreg.fut:133:1-314:110 -> bfast-irreg.fut:189:5-198:25 -> /futlib/soacs.fut:51:3-37 -> /futlib/soacs.fut:51:19-23 -> bfast-irreg.fut:194:30-91 -> bfast-irreg.fut:37:13-20 -> /futlib/array.fut:18:29-34",
                               "Index [", i_52736,
                               "] out of bounds for array of shape [", N_52458,
                               "].");
        if (memblock_unref(ctx, &mem_53449, "mem_53449") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53434, "mem_53434") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53419, "mem_53419") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53408, "mem_53408") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53402, "mem_53402") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53398, "mem_53398") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53364, "mem_53364") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53347, "mem_53347") != 0)
            return 1;
        if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_53342,
                           "lifted_1_zlzb_arg_mem_53342") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53616, "out_mem_53616") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53614, "out_mem_53614") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53612, "out_mem_53612") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53610, "out_mem_53610") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53607, "out_mem_53607") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53605, "out_mem_53605") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53602, "out_mem_53602") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53599, "out_mem_53599") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53596, "out_mem_53596") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53593, "out_mem_53593") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53589, "out_mem_53589") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53585, "out_mem_53585") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53582, "out_mem_53582") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53580, "out_mem_53580") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53578, "out_mem_53578") != 0)
            return 1;
        return 1;
    }
    
    int64_t bytes_53460 = 4 * binop_x_53359;
    struct memblock mem_53462;
    
    mem_53462.references = NULL;
    if (memblock_alloc(ctx, &mem_53462, bytes_53460, "mem_53462"))
        return 1;
    
    struct memblock mem_53467;
    
    mem_53467.references = NULL;
    if (memblock_alloc(ctx, &mem_53467, bytes_53445, "mem_53467"))
        return 1;
    
    struct memblock mem_53472;
    
    mem_53472.references = NULL;
    if (memblock_alloc(ctx, &mem_53472, bytes_53445, "mem_53472"))
        return 1;
    
    int64_t bytes_53476 = 4 * binop_x_53344;
    struct memblock mem_53478;
    
    mem_53478.references = NULL;
    if (memblock_alloc(ctx, &mem_53478, bytes_53476, "mem_53478"))
        return 1;
    
    struct memblock mem_53481;
    
    mem_53481.references = NULL;
    if (memblock_alloc(ctx, &mem_53481, bytes_53476, "mem_53481"))
        return 1;
    
    struct memblock mem_53488;
    
    mem_53488.references = NULL;
    if (memblock_alloc(ctx, &mem_53488, bytes_53476, "mem_53488"))
        return 1;
    for (int32_t i_53644 = 0; i_53644 < N_52458; i_53644++) {
        ((float *) mem_53488.mem)[i_53644] = NAN;
    }
    
    struct memblock mem_53491;
    
    mem_53491.references = NULL;
    if (memblock_alloc(ctx, &mem_53491, bytes_53476, "mem_53491"))
        return 1;
    for (int32_t i_53645 = 0; i_53645 < N_52458; i_53645++) {
        ((int32_t *) mem_53491.mem)[i_53645] = 0;
    }
    
    struct memblock mem_53496;
    
    mem_53496.references = NULL;
    if (memblock_alloc(ctx, &mem_53496, bytes_53476, "mem_53496"))
        return 1;
    
    struct memblock mem_53506;
    
    mem_53506.references = NULL;
    if (memblock_alloc(ctx, &mem_53506, bytes_53476, "mem_53506"))
        return 1;
    for (int32_t i_53114 = 0; i_53114 < m_52459; i_53114++) {
        int32_t discard_53089;
        int32_t scanacc_53083 = 0;
        
        for (int32_t i_53086 = 0; i_53086 < N_52458; i_53086++) {
            float x_52770 = ((float *) images_mem_53311.mem)[i_53114 * N_52460 +
                                                             i_53086];
            float x_52771 = ((float *) mem_53449.mem)[i_53114 * N_52458 +
                                                      i_53086];
            bool res_52772;
            
            res_52772 = futrts_isnan32(x_52770);
            
            bool cond_52773 = !res_52772;
            float res_52774;
            
            if (cond_52773) {
                float res_52775 = x_52770 - x_52771;
                
                res_52774 = res_52775;
            } else {
                res_52774 = NAN;
            }
            
            bool res_52776;
            
            res_52776 = futrts_isnan32(res_52774);
            
            bool res_52777 = !res_52776;
            int32_t res_52778;
            
            if (res_52777) {
                res_52778 = 1;
            } else {
                res_52778 = 0;
            }
            
            int32_t res_52769 = res_52778 + scanacc_53083;
            
            ((int32_t *) mem_53478.mem)[i_53086] = res_52769;
            ((float *) mem_53481.mem)[i_53086] = res_52774;
            
            int32_t scanacc_tmp_53649 = res_52769;
            
            scanacc_53083 = scanacc_tmp_53649;
        }
        discard_53089 = scanacc_53083;
        memmove(mem_53467.mem + i_53114 * N_52458 * 4, mem_53488.mem + 0,
                N_52458 * sizeof(float));
        memmove(mem_53472.mem + i_53114 * N_52458 * 4, mem_53491.mem + 0,
                N_52458 * sizeof(int32_t));
        for (int32_t write_iter_53090 = 0; write_iter_53090 < N_52458;
             write_iter_53090++) {
            float write_iv_53093 = ((float *) mem_53481.mem)[write_iter_53090];
            int32_t write_iv_53094 =
                    ((int32_t *) mem_53478.mem)[write_iter_53090];
            bool res_52789;
            
            res_52789 = futrts_isnan32(write_iv_53093);
            
            bool res_52790 = !res_52789;
            int32_t res_52791;
            
            if (res_52790) {
                int32_t res_52792 = write_iv_53094 - 1;
                
                res_52791 = res_52792;
            } else {
                res_52791 = -1;
            }
            
            bool less_than_zzero_53096 = slt32(res_52791, 0);
            bool greater_than_sizze_53097 = sle32(N_52458, res_52791);
            bool outside_bounds_dim_53098 = less_than_zzero_53096 ||
                 greater_than_sizze_53097;
            
            memmove(mem_53496.mem + 0, mem_53472.mem + i_53114 * N_52458 * 4,
                    N_52458 * sizeof(int32_t));
            
            struct memblock write_out_mem_53503;
            
            write_out_mem_53503.references = NULL;
            if (outside_bounds_dim_53098) {
                if (memblock_set(ctx, &write_out_mem_53503, &mem_53496,
                                 "mem_53496") != 0)
                    return 1;
            } else {
                struct memblock mem_53499;
                
                mem_53499.references = NULL;
                if (memblock_alloc(ctx, &mem_53499, 4, "mem_53499"))
                    return 1;
                
                int32_t x_53655;
                
                for (int32_t i_53654 = 0; i_53654 < 1; i_53654++) {
                    x_53655 = write_iter_53090 + sext_i32_i32(i_53654);
                    ((int32_t *) mem_53499.mem)[i_53654] = x_53655;
                }
                
                struct memblock mem_53502;
                
                mem_53502.references = NULL;
                if (memblock_alloc(ctx, &mem_53502, bytes_53476, "mem_53502"))
                    return 1;
                memmove(mem_53502.mem + 0, mem_53472.mem + i_53114 * N_52458 *
                        4, N_52458 * sizeof(int32_t));
                memmove(mem_53502.mem + res_52791 * 4, mem_53499.mem + 0,
                        sizeof(int32_t));
                if (memblock_unref(ctx, &mem_53499, "mem_53499") != 0)
                    return 1;
                if (memblock_set(ctx, &write_out_mem_53503, &mem_53502,
                                 "mem_53502") != 0)
                    return 1;
                if (memblock_unref(ctx, &mem_53502, "mem_53502") != 0)
                    return 1;
                if (memblock_unref(ctx, &mem_53499, "mem_53499") != 0)
                    return 1;
            }
            memmove(mem_53472.mem + i_53114 * N_52458 * 4,
                    write_out_mem_53503.mem + 0, N_52458 * sizeof(int32_t));
            if (memblock_unref(ctx, &write_out_mem_53503,
                               "write_out_mem_53503") != 0)
                return 1;
            memmove(mem_53506.mem + 0, mem_53467.mem + i_53114 * N_52458 * 4,
                    N_52458 * sizeof(float));
            
            struct memblock write_out_mem_53510;
            
            write_out_mem_53510.references = NULL;
            if (outside_bounds_dim_53098) {
                if (memblock_set(ctx, &write_out_mem_53510, &mem_53506,
                                 "mem_53506") != 0)
                    return 1;
            } else {
                struct memblock mem_53509;
                
                mem_53509.references = NULL;
                if (memblock_alloc(ctx, &mem_53509, bytes_53476, "mem_53509"))
                    return 1;
                memmove(mem_53509.mem + 0, mem_53467.mem + i_53114 * N_52458 *
                        4, N_52458 * sizeof(float));
                memmove(mem_53509.mem + res_52791 * 4, mem_53481.mem +
                        write_iter_53090 * 4, sizeof(float));
                if (memblock_set(ctx, &write_out_mem_53510, &mem_53509,
                                 "mem_53509") != 0)
                    return 1;
                if (memblock_unref(ctx, &mem_53509, "mem_53509") != 0)
                    return 1;
            }
            memmove(mem_53467.mem + i_53114 * N_52458 * 4,
                    write_out_mem_53510.mem + 0, N_52458 * sizeof(float));
            if (memblock_unref(ctx, &write_out_mem_53510,
                               "write_out_mem_53510") != 0)
                return 1;
            if (memblock_unref(ctx, &write_out_mem_53510,
                               "write_out_mem_53510") != 0)
                return 1;
            if (memblock_unref(ctx, &write_out_mem_53503,
                               "write_out_mem_53503") != 0)
                return 1;
        }
        memmove(mem_53462.mem + i_53114 * 4, mem_53478.mem + i_52736 * 4,
                sizeof(int32_t));
    }
    if (memblock_unref(ctx, &mem_53478, "mem_53478") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53481, "mem_53481") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53488, "mem_53488") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53491, "mem_53491") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53496, "mem_53496") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53506, "mem_53506") != 0)
        return 1;
    
    struct memblock mem_53518;
    
    mem_53518.references = NULL;
    if (memblock_alloc(ctx, &mem_53518, bytes_53460, "mem_53518"))
        return 1;
    
    struct memblock mem_53521;
    
    mem_53521.references = NULL;
    if (memblock_alloc(ctx, &mem_53521, bytes_53460, "mem_53521"))
        return 1;
    
    struct memblock mem_53524;
    
    mem_53524.references = NULL;
    if (memblock_alloc(ctx, &mem_53524, bytes_53460, "mem_53524"))
        return 1;
    for (int32_t i_53128 = 0; i_53128 < m_52459; i_53128++) {
        int32_t res_52813;
        int32_t redout_53118 = 0;
        
        for (int32_t i_53119 = 0; i_53119 < n_52463; i_53119++) {
            float x_52817 = ((float *) images_mem_53311.mem)[i_53128 * N_52460 +
                                                             i_53119];
            bool res_52818;
            
            res_52818 = futrts_isnan32(x_52817);
            
            bool cond_52819 = !res_52818;
            int32_t res_52820;
            
            if (cond_52819) {
                res_52820 = 1;
            } else {
                res_52820 = 0;
            }
            
            int32_t res_52816 = res_52820 + redout_53118;
            int32_t redout_tmp_53659 = res_52816;
            
            redout_53118 = redout_tmp_53659;
        }
        res_52813 = redout_53118;
        
        float res_52821;
        float redout_53120 = 0.0F;
        
        for (int32_t i_53121 = 0; i_53121 < n_52463; i_53121++) {
            float y_error_elem_52826 = ((float *) mem_53467.mem)[i_53128 *
                                                                 N_52458 +
                                                                 i_53121];
            bool cond_52827 = slt32(i_53121, res_52813);
            float res_52828;
            
            if (cond_52827) {
                res_52828 = y_error_elem_52826;
            } else {
                res_52828 = 0.0F;
            }
            
            float res_52829 = res_52828 * res_52828;
            float res_52824 = res_52829 + redout_53120;
            float redout_tmp_53660 = res_52824;
            
            redout_53120 = redout_tmp_53660;
        }
        res_52821 = redout_53120;
        
        int32_t r32_arg_52830 = res_52813 - k2p2_52480;
        float res_52831 = sitofp_i32_f32(r32_arg_52830);
        float sqrt_arg_52832 = res_52821 / res_52831;
        float res_52833;
        
        res_52833 = futrts_sqrt32(sqrt_arg_52832);
        
        float res_52834 = sitofp_i32_f32(res_52813);
        float t32_arg_52835 = hfrac_52465 * res_52834;
        int32_t res_52836 = fptosi_f32_i32(t32_arg_52835);
        
        ((int32_t *) mem_53518.mem)[i_53128] = res_52836;
        ((int32_t *) mem_53521.mem)[i_53128] = res_52813;
        ((float *) mem_53524.mem)[i_53128] = res_52833;
    }
    
    int32_t res_52840;
    int32_t redout_53132 = 0;
    
    for (int32_t i_53133 = 0; i_53133 < m_52459; i_53133++) {
        int32_t x_52844 = ((int32_t *) mem_53518.mem)[i_53133];
        int32_t res_52843 = smax32(x_52844, redout_53132);
        int32_t redout_tmp_53661 = res_52843;
        
        redout_53132 = redout_tmp_53661;
    }
    res_52840 = redout_53132;
    
    struct memblock mem_53533;
    
    mem_53533.references = NULL;
    if (memblock_alloc(ctx, &mem_53533, bytes_53460, "mem_53533"))
        return 1;
    for (int32_t i_53138 = 0; i_53138 < m_52459; i_53138++) {
        int32_t x_52848 = ((int32_t *) mem_53521.mem)[i_53138];
        int32_t x_52849 = ((int32_t *) mem_53518.mem)[i_53138];
        float res_52850;
        float redout_53134 = 0.0F;
        
        for (int32_t i_53135 = 0; i_53135 < res_52840; i_53135++) {
            bool cond_52855 = slt32(i_53135, x_52849);
            float res_52856;
            
            if (cond_52855) {
                int32_t x_52857 = x_52848 + i_53135;
                int32_t x_52858 = x_52857 - x_52849;
                int32_t i_52859 = 1 + x_52858;
                float res_52860 = ((float *) mem_53467.mem)[i_53138 * N_52458 +
                                                            i_52859];
                
                res_52856 = res_52860;
            } else {
                res_52856 = 0.0F;
            }
            
            float res_52853 = res_52856 + redout_53134;
            float redout_tmp_53663 = res_52853;
            
            redout_53134 = redout_tmp_53663;
        }
        res_52850 = redout_53134;
        ((float *) mem_53533.mem)[i_53138] = res_52850;
    }
    
    int32_t iota_arg_52862 = N_52458 - n_52463;
    bool bounds_invalid_upwards_52863 = slt32(iota_arg_52862, 0);
    bool eq_x_zz_52864 = 0 == iota_arg_52862;
    bool not_p_52865 = !bounds_invalid_upwards_52863;
    bool p_and_eq_x_y_52866 = eq_x_zz_52864 && not_p_52865;
    bool dim_zzero_52867 = bounds_invalid_upwards_52863 || p_and_eq_x_y_52866;
    bool both_empty_52868 = eq_x_zz_52864 && dim_zzero_52867;
    bool eq_x_y_52869 = iota_arg_52862 == 0;
    bool p_and_eq_x_y_52870 = bounds_invalid_upwards_52863 && eq_x_y_52869;
    bool dim_match_52871 = not_p_52865 || p_and_eq_x_y_52870;
    bool empty_or_match_52872 = both_empty_52868 || dim_match_52871;
    bool empty_or_match_cert_52873;
    
    if (!empty_or_match_52872) {
        ctx->error = msgprintf("Error at %s:\n%s%s%s%d%s%s\n",
                               "bfast-irreg.fut:133:1-314:110 -> bfast-irreg.fut:242:22-31 -> /futlib/array.fut:61:1-62:12",
                               "Function return value does not match shape of type ",
                               "*", "[", iota_arg_52862, "]", "intrinsics.i32");
        if (memblock_unref(ctx, &mem_53533, "mem_53533") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53524, "mem_53524") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53521, "mem_53521") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53518, "mem_53518") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53506, "mem_53506") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53496, "mem_53496") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53491, "mem_53491") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53488, "mem_53488") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53481, "mem_53481") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53478, "mem_53478") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53472, "mem_53472") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53467, "mem_53467") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53462, "mem_53462") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53449, "mem_53449") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53434, "mem_53434") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53419, "mem_53419") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53408, "mem_53408") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53402, "mem_53402") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53398, "mem_53398") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53364, "mem_53364") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53347, "mem_53347") != 0)
            return 1;
        if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_53342,
                           "lifted_1_zlzb_arg_mem_53342") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53616, "out_mem_53616") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53614, "out_mem_53614") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53612, "out_mem_53612") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53610, "out_mem_53610") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53607, "out_mem_53607") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53605, "out_mem_53605") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53602, "out_mem_53602") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53599, "out_mem_53599") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53596, "out_mem_53596") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53593, "out_mem_53593") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53589, "out_mem_53589") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53585, "out_mem_53585") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53582, "out_mem_53582") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53580, "out_mem_53580") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53578, "out_mem_53578") != 0)
            return 1;
        return 1;
    }
    
    int32_t x_52875 = 1 + n_52463;
    bool index_certs_52876;
    
    if (!x_52737) {
        ctx->error = msgprintf("Error at %s:\n%s%d%s%d%s\n",
                               "bfast-irreg.fut:133:1-314:110 -> bfast-irreg.fut:238:15-242:32 -> bfast-irreg.fut:240:63-81",
                               "Index [", i_52736,
                               "] out of bounds for array of shape [", N_52458,
                               "].");
        if (memblock_unref(ctx, &mem_53533, "mem_53533") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53524, "mem_53524") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53521, "mem_53521") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53518, "mem_53518") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53506, "mem_53506") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53496, "mem_53496") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53491, "mem_53491") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53488, "mem_53488") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53481, "mem_53481") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53478, "mem_53478") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53472, "mem_53472") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53467, "mem_53467") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53462, "mem_53462") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53449, "mem_53449") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53434, "mem_53434") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53419, "mem_53419") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53408, "mem_53408") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53402, "mem_53402") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53398, "mem_53398") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53364, "mem_53364") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53347, "mem_53347") != 0)
            return 1;
        if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_53342,
                           "lifted_1_zlzb_arg_mem_53342") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53616, "out_mem_53616") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53614, "out_mem_53614") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53612, "out_mem_53612") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53610, "out_mem_53610") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53607, "out_mem_53607") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53605, "out_mem_53605") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53602, "out_mem_53602") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53599, "out_mem_53599") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53596, "out_mem_53596") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53593, "out_mem_53593") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53589, "out_mem_53589") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53585, "out_mem_53585") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53582, "out_mem_53582") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53580, "out_mem_53580") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53578, "out_mem_53578") != 0)
            return 1;
        return 1;
    }
    
    int32_t r32_arg_52877 = ((int32_t *) mappingindices_mem_53310.mem)[i_52736];
    float res_52878 = sitofp_i32_f32(r32_arg_52877);
    int64_t binop_x_53537 = sext_i32_i64(iota_arg_52862);
    int64_t bytes_53536 = 4 * binop_x_53537;
    struct memblock mem_53538;
    
    mem_53538.references = NULL;
    if (memblock_alloc(ctx, &mem_53538, bytes_53536, "mem_53538"))
        return 1;
    for (int32_t i_53142 = 0; i_53142 < iota_arg_52862; i_53142++) {
        int32_t t_52881 = x_52875 + i_53142;
        int32_t i_52882 = t_52881 - 1;
        int32_t time_52883 =
                ((int32_t *) mappingindices_mem_53310.mem)[i_52882];
        float res_52884 = sitofp_i32_f32(time_52883);
        float logplus_arg_52885 = res_52884 / res_52878;
        bool cond_52886 = 2.7182817F < logplus_arg_52885;
        float res_52887;
        
        if (cond_52886) {
            float res_52888;
            
            res_52888 = futrts_log32(logplus_arg_52885);
            res_52887 = res_52888;
        } else {
            res_52887 = 1.0F;
        }
        
        float res_52889;
        
        res_52889 = futrts_sqrt32(res_52887);
        
        float res_52890 = lam_52466 * res_52889;
        
        ((float *) mem_53538.mem)[i_53142] = res_52890;
    }
    
    struct memblock mem_53543;
    
    mem_53543.references = NULL;
    if (memblock_alloc(ctx, &mem_53543, bytes_53460, "mem_53543"))
        return 1;
    
    struct memblock mem_53546;
    
    mem_53546.references = NULL;
    if (memblock_alloc(ctx, &mem_53546, bytes_53460, "mem_53546"))
        return 1;
    
    struct memblock mem_53551;
    
    mem_53551.references = NULL;
    if (memblock_alloc(ctx, &mem_53551, bytes_53536, "mem_53551"))
        return 1;
    for (int32_t i_53158 = 0; i_53158 < m_52459; i_53158++) {
        int32_t x_52893 = ((int32_t *) mem_53462.mem)[i_53158];
        int32_t x_52894 = ((int32_t *) mem_53521.mem)[i_53158];
        float x_52895 = ((float *) mem_53524.mem)[i_53158];
        int32_t x_52896 = ((int32_t *) mem_53518.mem)[i_53158];
        float x_52897 = ((float *) mem_53533.mem)[i_53158];
        int32_t y_52900 = x_52893 - x_52894;
        float res_52901 = sitofp_i32_f32(x_52894);
        float res_52902;
        
        res_52902 = futrts_sqrt32(res_52901);
        
        float y_52903 = x_52895 * res_52902;
        float discard_53149;
        float scanacc_53145 = 0.0F;
        
        for (int32_t i_53147 = 0; i_53147 < iota_arg_52862; i_53147++) {
            bool cond_52920 = sle32(y_52900, i_53147);
            float res_52921;
            
            if (cond_52920) {
                res_52921 = 0.0F;
            } else {
                bool cond_52922 = i_53147 == 0;
                float res_52923;
                
                if (cond_52922) {
                    res_52923 = x_52897;
                } else {
                    int32_t x_52924 = x_52894 - x_52896;
                    int32_t i_52925 = x_52924 + i_53147;
                    float negate_arg_52926 = ((float *) mem_53467.mem)[i_53158 *
                                                                       N_52458 +
                                                                       i_52925];
                    float x_52927 = 0.0F - negate_arg_52926;
                    int32_t i_52928 = x_52894 + i_53147;
                    float y_52929 = ((float *) mem_53467.mem)[i_53158 *
                                                              N_52458 +
                                                              i_52928];
                    float res_52930 = x_52927 + y_52929;
                    
                    res_52923 = res_52930;
                }
                res_52921 = res_52923;
            }
            
            float res_52918 = res_52921 + scanacc_53145;
            
            ((float *) mem_53551.mem)[i_53147] = res_52918;
            
            float scanacc_tmp_53667 = res_52918;
            
            scanacc_53145 = scanacc_tmp_53667;
        }
        discard_53149 = scanacc_53145;
        
        bool acc0_52936;
        int32_t acc0_52937;
        float acc0_52938;
        bool redout_53150;
        int32_t redout_53151;
        float redout_53152;
        
        redout_53150 = 0;
        redout_53151 = -1;
        redout_53152 = 0.0F;
        for (int32_t i_53153 = 0; i_53153 < iota_arg_52862; i_53153++) {
            float x_52953 = ((float *) mem_53551.mem)[i_53153];
            float x_52954 = ((float *) mem_53538.mem)[i_53153];
            int32_t x_52955 = i_53153;
            float res_52957 = x_52953 / y_52903;
            bool cond_52958 = slt32(i_53153, y_52900);
            bool res_52959;
            
            res_52959 = futrts_isnan32(res_52957);
            
            bool res_52960 = !res_52959;
            bool x_52961 = cond_52958 && res_52960;
            float res_52962 = (float) fabs(res_52957);
            bool res_52963 = x_52954 < res_52962;
            bool x_52964 = x_52961 && res_52963;
            float res_52965;
            
            if (cond_52958) {
                res_52965 = res_52957;
            } else {
                res_52965 = 0.0F;
            }
            
            bool res_52945;
            int32_t res_52946;
            
            if (redout_53150) {
                res_52945 = redout_53150;
                res_52946 = redout_53151;
            } else {
                bool x_52948 = !x_52964;
                bool y_52949 = x_52948 && redout_53150;
                bool res_52950 = y_52949 || x_52964;
                int32_t res_52951;
                
                if (x_52964) {
                    res_52951 = x_52955;
                } else {
                    res_52951 = redout_53151;
                }
                res_52945 = res_52950;
                res_52946 = res_52951;
            }
            
            float res_52952 = res_52965 + redout_53152;
            bool redout_tmp_53669 = res_52945;
            int32_t redout_tmp_53670 = res_52946;
            float redout_tmp_53671;
            
            redout_tmp_53671 = res_52952;
            redout_53150 = redout_tmp_53669;
            redout_53151 = redout_tmp_53670;
            redout_53152 = redout_tmp_53671;
        }
        acc0_52936 = redout_53150;
        acc0_52937 = redout_53151;
        acc0_52938 = redout_53152;
        
        int32_t res_52972;
        
        if (acc0_52936) {
            res_52972 = acc0_52937;
        } else {
            res_52972 = -1;
        }
        
        bool cond_52974 = !acc0_52936;
        int32_t fst_breakzq_52975;
        
        if (cond_52974) {
            fst_breakzq_52975 = -1;
        } else {
            bool cond_52976 = slt32(res_52972, y_52900);
            int32_t res_52977;
            
            if (cond_52976) {
                int32_t i_52978 = x_52894 + res_52972;
                int32_t x_52979 = ((int32_t *) mem_53472.mem)[i_53158 *
                                                              N_52458 +
                                                              i_52978];
                int32_t res_52980 = x_52979 - n_52463;
                
                res_52977 = res_52980;
            } else {
                res_52977 = -1;
            }
            
            int32_t x_52981 = res_52977 - 1;
            int32_t x_52982 = sdiv32(x_52981, 2);
            int32_t x_52983 = 2 * x_52982;
            int32_t res_52984 = 1 + x_52983;
            
            fst_breakzq_52975 = res_52984;
        }
        
        bool cond_52985 = sle32(x_52894, 5);
        bool res_52986 = sle32(y_52900, 5);
        bool x_52987 = !cond_52985;
        bool y_52988 = res_52986 && x_52987;
        bool cond_52989 = cond_52985 || y_52988;
        int32_t fst_breakzq_52990;
        
        if (cond_52989) {
            fst_breakzq_52990 = -2;
        } else {
            fst_breakzq_52990 = fst_breakzq_52975;
        }
        ((int32_t *) mem_53543.mem)[i_53158] = fst_breakzq_52990;
        ((float *) mem_53546.mem)[i_53158] = acc0_52938;
    }
    if (memblock_unref(ctx, &mem_53538, "mem_53538") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53551, "mem_53551") != 0)
        return 1;
    out_arrsizze_53579 = m_52459;
    out_arrsizze_53581 = m_52459;
    out_arrsizze_53583 = N_52458;
    out_arrsizze_53584 = k2p2zq_52482;
    out_arrsizze_53586 = m_52459;
    out_arrsizze_53587 = k2p2zq_52482;
    out_arrsizze_53588 = k2p2zq_52482;
    out_arrsizze_53590 = m_52459;
    out_arrsizze_53591 = k2p2zq_52482;
    out_arrsizze_53592 = j_m_i_52616;
    out_arrsizze_53594 = m_52459;
    out_arrsizze_53595 = k2p2zq_52482;
    out_arrsizze_53597 = m_52459;
    out_arrsizze_53598 = k2p2zq_52482;
    out_arrsizze_53600 = m_52459;
    out_arrsizze_53601 = N_52458;
    out_arrsizze_53603 = m_52459;
    out_arrsizze_53604 = N_52458;
    out_arrsizze_53606 = m_52459;
    out_arrsizze_53608 = m_52459;
    out_arrsizze_53609 = N_52458;
    out_arrsizze_53611 = m_52459;
    out_arrsizze_53613 = m_52459;
    out_arrsizze_53615 = m_52459;
    out_arrsizze_53617 = m_52459;
    if (memblock_set(ctx, &out_mem_53578, &mem_53543, "mem_53543") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_53580, &mem_53546, "mem_53546") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_53582, &mem_53347, "mem_53347") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_53585, &mem_53364, "mem_53364") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_53589, &mem_53398, "mem_53398") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_53593, &mem_53419, "mem_53419") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_53596, &mem_53434, "mem_53434") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_53599, &mem_53449, "mem_53449") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_53602, &mem_53467, "mem_53467") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_53605, &mem_53462, "mem_53462") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_53607, &mem_53472, "mem_53472") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_53610, &mem_53518, "mem_53518") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_53612, &mem_53521, "mem_53521") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_53614, &mem_53524, "mem_53524") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_53616, &mem_53533, "mem_53533") != 0)
        return 1;
    (*out_mem_p_53672).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_53672, &out_mem_53578, "out_mem_53578") !=
        0)
        return 1;
    *out_out_arrsizze_53673 = out_arrsizze_53579;
    (*out_mem_p_53674).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_53674, &out_mem_53580, "out_mem_53580") !=
        0)
        return 1;
    *out_out_arrsizze_53675 = out_arrsizze_53581;
    (*out_mem_p_53676).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_53676, &out_mem_53582, "out_mem_53582") !=
        0)
        return 1;
    *out_out_arrsizze_53677 = out_arrsizze_53583;
    *out_out_arrsizze_53678 = out_arrsizze_53584;
    (*out_mem_p_53679).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_53679, &out_mem_53585, "out_mem_53585") !=
        0)
        return 1;
    *out_out_arrsizze_53680 = out_arrsizze_53586;
    *out_out_arrsizze_53681 = out_arrsizze_53587;
    *out_out_arrsizze_53682 = out_arrsizze_53588;
    (*out_mem_p_53683).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_53683, &out_mem_53589, "out_mem_53589") !=
        0)
        return 1;
    *out_out_arrsizze_53684 = out_arrsizze_53590;
    *out_out_arrsizze_53685 = out_arrsizze_53591;
    *out_out_arrsizze_53686 = out_arrsizze_53592;
    (*out_mem_p_53687).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_53687, &out_mem_53593, "out_mem_53593") !=
        0)
        return 1;
    *out_out_arrsizze_53688 = out_arrsizze_53594;
    *out_out_arrsizze_53689 = out_arrsizze_53595;
    (*out_mem_p_53690).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_53690, &out_mem_53596, "out_mem_53596") !=
        0)
        return 1;
    *out_out_arrsizze_53691 = out_arrsizze_53597;
    *out_out_arrsizze_53692 = out_arrsizze_53598;
    (*out_mem_p_53693).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_53693, &out_mem_53599, "out_mem_53599") !=
        0)
        return 1;
    *out_out_arrsizze_53694 = out_arrsizze_53600;
    *out_out_arrsizze_53695 = out_arrsizze_53601;
    (*out_mem_p_53696).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_53696, &out_mem_53602, "out_mem_53602") !=
        0)
        return 1;
    *out_out_arrsizze_53697 = out_arrsizze_53603;
    *out_out_arrsizze_53698 = out_arrsizze_53604;
    (*out_mem_p_53699).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_53699, &out_mem_53605, "out_mem_53605") !=
        0)
        return 1;
    *out_out_arrsizze_53700 = out_arrsizze_53606;
    (*out_mem_p_53701).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_53701, &out_mem_53607, "out_mem_53607") !=
        0)
        return 1;
    *out_out_arrsizze_53702 = out_arrsizze_53608;
    *out_out_arrsizze_53703 = out_arrsizze_53609;
    (*out_mem_p_53704).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_53704, &out_mem_53610, "out_mem_53610") !=
        0)
        return 1;
    *out_out_arrsizze_53705 = out_arrsizze_53611;
    (*out_mem_p_53706).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_53706, &out_mem_53612, "out_mem_53612") !=
        0)
        return 1;
    *out_out_arrsizze_53707 = out_arrsizze_53613;
    (*out_mem_p_53708).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_53708, &out_mem_53614, "out_mem_53614") !=
        0)
        return 1;
    *out_out_arrsizze_53709 = out_arrsizze_53615;
    (*out_mem_p_53710).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_53710, &out_mem_53616, "out_mem_53616") !=
        0)
        return 1;
    *out_out_arrsizze_53711 = out_arrsizze_53617;
    if (memblock_unref(ctx, &mem_53551, "mem_53551") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53546, "mem_53546") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53543, "mem_53543") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53538, "mem_53538") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53533, "mem_53533") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53524, "mem_53524") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53521, "mem_53521") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53518, "mem_53518") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53506, "mem_53506") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53496, "mem_53496") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53491, "mem_53491") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53488, "mem_53488") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53481, "mem_53481") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53478, "mem_53478") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53472, "mem_53472") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53467, "mem_53467") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53462, "mem_53462") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53449, "mem_53449") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53434, "mem_53434") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53419, "mem_53419") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53408, "mem_53408") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53402, "mem_53402") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53398, "mem_53398") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53364, "mem_53364") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53347, "mem_53347") != 0)
        return 1;
    if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_53342,
                       "lifted_1_zlzb_arg_mem_53342") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_53616, "out_mem_53616") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_53614, "out_mem_53614") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_53612, "out_mem_53612") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_53610, "out_mem_53610") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_53607, "out_mem_53607") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_53605, "out_mem_53605") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_53602, "out_mem_53602") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_53599, "out_mem_53599") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_53596, "out_mem_53596") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_53593, "out_mem_53593") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_53589, "out_mem_53589") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_53585, "out_mem_53585") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_53582, "out_mem_53582") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_53580, "out_mem_53580") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_53578, "out_mem_53578") != 0)
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
                       struct futhark_i32_1d **out0,
                       struct futhark_f32_1d **out1,
                       struct futhark_f32_2d **out2,
                       struct futhark_f32_3d **out3,
                       struct futhark_f32_3d **out4,
                       struct futhark_f32_2d **out5,
                       struct futhark_f32_2d **out6,
                       struct futhark_f32_2d **out7,
                       struct futhark_f32_2d **out8,
                       struct futhark_i32_1d **out9,
                       struct futhark_i32_2d **out10,
                       struct futhark_i32_1d **out11,
                       struct futhark_i32_1d **out12,
                       struct futhark_f32_1d **out13,
                       struct futhark_f32_1d **out14, const int32_t in0, const
                       int32_t in1, const int32_t in2, const float in3, const
                       float in4, const float in5, const
                       struct futhark_i32_1d *in6, const
                       struct futhark_f32_2d *in7)
{
    struct memblock mappingindices_mem_53310;
    
    mappingindices_mem_53310.references = NULL;
    
    struct memblock images_mem_53311;
    
    images_mem_53311.references = NULL;
    
    int32_t N_52458;
    int32_t m_52459;
    int32_t N_52460;
    int32_t trend_52461;
    int32_t k_52462;
    int32_t n_52463;
    float freq_52464;
    float hfrac_52465;
    float lam_52466;
    struct memblock out_mem_53578;
    
    out_mem_53578.references = NULL;
    
    int32_t out_arrsizze_53579;
    struct memblock out_mem_53580;
    
    out_mem_53580.references = NULL;
    
    int32_t out_arrsizze_53581;
    struct memblock out_mem_53582;
    
    out_mem_53582.references = NULL;
    
    int32_t out_arrsizze_53583;
    int32_t out_arrsizze_53584;
    struct memblock out_mem_53585;
    
    out_mem_53585.references = NULL;
    
    int32_t out_arrsizze_53586;
    int32_t out_arrsizze_53587;
    int32_t out_arrsizze_53588;
    struct memblock out_mem_53589;
    
    out_mem_53589.references = NULL;
    
    int32_t out_arrsizze_53590;
    int32_t out_arrsizze_53591;
    int32_t out_arrsizze_53592;
    struct memblock out_mem_53593;
    
    out_mem_53593.references = NULL;
    
    int32_t out_arrsizze_53594;
    int32_t out_arrsizze_53595;
    struct memblock out_mem_53596;
    
    out_mem_53596.references = NULL;
    
    int32_t out_arrsizze_53597;
    int32_t out_arrsizze_53598;
    struct memblock out_mem_53599;
    
    out_mem_53599.references = NULL;
    
    int32_t out_arrsizze_53600;
    int32_t out_arrsizze_53601;
    struct memblock out_mem_53602;
    
    out_mem_53602.references = NULL;
    
    int32_t out_arrsizze_53603;
    int32_t out_arrsizze_53604;
    struct memblock out_mem_53605;
    
    out_mem_53605.references = NULL;
    
    int32_t out_arrsizze_53606;
    struct memblock out_mem_53607;
    
    out_mem_53607.references = NULL;
    
    int32_t out_arrsizze_53608;
    int32_t out_arrsizze_53609;
    struct memblock out_mem_53610;
    
    out_mem_53610.references = NULL;
    
    int32_t out_arrsizze_53611;
    struct memblock out_mem_53612;
    
    out_mem_53612.references = NULL;
    
    int32_t out_arrsizze_53613;
    struct memblock out_mem_53614;
    
    out_mem_53614.references = NULL;
    
    int32_t out_arrsizze_53615;
    struct memblock out_mem_53616;
    
    out_mem_53616.references = NULL;
    
    int32_t out_arrsizze_53617;
    
    lock_lock(&ctx->lock);
    trend_52461 = in0;
    k_52462 = in1;
    n_52463 = in2;
    freq_52464 = in3;
    hfrac_52465 = in4;
    lam_52466 = in5;
    mappingindices_mem_53310 = in6->mem;
    N_52458 = in6->shape[0];
    images_mem_53311 = in7->mem;
    m_52459 = in7->shape[0];
    N_52460 = in7->shape[1];
    
    int ret = futrts_main(ctx, &out_mem_53578, &out_arrsizze_53579,
                          &out_mem_53580, &out_arrsizze_53581, &out_mem_53582,
                          &out_arrsizze_53583, &out_arrsizze_53584,
                          &out_mem_53585, &out_arrsizze_53586,
                          &out_arrsizze_53587, &out_arrsizze_53588,
                          &out_mem_53589, &out_arrsizze_53590,
                          &out_arrsizze_53591, &out_arrsizze_53592,
                          &out_mem_53593, &out_arrsizze_53594,
                          &out_arrsizze_53595, &out_mem_53596,
                          &out_arrsizze_53597, &out_arrsizze_53598,
                          &out_mem_53599, &out_arrsizze_53600,
                          &out_arrsizze_53601, &out_mem_53602,
                          &out_arrsizze_53603, &out_arrsizze_53604,
                          &out_mem_53605, &out_arrsizze_53606, &out_mem_53607,
                          &out_arrsizze_53608, &out_arrsizze_53609,
                          &out_mem_53610, &out_arrsizze_53611, &out_mem_53612,
                          &out_arrsizze_53613, &out_mem_53614,
                          &out_arrsizze_53615, &out_mem_53616,
                          &out_arrsizze_53617, mappingindices_mem_53310,
                          images_mem_53311, N_52458, m_52459, N_52460,
                          trend_52461, k_52462, n_52463, freq_52464,
                          hfrac_52465, lam_52466);
    
    if (ret == 0) {
        assert((*out0 =
                (struct futhark_i32_1d *) malloc(sizeof(struct futhark_i32_1d))) !=
            NULL);
        (*out0)->mem = out_mem_53578;
        (*out0)->shape[0] = out_arrsizze_53579;
        assert((*out1 =
                (struct futhark_f32_1d *) malloc(sizeof(struct futhark_f32_1d))) !=
            NULL);
        (*out1)->mem = out_mem_53580;
        (*out1)->shape[0] = out_arrsizze_53581;
        assert((*out2 =
                (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) !=
            NULL);
        (*out2)->mem = out_mem_53582;
        (*out2)->shape[0] = out_arrsizze_53583;
        (*out2)->shape[1] = out_arrsizze_53584;
        assert((*out3 =
                (struct futhark_f32_3d *) malloc(sizeof(struct futhark_f32_3d))) !=
            NULL);
        (*out3)->mem = out_mem_53585;
        (*out3)->shape[0] = out_arrsizze_53586;
        (*out3)->shape[1] = out_arrsizze_53587;
        (*out3)->shape[2] = out_arrsizze_53588;
        assert((*out4 =
                (struct futhark_f32_3d *) malloc(sizeof(struct futhark_f32_3d))) !=
            NULL);
        (*out4)->mem = out_mem_53589;
        (*out4)->shape[0] = out_arrsizze_53590;
        (*out4)->shape[1] = out_arrsizze_53591;
        (*out4)->shape[2] = out_arrsizze_53592;
        assert((*out5 =
                (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) !=
            NULL);
        (*out5)->mem = out_mem_53593;
        (*out5)->shape[0] = out_arrsizze_53594;
        (*out5)->shape[1] = out_arrsizze_53595;
        assert((*out6 =
                (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) !=
            NULL);
        (*out6)->mem = out_mem_53596;
        (*out6)->shape[0] = out_arrsizze_53597;
        (*out6)->shape[1] = out_arrsizze_53598;
        assert((*out7 =
                (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) !=
            NULL);
        (*out7)->mem = out_mem_53599;
        (*out7)->shape[0] = out_arrsizze_53600;
        (*out7)->shape[1] = out_arrsizze_53601;
        assert((*out8 =
                (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) !=
            NULL);
        (*out8)->mem = out_mem_53602;
        (*out8)->shape[0] = out_arrsizze_53603;
        (*out8)->shape[1] = out_arrsizze_53604;
        assert((*out9 =
                (struct futhark_i32_1d *) malloc(sizeof(struct futhark_i32_1d))) !=
            NULL);
        (*out9)->mem = out_mem_53605;
        (*out9)->shape[0] = out_arrsizze_53606;
        assert((*out10 =
                (struct futhark_i32_2d *) malloc(sizeof(struct futhark_i32_2d))) !=
            NULL);
        (*out10)->mem = out_mem_53607;
        (*out10)->shape[0] = out_arrsizze_53608;
        (*out10)->shape[1] = out_arrsizze_53609;
        assert((*out11 =
                (struct futhark_i32_1d *) malloc(sizeof(struct futhark_i32_1d))) !=
            NULL);
        (*out11)->mem = out_mem_53610;
        (*out11)->shape[0] = out_arrsizze_53611;
        assert((*out12 =
                (struct futhark_i32_1d *) malloc(sizeof(struct futhark_i32_1d))) !=
            NULL);
        (*out12)->mem = out_mem_53612;
        (*out12)->shape[0] = out_arrsizze_53613;
        assert((*out13 =
                (struct futhark_f32_1d *) malloc(sizeof(struct futhark_f32_1d))) !=
            NULL);
        (*out13)->mem = out_mem_53614;
        (*out13)->shape[0] = out_arrsizze_53615;
        assert((*out14 =
                (struct futhark_f32_1d *) malloc(sizeof(struct futhark_f32_1d))) !=
            NULL);
        (*out14)->mem = out_mem_53616;
        (*out14)->shape[0] = out_arrsizze_53617;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
