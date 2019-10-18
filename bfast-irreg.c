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
                       struct futhark_i32_2d **out10, const int32_t in0, const
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
    
    int32_t read_value_53584;
    
    if (read_scalar(&i32_info, &read_value_53584) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 0,
              i32_info.type_name, strerror(errno));
    
    int32_t read_value_53585;
    
    if (read_scalar(&i32_info, &read_value_53585) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 1,
              i32_info.type_name, strerror(errno));
    
    int32_t read_value_53586;
    
    if (read_scalar(&i32_info, &read_value_53586) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 2,
              i32_info.type_name, strerror(errno));
    
    float read_value_53587;
    
    if (read_scalar(&f32_info, &read_value_53587) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 3,
              f32_info.type_name, strerror(errno));
    
    float read_value_53588;
    
    if (read_scalar(&f32_info, &read_value_53588) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 4,
              f32_info.type_name, strerror(errno));
    
    float read_value_53589;
    
    if (read_scalar(&f32_info, &read_value_53589) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 5,
              f32_info.type_name, strerror(errno));
    
    struct futhark_i32_1d *read_value_53590;
    int64_t read_shape_53591[1];
    int32_t *read_arr_53592 = NULL;
    
    errno = 0;
    if (read_array(&i32_info, (void **) &read_arr_53592, read_shape_53591, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 6, "[]",
              i32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_53593;
    int64_t read_shape_53594[2];
    float *read_arr_53595 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_53595, read_shape_53594, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 7, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_i32_1d *result_53596;
    struct futhark_f32_1d *result_53597;
    struct futhark_f32_2d *result_53598;
    struct futhark_f32_3d *result_53599;
    struct futhark_f32_3d *result_53600;
    struct futhark_f32_2d *result_53601;
    struct futhark_f32_2d *result_53602;
    struct futhark_f32_2d *result_53603;
    struct futhark_f32_2d *result_53604;
    struct futhark_i32_1d *result_53605;
    struct futhark_i32_2d *result_53606;
    
    if (perform_warmup) {
        time_runs = 0;
        
        int r;
        
        ;
        ;
        ;
        ;
        ;
        ;
        assert((read_value_53590 = futhark_new_i32_1d(ctx, read_arr_53592,
                                                      read_shape_53591[0])) !=
            0);
        assert((read_value_53593 = futhark_new_f32_2d(ctx, read_arr_53595,
                                                      read_shape_53594[0],
                                                      read_shape_53594[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_main(ctx, &result_53596, &result_53597, &result_53598,
                               &result_53599, &result_53600, &result_53601,
                               &result_53602, &result_53603, &result_53604,
                               &result_53605, &result_53606, read_value_53584,
                               read_value_53585, read_value_53586,
                               read_value_53587, read_value_53588,
                               read_value_53589, read_value_53590,
                               read_value_53593);
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
        assert(futhark_free_i32_1d(ctx, read_value_53590) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_53593) == 0);
        assert(futhark_free_i32_1d(ctx, result_53596) == 0);
        assert(futhark_free_f32_1d(ctx, result_53597) == 0);
        assert(futhark_free_f32_2d(ctx, result_53598) == 0);
        assert(futhark_free_f32_3d(ctx, result_53599) == 0);
        assert(futhark_free_f32_3d(ctx, result_53600) == 0);
        assert(futhark_free_f32_2d(ctx, result_53601) == 0);
        assert(futhark_free_f32_2d(ctx, result_53602) == 0);
        assert(futhark_free_f32_2d(ctx, result_53603) == 0);
        assert(futhark_free_f32_2d(ctx, result_53604) == 0);
        assert(futhark_free_i32_1d(ctx, result_53605) == 0);
        assert(futhark_free_i32_2d(ctx, result_53606) == 0);
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
        assert((read_value_53590 = futhark_new_i32_1d(ctx, read_arr_53592,
                                                      read_shape_53591[0])) !=
            0);
        assert((read_value_53593 = futhark_new_f32_2d(ctx, read_arr_53595,
                                                      read_shape_53594[0],
                                                      read_shape_53594[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_main(ctx, &result_53596, &result_53597, &result_53598,
                               &result_53599, &result_53600, &result_53601,
                               &result_53602, &result_53603, &result_53604,
                               &result_53605, &result_53606, read_value_53584,
                               read_value_53585, read_value_53586,
                               read_value_53587, read_value_53588,
                               read_value_53589, read_value_53590,
                               read_value_53593);
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
        assert(futhark_free_i32_1d(ctx, read_value_53590) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_53593) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_i32_1d(ctx, result_53596) == 0);
            assert(futhark_free_f32_1d(ctx, result_53597) == 0);
            assert(futhark_free_f32_2d(ctx, result_53598) == 0);
            assert(futhark_free_f32_3d(ctx, result_53599) == 0);
            assert(futhark_free_f32_3d(ctx, result_53600) == 0);
            assert(futhark_free_f32_2d(ctx, result_53601) == 0);
            assert(futhark_free_f32_2d(ctx, result_53602) == 0);
            assert(futhark_free_f32_2d(ctx, result_53603) == 0);
            assert(futhark_free_f32_2d(ctx, result_53604) == 0);
            assert(futhark_free_i32_1d(ctx, result_53605) == 0);
            assert(futhark_free_i32_2d(ctx, result_53606) == 0);
        }
    }
    ;
    ;
    ;
    ;
    ;
    ;
    free(read_arr_53592);
    free(read_arr_53595);
    if (binary_output)
        set_binary_mode(stdout);
    {
        int32_t *arr = calloc(sizeof(int32_t), futhark_shape_i32_1d(ctx,
                                                                    result_53596)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_i32_1d(ctx, result_53596, arr) == 0);
        write_array(stdout, binary_output, &i32_info, arr,
                    futhark_shape_i32_1d(ctx, result_53596), 1);
        free(arr);
    }
    printf("\n");
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_1d(ctx,
                                                                result_53597)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_1d(ctx, result_53597, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_1d(ctx, result_53597), 1);
        free(arr);
    }
    printf("\n");
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_2d(ctx,
                                                                result_53598)[0] *
                            futhark_shape_f32_2d(ctx, result_53598)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_2d(ctx, result_53598, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_2d(ctx, result_53598), 2);
        free(arr);
    }
    printf("\n");
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_3d(ctx,
                                                                result_53599)[0] *
                            futhark_shape_f32_3d(ctx, result_53599)[1] *
                            futhark_shape_f32_3d(ctx, result_53599)[2]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_3d(ctx, result_53599, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_3d(ctx, result_53599), 3);
        free(arr);
    }
    printf("\n");
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_3d(ctx,
                                                                result_53600)[0] *
                            futhark_shape_f32_3d(ctx, result_53600)[1] *
                            futhark_shape_f32_3d(ctx, result_53600)[2]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_3d(ctx, result_53600, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_3d(ctx, result_53600), 3);
        free(arr);
    }
    printf("\n");
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_2d(ctx,
                                                                result_53601)[0] *
                            futhark_shape_f32_2d(ctx, result_53601)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_2d(ctx, result_53601, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_2d(ctx, result_53601), 2);
        free(arr);
    }
    printf("\n");
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_2d(ctx,
                                                                result_53602)[0] *
                            futhark_shape_f32_2d(ctx, result_53602)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_2d(ctx, result_53602, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_2d(ctx, result_53602), 2);
        free(arr);
    }
    printf("\n");
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_2d(ctx,
                                                                result_53603)[0] *
                            futhark_shape_f32_2d(ctx, result_53603)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_2d(ctx, result_53603, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_2d(ctx, result_53603), 2);
        free(arr);
    }
    printf("\n");
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_2d(ctx,
                                                                result_53604)[0] *
                            futhark_shape_f32_2d(ctx, result_53604)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_2d(ctx, result_53604, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_2d(ctx, result_53604), 2);
        free(arr);
    }
    printf("\n");
    {
        int32_t *arr = calloc(sizeof(int32_t), futhark_shape_i32_1d(ctx,
                                                                    result_53605)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_i32_1d(ctx, result_53605, arr) == 0);
        write_array(stdout, binary_output, &i32_info, arr,
                    futhark_shape_i32_1d(ctx, result_53605), 1);
        free(arr);
    }
    printf("\n");
    {
        int32_t *arr = calloc(sizeof(int32_t), futhark_shape_i32_2d(ctx,
                                                                    result_53606)[0] *
                              futhark_shape_i32_2d(ctx, result_53606)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_i32_2d(ctx, result_53606, arr) == 0);
        write_array(stdout, binary_output, &i32_info, arr,
                    futhark_shape_i32_2d(ctx, result_53606), 2);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_i32_1d(ctx, result_53596) == 0);
    assert(futhark_free_f32_1d(ctx, result_53597) == 0);
    assert(futhark_free_f32_2d(ctx, result_53598) == 0);
    assert(futhark_free_f32_3d(ctx, result_53599) == 0);
    assert(futhark_free_f32_3d(ctx, result_53600) == 0);
    assert(futhark_free_f32_2d(ctx, result_53601) == 0);
    assert(futhark_free_f32_2d(ctx, result_53602) == 0);
    assert(futhark_free_f32_2d(ctx, result_53603) == 0);
    assert(futhark_free_f32_2d(ctx, result_53604) == 0);
    assert(futhark_free_i32_1d(ctx, result_53605) == 0);
    assert(futhark_free_i32_2d(ctx, result_53606) == 0);
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
                       struct memblock *out_mem_p_53552,
                       int32_t *out_out_arrsizze_53553,
                       struct memblock *out_mem_p_53554,
                       int32_t *out_out_arrsizze_53555,
                       struct memblock *out_mem_p_53556,
                       int32_t *out_out_arrsizze_53557,
                       int32_t *out_out_arrsizze_53558,
                       struct memblock *out_mem_p_53559,
                       int32_t *out_out_arrsizze_53560,
                       int32_t *out_out_arrsizze_53561,
                       int32_t *out_out_arrsizze_53562,
                       struct memblock *out_mem_p_53563,
                       int32_t *out_out_arrsizze_53564,
                       int32_t *out_out_arrsizze_53565,
                       int32_t *out_out_arrsizze_53566,
                       struct memblock *out_mem_p_53567,
                       int32_t *out_out_arrsizze_53568,
                       int32_t *out_out_arrsizze_53569,
                       struct memblock *out_mem_p_53570,
                       int32_t *out_out_arrsizze_53571,
                       int32_t *out_out_arrsizze_53572,
                       struct memblock *out_mem_p_53573,
                       int32_t *out_out_arrsizze_53574,
                       int32_t *out_out_arrsizze_53575,
                       struct memblock *out_mem_p_53576,
                       int32_t *out_out_arrsizze_53577,
                       int32_t *out_out_arrsizze_53578,
                       struct memblock *out_mem_p_53579,
                       int32_t *out_out_arrsizze_53580,
                       struct memblock *out_mem_p_53581,
                       int32_t *out_out_arrsizze_53582,
                       int32_t *out_out_arrsizze_53583,
                       struct memblock mappingindices_mem_53198,
                       struct memblock images_mem_53199, int32_t N_52346,
                       int32_t m_52347, int32_t N_52348, int32_t trend_52349,
                       int32_t k_52350, int32_t n_52351, float freq_52352,
                       float hfrac_52353, float lam_52354);
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
                       struct memblock *out_mem_p_53552,
                       int32_t *out_out_arrsizze_53553,
                       struct memblock *out_mem_p_53554,
                       int32_t *out_out_arrsizze_53555,
                       struct memblock *out_mem_p_53556,
                       int32_t *out_out_arrsizze_53557,
                       int32_t *out_out_arrsizze_53558,
                       struct memblock *out_mem_p_53559,
                       int32_t *out_out_arrsizze_53560,
                       int32_t *out_out_arrsizze_53561,
                       int32_t *out_out_arrsizze_53562,
                       struct memblock *out_mem_p_53563,
                       int32_t *out_out_arrsizze_53564,
                       int32_t *out_out_arrsizze_53565,
                       int32_t *out_out_arrsizze_53566,
                       struct memblock *out_mem_p_53567,
                       int32_t *out_out_arrsizze_53568,
                       int32_t *out_out_arrsizze_53569,
                       struct memblock *out_mem_p_53570,
                       int32_t *out_out_arrsizze_53571,
                       int32_t *out_out_arrsizze_53572,
                       struct memblock *out_mem_p_53573,
                       int32_t *out_out_arrsizze_53574,
                       int32_t *out_out_arrsizze_53575,
                       struct memblock *out_mem_p_53576,
                       int32_t *out_out_arrsizze_53577,
                       int32_t *out_out_arrsizze_53578,
                       struct memblock *out_mem_p_53579,
                       int32_t *out_out_arrsizze_53580,
                       struct memblock *out_mem_p_53581,
                       int32_t *out_out_arrsizze_53582,
                       int32_t *out_out_arrsizze_53583,
                       struct memblock mappingindices_mem_53198,
                       struct memblock images_mem_53199, int32_t N_52346,
                       int32_t m_52347, int32_t N_52348, int32_t trend_52349,
                       int32_t k_52350, int32_t n_52351, float freq_52352,
                       float hfrac_52353, float lam_52354)
{
    struct memblock out_mem_53466;
    
    out_mem_53466.references = NULL;
    
    int32_t out_arrsizze_53467;
    struct memblock out_mem_53468;
    
    out_mem_53468.references = NULL;
    
    int32_t out_arrsizze_53469;
    struct memblock out_mem_53470;
    
    out_mem_53470.references = NULL;
    
    int32_t out_arrsizze_53471;
    int32_t out_arrsizze_53472;
    struct memblock out_mem_53473;
    
    out_mem_53473.references = NULL;
    
    int32_t out_arrsizze_53474;
    int32_t out_arrsizze_53475;
    int32_t out_arrsizze_53476;
    struct memblock out_mem_53477;
    
    out_mem_53477.references = NULL;
    
    int32_t out_arrsizze_53478;
    int32_t out_arrsizze_53479;
    int32_t out_arrsizze_53480;
    struct memblock out_mem_53481;
    
    out_mem_53481.references = NULL;
    
    int32_t out_arrsizze_53482;
    int32_t out_arrsizze_53483;
    struct memblock out_mem_53484;
    
    out_mem_53484.references = NULL;
    
    int32_t out_arrsizze_53485;
    int32_t out_arrsizze_53486;
    struct memblock out_mem_53487;
    
    out_mem_53487.references = NULL;
    
    int32_t out_arrsizze_53488;
    int32_t out_arrsizze_53489;
    struct memblock out_mem_53490;
    
    out_mem_53490.references = NULL;
    
    int32_t out_arrsizze_53491;
    int32_t out_arrsizze_53492;
    struct memblock out_mem_53493;
    
    out_mem_53493.references = NULL;
    
    int32_t out_arrsizze_53494;
    struct memblock out_mem_53495;
    
    out_mem_53495.references = NULL;
    
    int32_t out_arrsizze_53496;
    int32_t out_arrsizze_53497;
    bool dim_zzero_52357 = 0 == m_52347;
    bool dim_zzero_52358 = 0 == N_52348;
    bool old_empty_52359 = dim_zzero_52357 || dim_zzero_52358;
    bool dim_zzero_52360 = 0 == N_52346;
    bool new_empty_52361 = dim_zzero_52357 || dim_zzero_52360;
    bool both_empty_52362 = old_empty_52359 && new_empty_52361;
    bool dim_match_52363 = N_52346 == N_52348;
    bool empty_or_match_52364 = both_empty_52362 || dim_match_52363;
    bool empty_or_match_cert_52365;
    
    if (!empty_or_match_52364) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "bfast-irreg.fut:133:1-314:84",
                               "function arguments of wrong shape");
        if (memblock_unref(ctx, &out_mem_53495, "out_mem_53495") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53493, "out_mem_53493") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53490, "out_mem_53490") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53487, "out_mem_53487") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53484, "out_mem_53484") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53481, "out_mem_53481") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53477, "out_mem_53477") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53473, "out_mem_53473") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53470, "out_mem_53470") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53468, "out_mem_53468") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53466, "out_mem_53466") != 0)
            return 1;
        return 1;
    }
    
    int32_t x_52367 = 2 * k_52350;
    int32_t k2p2_52368 = 2 + x_52367;
    bool cond_52369 = slt32(0, trend_52349);
    int32_t k2p2zq_52370;
    
    if (cond_52369) {
        k2p2zq_52370 = k2p2_52368;
    } else {
        int32_t res_52371 = k2p2_52368 - 1;
        
        k2p2zq_52370 = res_52371;
    }
    
    int64_t binop_x_53201 = sext_i32_i64(k2p2zq_52370);
    int64_t binop_y_53202 = sext_i32_i64(N_52346);
    int64_t binop_x_53203 = binop_x_53201 * binop_y_53202;
    int64_t bytes_53200 = 4 * binop_x_53203;
    int64_t binop_x_53216 = sext_i32_i64(k2p2zq_52370);
    int64_t binop_y_53217 = sext_i32_i64(N_52346);
    int64_t binop_x_53218 = binop_x_53216 * binop_y_53217;
    int64_t bytes_53215 = 4 * binop_x_53218;
    struct memblock lifted_1_zlzb_arg_mem_53230;
    
    lifted_1_zlzb_arg_mem_53230.references = NULL;
    if (cond_52369) {
        bool bounds_invalid_upwards_52373 = slt32(k2p2zq_52370, 0);
        bool eq_x_zz_52374 = 0 == k2p2zq_52370;
        bool not_p_52375 = !bounds_invalid_upwards_52373;
        bool p_and_eq_x_y_52376 = eq_x_zz_52374 && not_p_52375;
        bool dim_zzero_52377 = bounds_invalid_upwards_52373 ||
             p_and_eq_x_y_52376;
        bool both_empty_52378 = eq_x_zz_52374 && dim_zzero_52377;
        bool empty_or_match_52382 = not_p_52375 || both_empty_52378;
        bool empty_or_match_cert_52383;
        
        if (!empty_or_match_52382) {
            ctx->error = msgprintf("Error at %s:\n%s%s%s%d%s%s\n",
                                   "bfast-irreg.fut:133:1-314:84 -> bfast-irreg.fut:144:16-55 -> bfast-irreg.fut:64:10-18 -> /futlib/array.fut:61:1-62:12",
                                   "Function return value does not match shape of type ",
                                   "*", "[", k2p2zq_52370, "]",
                                   "intrinsics.i32");
            if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_53230,
                               "lifted_1_zlzb_arg_mem_53230") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53495, "out_mem_53495") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53493, "out_mem_53493") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53490, "out_mem_53490") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53487, "out_mem_53487") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53484, "out_mem_53484") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53481, "out_mem_53481") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53477, "out_mem_53477") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53473, "out_mem_53473") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53470, "out_mem_53470") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53468, "out_mem_53468") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53466, "out_mem_53466") != 0)
                return 1;
            return 1;
        }
        
        struct memblock mem_53204;
        
        mem_53204.references = NULL;
        if (memblock_alloc(ctx, &mem_53204, bytes_53200, "mem_53204"))
            return 1;
        for (int32_t i_52885 = 0; i_52885 < k2p2zq_52370; i_52885++) {
            bool cond_52387 = i_52885 == 0;
            bool cond_52388 = i_52885 == 1;
            int32_t r32_arg_52389 = sdiv32(i_52885, 2);
            int32_t x_52390 = smod32(i_52885, 2);
            float res_52391 = sitofp_i32_f32(r32_arg_52389);
            bool cond_52392 = x_52390 == 0;
            float x_52393 = 6.2831855F * res_52391;
            
            for (int32_t i_52881 = 0; i_52881 < N_52346; i_52881++) {
                int32_t x_52395 =
                        ((int32_t *) mappingindices_mem_53198.mem)[i_52881];
                float res_52396;
                
                if (cond_52387) {
                    res_52396 = 1.0F;
                } else {
                    float res_52397;
                    
                    if (cond_52388) {
                        float res_52398 = sitofp_i32_f32(x_52395);
                        
                        res_52397 = res_52398;
                    } else {
                        float res_52399 = sitofp_i32_f32(x_52395);
                        float x_52400 = x_52393 * res_52399;
                        float angle_52401 = x_52400 / freq_52352;
                        float res_52402;
                        
                        if (cond_52392) {
                            float res_52403;
                            
                            res_52403 = futrts_sin32(angle_52401);
                            res_52402 = res_52403;
                        } else {
                            float res_52404;
                            
                            res_52404 = futrts_cos32(angle_52401);
                            res_52402 = res_52404;
                        }
                        res_52397 = res_52402;
                    }
                    res_52396 = res_52397;
                }
                ((float *) mem_53204.mem)[i_52885 * N_52346 + i_52881] =
                    res_52396;
            }
        }
        if (memblock_set(ctx, &lifted_1_zlzb_arg_mem_53230, &mem_53204,
                         "mem_53204") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53204, "mem_53204") != 0)
            return 1;
    } else {
        bool bounds_invalid_upwards_52405 = slt32(k2p2zq_52370, 0);
        bool eq_x_zz_52406 = 0 == k2p2zq_52370;
        bool not_p_52407 = !bounds_invalid_upwards_52405;
        bool p_and_eq_x_y_52408 = eq_x_zz_52406 && not_p_52407;
        bool dim_zzero_52409 = bounds_invalid_upwards_52405 ||
             p_and_eq_x_y_52408;
        bool both_empty_52410 = eq_x_zz_52406 && dim_zzero_52409;
        bool empty_or_match_52414 = not_p_52407 || both_empty_52410;
        bool empty_or_match_cert_52415;
        
        if (!empty_or_match_52414) {
            ctx->error = msgprintf("Error at %s:\n%s%s%s%d%s%s\n",
                                   "bfast-irreg.fut:133:1-314:84 -> bfast-irreg.fut:145:16-55 -> bfast-irreg.fut:76:10-20 -> /futlib/array.fut:61:1-62:12",
                                   "Function return value does not match shape of type ",
                                   "*", "[", k2p2zq_52370, "]",
                                   "intrinsics.i32");
            if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_53230,
                               "lifted_1_zlzb_arg_mem_53230") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53495, "out_mem_53495") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53493, "out_mem_53493") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53490, "out_mem_53490") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53487, "out_mem_53487") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53484, "out_mem_53484") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53481, "out_mem_53481") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53477, "out_mem_53477") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53473, "out_mem_53473") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53470, "out_mem_53470") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53468, "out_mem_53468") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53466, "out_mem_53466") != 0)
                return 1;
            return 1;
        }
        
        struct memblock mem_53219;
        
        mem_53219.references = NULL;
        if (memblock_alloc(ctx, &mem_53219, bytes_53215, "mem_53219"))
            return 1;
        for (int32_t i_52893 = 0; i_52893 < k2p2zq_52370; i_52893++) {
            bool cond_52419 = i_52893 == 0;
            int32_t i_52420 = 1 + i_52893;
            int32_t r32_arg_52421 = sdiv32(i_52420, 2);
            int32_t x_52422 = smod32(i_52420, 2);
            float res_52423 = sitofp_i32_f32(r32_arg_52421);
            bool cond_52424 = x_52422 == 0;
            float x_52425 = 6.2831855F * res_52423;
            
            for (int32_t i_52889 = 0; i_52889 < N_52346; i_52889++) {
                int32_t x_52427 =
                        ((int32_t *) mappingindices_mem_53198.mem)[i_52889];
                float res_52428;
                
                if (cond_52419) {
                    res_52428 = 1.0F;
                } else {
                    float res_52429 = sitofp_i32_f32(x_52427);
                    float x_52430 = x_52425 * res_52429;
                    float angle_52431 = x_52430 / freq_52352;
                    float res_52432;
                    
                    if (cond_52424) {
                        float res_52433;
                        
                        res_52433 = futrts_sin32(angle_52431);
                        res_52432 = res_52433;
                    } else {
                        float res_52434;
                        
                        res_52434 = futrts_cos32(angle_52431);
                        res_52432 = res_52434;
                    }
                    res_52428 = res_52432;
                }
                ((float *) mem_53219.mem)[i_52893 * N_52346 + i_52889] =
                    res_52428;
            }
        }
        if (memblock_set(ctx, &lifted_1_zlzb_arg_mem_53230, &mem_53219,
                         "mem_53219") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53219, "mem_53219") != 0)
            return 1;
    }
    
    int32_t x_52436 = N_52346 * N_52346;
    int32_t y_52437 = 2 * N_52346;
    int32_t x_52438 = x_52436 + y_52437;
    int32_t x_52439 = 1 + x_52438;
    int32_t y_52440 = 1 + N_52346;
    int32_t x_52441 = sdiv32(x_52439, y_52440);
    int32_t x_52442 = x_52441 - N_52346;
    int32_t lifted_1_zlzb_arg_52443 = x_52442 - 1;
    float res_52444 = sitofp_i32_f32(lifted_1_zlzb_arg_52443);
    int64_t binop_x_53232 = sext_i32_i64(N_52346);
    int64_t binop_y_53233 = sext_i32_i64(k2p2zq_52370);
    int64_t binop_x_53234 = binop_x_53232 * binop_y_53233;
    int64_t bytes_53231 = 4 * binop_x_53234;
    struct memblock mem_53235;
    
    mem_53235.references = NULL;
    if (memblock_alloc(ctx, &mem_53235, bytes_53231, "mem_53235"))
        return 1;
    for (int32_t i_52901 = 0; i_52901 < N_52346; i_52901++) {
        for (int32_t i_52897 = 0; i_52897 < k2p2zq_52370; i_52897++) {
            float x_52449 =
                  ((float *) lifted_1_zlzb_arg_mem_53230.mem)[i_52897 *
                                                              N_52346 +
                                                              i_52901];
            float res_52450 = res_52444 + x_52449;
            
            ((float *) mem_53235.mem)[i_52901 * k2p2zq_52370 + i_52897] =
                res_52450;
        }
    }
    
    int32_t m_52453 = k2p2zq_52370 - 1;
    bool empty_slice_52460 = n_52351 == 0;
    int32_t m_52461 = n_52351 - 1;
    bool zzero_leq_i_p_m_t_s_52462 = sle32(0, m_52461);
    bool i_p_m_t_s_leq_w_52463 = slt32(m_52461, N_52346);
    bool i_lte_j_52464 = sle32(0, n_52351);
    bool y_52465 = zzero_leq_i_p_m_t_s_52462 && i_p_m_t_s_leq_w_52463;
    bool y_52466 = i_lte_j_52464 && y_52465;
    bool ok_or_empty_52467 = empty_slice_52460 || y_52466;
    bool index_certs_52469;
    
    if (!ok_or_empty_52467) {
        ctx->error = msgprintf("Error at %s:\n%s%d%s%s%s%d%s%d%s%d%s\n",
                               "bfast-irreg.fut:133:1-314:84 -> bfast-irreg.fut:154:15-21",
                               "Index [", 0, ", ", "", ":", n_52351,
                               "] out of bounds for array of shape [",
                               k2p2zq_52370, "][", N_52346, "].");
        if (memblock_unref(ctx, &mem_53235, "mem_53235") != 0)
            return 1;
        if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_53230,
                           "lifted_1_zlzb_arg_mem_53230") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53495, "out_mem_53495") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53493, "out_mem_53493") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53490, "out_mem_53490") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53487, "out_mem_53487") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53484, "out_mem_53484") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53481, "out_mem_53481") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53477, "out_mem_53477") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53473, "out_mem_53473") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53470, "out_mem_53470") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53468, "out_mem_53468") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53466, "out_mem_53466") != 0)
            return 1;
        return 1;
    }
    
    bool index_certs_52471;
    
    if (!ok_or_empty_52467) {
        ctx->error = msgprintf("Error at %s:\n%s%s%s%d%s%d%s%d%s%d%s\n",
                               "bfast-irreg.fut:133:1-314:84 -> bfast-irreg.fut:155:15-22",
                               "Index [", "", ":", n_52351, ", ", 0,
                               "] out of bounds for array of shape [", N_52346,
                               "][", k2p2zq_52370, "].");
        if (memblock_unref(ctx, &mem_53235, "mem_53235") != 0)
            return 1;
        if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_53230,
                           "lifted_1_zlzb_arg_mem_53230") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53495, "out_mem_53495") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53493, "out_mem_53493") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53490, "out_mem_53490") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53487, "out_mem_53487") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53484, "out_mem_53484") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53481, "out_mem_53481") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53477, "out_mem_53477") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53473, "out_mem_53473") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53470, "out_mem_53470") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53468, "out_mem_53468") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53466, "out_mem_53466") != 0)
            return 1;
        return 1;
    }
    
    bool index_certs_52482;
    
    if (!ok_or_empty_52467) {
        ctx->error = msgprintf("Error at %s:\n%s%d%s%s%s%d%s%d%s%d%s\n",
                               "bfast-irreg.fut:133:1-314:84 -> bfast-irreg.fut:156:15-26",
                               "Index [", 0, ", ", "", ":", n_52351,
                               "] out of bounds for array of shape [", m_52347,
                               "][", N_52346, "].");
        if (memblock_unref(ctx, &mem_53235, "mem_53235") != 0)
            return 1;
        if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_53230,
                           "lifted_1_zlzb_arg_mem_53230") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53495, "out_mem_53495") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53493, "out_mem_53493") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53490, "out_mem_53490") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53487, "out_mem_53487") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53484, "out_mem_53484") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53481, "out_mem_53481") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53477, "out_mem_53477") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53473, "out_mem_53473") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53470, "out_mem_53470") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53468, "out_mem_53468") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53466, "out_mem_53466") != 0)
            return 1;
        return 1;
    }
    
    int64_t binop_x_53247 = sext_i32_i64(m_52347);
    int64_t binop_x_53249 = binop_y_53233 * binop_x_53247;
    int64_t binop_x_53251 = binop_y_53233 * binop_x_53249;
    int64_t bytes_53246 = 4 * binop_x_53251;
    struct memblock mem_53252;
    
    mem_53252.references = NULL;
    if (memblock_alloc(ctx, &mem_53252, bytes_53246, "mem_53252"))
        return 1;
    for (int32_t i_52915 = 0; i_52915 < m_52347; i_52915++) {
        for (int32_t i_52911 = 0; i_52911 < k2p2zq_52370; i_52911++) {
            for (int32_t i_52907 = 0; i_52907 < k2p2zq_52370; i_52907++) {
                float res_52491;
                float redout_52903 = 0.0F;
                
                for (int32_t i_52904 = 0; i_52904 < n_52351; i_52904++) {
                    float x_52495 = ((float *) images_mem_53199.mem)[i_52915 *
                                                                     N_52348 +
                                                                     i_52904];
                    float x_52496 =
                          ((float *) lifted_1_zlzb_arg_mem_53230.mem)[i_52911 *
                                                                      N_52346 +
                                                                      i_52904];
                    float x_52497 = ((float *) mem_53235.mem)[i_52904 *
                                                              k2p2zq_52370 +
                                                              i_52907];
                    float x_52498 = x_52496 * x_52497;
                    bool res_52499;
                    
                    res_52499 = futrts_isnan32(x_52495);
                    
                    float y_52500;
                    
                    if (res_52499) {
                        y_52500 = 0.0F;
                    } else {
                        y_52500 = 1.0F;
                    }
                    
                    float res_52501 = x_52498 * y_52500;
                    float res_52494 = res_52501 + redout_52903;
                    float redout_tmp_53507 = res_52494;
                    
                    redout_52903 = redout_tmp_53507;
                }
                res_52491 = redout_52903;
                ((float *) mem_53252.mem)[i_52915 * (k2p2zq_52370 *
                                                     k2p2zq_52370) + i_52911 *
                                          k2p2zq_52370 + i_52907] = res_52491;
            }
        }
    }
    
    int32_t j_52503 = 2 * k2p2zq_52370;
    int32_t j_m_i_52504 = j_52503 - k2p2zq_52370;
    int32_t nm_52507 = k2p2zq_52370 * j_52503;
    bool empty_slice_52520 = j_m_i_52504 == 0;
    int32_t m_52521 = j_m_i_52504 - 1;
    int32_t i_p_m_t_s_52522 = k2p2zq_52370 + m_52521;
    bool zzero_leq_i_p_m_t_s_52523 = sle32(0, i_p_m_t_s_52522);
    bool ok_or_empty_52530 = empty_slice_52520 || zzero_leq_i_p_m_t_s_52523;
    bool index_certs_52532;
    
    if (!ok_or_empty_52530) {
        ctx->error = msgprintf("Error at %s:\n%s%d%s%d%s%d%s%d%s%d%s%d%s\n",
                               "bfast-irreg.fut:133:1-314:84 -> bfast-irreg.fut:168:14-29 -> bfast-irreg.fut:109:8-37",
                               "Index [", 0, ":", k2p2zq_52370, ", ",
                               k2p2zq_52370, ":", j_52503,
                               "] out of bounds for array of shape [",
                               k2p2zq_52370, "][", j_52503, "].");
        if (memblock_unref(ctx, &mem_53252, "mem_53252") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53235, "mem_53235") != 0)
            return 1;
        if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_53230,
                           "lifted_1_zlzb_arg_mem_53230") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53495, "out_mem_53495") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53493, "out_mem_53493") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53490, "out_mem_53490") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53487, "out_mem_53487") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53484, "out_mem_53484") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53481, "out_mem_53481") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53477, "out_mem_53477") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53473, "out_mem_53473") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53470, "out_mem_53470") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53468, "out_mem_53468") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53466, "out_mem_53466") != 0)
            return 1;
        return 1;
    }
    
    int64_t binop_y_53284 = sext_i32_i64(j_m_i_52504);
    int64_t binop_x_53285 = binop_x_53249 * binop_y_53284;
    int64_t bytes_53280 = 4 * binop_x_53285;
    struct memblock mem_53286;
    
    mem_53286.references = NULL;
    if (memblock_alloc(ctx, &mem_53286, bytes_53280, "mem_53286"))
        return 1;
    
    int64_t binop_x_53289 = sext_i32_i64(nm_52507);
    int64_t bytes_53288 = 4 * binop_x_53289;
    struct memblock mem_53290;
    
    mem_53290.references = NULL;
    if (memblock_alloc(ctx, &mem_53290, bytes_53288, "mem_53290"))
        return 1;
    
    struct memblock mem_53296;
    
    mem_53296.references = NULL;
    if (memblock_alloc(ctx, &mem_53296, bytes_53288, "mem_53296"))
        return 1;
    for (int32_t i_52937 = 0; i_52937 < m_52347; i_52937++) {
        for (int32_t i_52919 = 0; i_52919 < nm_52507; i_52919++) {
            int32_t res_52537 = sdiv32(i_52919, j_52503);
            int32_t res_52538 = smod32(i_52919, j_52503);
            bool cond_52539 = slt32(res_52538, k2p2zq_52370);
            float res_52540;
            
            if (cond_52539) {
                float res_52541 = ((float *) mem_53252.mem)[i_52937 *
                                                            (k2p2zq_52370 *
                                                             k2p2zq_52370) +
                                                            res_52537 *
                                                            k2p2zq_52370 +
                                                            res_52538];
                
                res_52540 = res_52541;
            } else {
                int32_t y_52542 = k2p2zq_52370 + res_52537;
                bool cond_52543 = res_52538 == y_52542;
                float res_52544;
                
                if (cond_52543) {
                    res_52544 = 1.0F;
                } else {
                    res_52544 = 0.0F;
                }
                res_52540 = res_52544;
            }
            ((float *) mem_53290.mem)[i_52919] = res_52540;
        }
        for (int32_t i_52547 = 0; i_52547 < k2p2zq_52370; i_52547++) {
            float v1_52552 = ((float *) mem_53290.mem)[i_52547];
            bool cond_52553 = v1_52552 == 0.0F;
            
            for (int32_t i_52923 = 0; i_52923 < nm_52507; i_52923++) {
                int32_t res_52556 = sdiv32(i_52923, j_52503);
                int32_t res_52557 = smod32(i_52923, j_52503);
                float res_52558;
                
                if (cond_52553) {
                    int32_t x_52559 = j_52503 * res_52556;
                    int32_t i_52560 = res_52557 + x_52559;
                    float res_52561 = ((float *) mem_53290.mem)[i_52560];
                    
                    res_52558 = res_52561;
                } else {
                    float x_52562 = ((float *) mem_53290.mem)[res_52557];
                    float x_52563 = x_52562 / v1_52552;
                    bool cond_52564 = slt32(res_52556, m_52453);
                    float res_52565;
                    
                    if (cond_52564) {
                        int32_t x_52566 = 1 + res_52556;
                        int32_t x_52567 = j_52503 * x_52566;
                        int32_t i_52568 = res_52557 + x_52567;
                        float x_52569 = ((float *) mem_53290.mem)[i_52568];
                        int32_t i_52570 = i_52547 + x_52567;
                        float x_52571 = ((float *) mem_53290.mem)[i_52570];
                        float y_52572 = x_52563 * x_52571;
                        float res_52573 = x_52569 - y_52572;
                        
                        res_52565 = res_52573;
                    } else {
                        res_52565 = x_52563;
                    }
                    res_52558 = res_52565;
                }
                ((float *) mem_53296.mem)[i_52923] = res_52558;
            }
            for (int32_t write_iter_52925 = 0; write_iter_52925 < nm_52507;
                 write_iter_52925++) {
                bool less_than_zzero_52929 = slt32(write_iter_52925, 0);
                bool greater_than_sizze_52930 = sle32(nm_52507,
                                                      write_iter_52925);
                bool outside_bounds_dim_52931 = less_than_zzero_52929 ||
                     greater_than_sizze_52930;
                
                if (!outside_bounds_dim_52931) {
                    memmove(mem_53290.mem + write_iter_52925 * 4,
                            mem_53296.mem + write_iter_52925 * 4,
                            sizeof(float));
                }
            }
        }
        for (int32_t i_53513 = 0; i_53513 < k2p2zq_52370; i_53513++) {
            for (int32_t i_53514 = 0; i_53514 < j_m_i_52504; i_53514++) {
                ((float *) mem_53286.mem)[i_52937 * (j_m_i_52504 *
                                                     k2p2zq_52370) + (i_53513 *
                                                                      j_m_i_52504 +
                                                                      i_53514)] =
                    ((float *) mem_53290.mem)[k2p2zq_52370 + (i_53513 *
                                                              j_52503 +
                                                              i_53514)];
            }
        }
    }
    if (memblock_unref(ctx, &mem_53290, "mem_53290") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53296, "mem_53296") != 0)
        return 1;
    
    int64_t bytes_53303 = 4 * binop_x_53249;
    struct memblock mem_53307;
    
    mem_53307.references = NULL;
    if (memblock_alloc(ctx, &mem_53307, bytes_53303, "mem_53307"))
        return 1;
    for (int32_t i_52947 = 0; i_52947 < m_52347; i_52947++) {
        for (int32_t i_52943 = 0; i_52943 < k2p2zq_52370; i_52943++) {
            float res_52584;
            float redout_52939 = 0.0F;
            
            for (int32_t i_52940 = 0; i_52940 < n_52351; i_52940++) {
                float x_52588 =
                      ((float *) lifted_1_zlzb_arg_mem_53230.mem)[i_52943 *
                                                                  N_52346 +
                                                                  i_52940];
                float x_52589 = ((float *) images_mem_53199.mem)[i_52947 *
                                                                 N_52348 +
                                                                 i_52940];
                bool res_52590;
                
                res_52590 = futrts_isnan32(x_52589);
                
                float res_52591;
                
                if (res_52590) {
                    res_52591 = 0.0F;
                } else {
                    float res_52592 = x_52588 * x_52589;
                    
                    res_52591 = res_52592;
                }
                
                float res_52587 = res_52591 + redout_52939;
                float redout_tmp_53517 = res_52587;
                
                redout_52939 = redout_tmp_53517;
            }
            res_52584 = redout_52939;
            ((float *) mem_53307.mem)[i_52947 * k2p2zq_52370 + i_52943] =
                res_52584;
        }
    }
    if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_53230,
                       "lifted_1_zlzb_arg_mem_53230") != 0)
        return 1;
    
    struct memblock mem_53322;
    
    mem_53322.references = NULL;
    if (memblock_alloc(ctx, &mem_53322, bytes_53303, "mem_53322"))
        return 1;
    for (int32_t i_52957 = 0; i_52957 < m_52347; i_52957++) {
        for (int32_t i_52953 = 0; i_52953 < k2p2zq_52370; i_52953++) {
            float res_52604;
            float redout_52949 = 0.0F;
            
            for (int32_t i_52950 = 0; i_52950 < j_m_i_52504; i_52950++) {
                float x_52608 = ((float *) mem_53307.mem)[i_52957 *
                                                          k2p2zq_52370 +
                                                          i_52950];
                float x_52609 = ((float *) mem_53286.mem)[i_52957 *
                                                          (j_m_i_52504 *
                                                           k2p2zq_52370) +
                                                          i_52953 *
                                                          j_m_i_52504 +
                                                          i_52950];
                float res_52610 = x_52608 * x_52609;
                float res_52607 = res_52610 + redout_52949;
                float redout_tmp_53520 = res_52607;
                
                redout_52949 = redout_tmp_53520;
            }
            res_52604 = redout_52949;
            ((float *) mem_53322.mem)[i_52957 * k2p2zq_52370 + i_52953] =
                res_52604;
        }
    }
    
    int64_t binop_x_53336 = binop_x_53232 * binop_x_53247;
    int64_t bytes_53333 = 4 * binop_x_53336;
    struct memblock mem_53337;
    
    mem_53337.references = NULL;
    if (memblock_alloc(ctx, &mem_53337, bytes_53333, "mem_53337"))
        return 1;
    for (int32_t i_52967 = 0; i_52967 < m_52347; i_52967++) {
        for (int32_t i_52963 = 0; i_52963 < N_52346; i_52963++) {
            float res_52616;
            float redout_52959 = 0.0F;
            
            for (int32_t i_52960 = 0; i_52960 < k2p2zq_52370; i_52960++) {
                float x_52620 = ((float *) mem_53322.mem)[i_52967 *
                                                          k2p2zq_52370 +
                                                          i_52960];
                float x_52621 = ((float *) mem_53235.mem)[i_52963 *
                                                          k2p2zq_52370 +
                                                          i_52960];
                float res_52622 = x_52620 * x_52621;
                float res_52619 = res_52622 + redout_52959;
                float redout_tmp_53523 = res_52619;
                
                redout_52959 = redout_tmp_53523;
            }
            res_52616 = redout_52959;
            ((float *) mem_53337.mem)[i_52967 * N_52346 + i_52963] = res_52616;
        }
    }
    
    int32_t i_52624 = N_52346 - 1;
    bool x_52625 = sle32(0, i_52624);
    bool index_certs_52628;
    
    if (!x_52625) {
        ctx->error = msgprintf("Error at %s:\n%s%d%s%d%s\n",
                               "bfast-irreg.fut:133:1-314:84 -> bfast-irreg.fut:189:5-198:25 -> /futlib/soacs.fut:51:3-37 -> /futlib/soacs.fut:51:19-23 -> bfast-irreg.fut:194:30-91 -> bfast-irreg.fut:37:13-20 -> /futlib/array.fut:18:29-34",
                               "Index [", i_52624,
                               "] out of bounds for array of shape [", N_52346,
                               "].");
        if (memblock_unref(ctx, &mem_53337, "mem_53337") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53322, "mem_53322") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53307, "mem_53307") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53296, "mem_53296") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53290, "mem_53290") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53286, "mem_53286") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53252, "mem_53252") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53235, "mem_53235") != 0)
            return 1;
        if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_53230,
                           "lifted_1_zlzb_arg_mem_53230") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53495, "out_mem_53495") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53493, "out_mem_53493") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53490, "out_mem_53490") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53487, "out_mem_53487") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53484, "out_mem_53484") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53481, "out_mem_53481") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53477, "out_mem_53477") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53473, "out_mem_53473") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53470, "out_mem_53470") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53468, "out_mem_53468") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53466, "out_mem_53466") != 0)
            return 1;
        return 1;
    }
    
    int64_t bytes_53348 = 4 * binop_x_53247;
    struct memblock mem_53350;
    
    mem_53350.references = NULL;
    if (memblock_alloc(ctx, &mem_53350, bytes_53348, "mem_53350"))
        return 1;
    
    struct memblock mem_53355;
    
    mem_53355.references = NULL;
    if (memblock_alloc(ctx, &mem_53355, bytes_53333, "mem_53355"))
        return 1;
    
    struct memblock mem_53360;
    
    mem_53360.references = NULL;
    if (memblock_alloc(ctx, &mem_53360, bytes_53333, "mem_53360"))
        return 1;
    
    int64_t bytes_53364 = 4 * binop_x_53232;
    struct memblock mem_53366;
    
    mem_53366.references = NULL;
    if (memblock_alloc(ctx, &mem_53366, bytes_53364, "mem_53366"))
        return 1;
    
    struct memblock mem_53369;
    
    mem_53369.references = NULL;
    if (memblock_alloc(ctx, &mem_53369, bytes_53364, "mem_53369"))
        return 1;
    
    struct memblock mem_53376;
    
    mem_53376.references = NULL;
    if (memblock_alloc(ctx, &mem_53376, bytes_53364, "mem_53376"))
        return 1;
    for (int32_t i_53524 = 0; i_53524 < N_52346; i_53524++) {
        ((float *) mem_53376.mem)[i_53524] = NAN;
    }
    
    struct memblock mem_53379;
    
    mem_53379.references = NULL;
    if (memblock_alloc(ctx, &mem_53379, bytes_53364, "mem_53379"))
        return 1;
    for (int32_t i_53525 = 0; i_53525 < N_52346; i_53525++) {
        ((int32_t *) mem_53379.mem)[i_53525] = 0;
    }
    
    struct memblock mem_53384;
    
    mem_53384.references = NULL;
    if (memblock_alloc(ctx, &mem_53384, bytes_53364, "mem_53384"))
        return 1;
    
    struct memblock mem_53394;
    
    mem_53394.references = NULL;
    if (memblock_alloc(ctx, &mem_53394, bytes_53364, "mem_53394"))
        return 1;
    for (int32_t i_53002 = 0; i_53002 < m_52347; i_53002++) {
        int32_t discard_52977;
        int32_t scanacc_52971 = 0;
        
        for (int32_t i_52974 = 0; i_52974 < N_52346; i_52974++) {
            float x_52658 = ((float *) images_mem_53199.mem)[i_53002 * N_52348 +
                                                             i_52974];
            float x_52659 = ((float *) mem_53337.mem)[i_53002 * N_52346 +
                                                      i_52974];
            bool res_52660;
            
            res_52660 = futrts_isnan32(x_52658);
            
            bool cond_52661 = !res_52660;
            float res_52662;
            
            if (cond_52661) {
                float res_52663 = x_52658 - x_52659;
                
                res_52662 = res_52663;
            } else {
                res_52662 = NAN;
            }
            
            bool res_52664;
            
            res_52664 = futrts_isnan32(res_52662);
            
            bool res_52665 = !res_52664;
            int32_t res_52666;
            
            if (res_52665) {
                res_52666 = 1;
            } else {
                res_52666 = 0;
            }
            
            int32_t res_52657 = res_52666 + scanacc_52971;
            
            ((int32_t *) mem_53366.mem)[i_52974] = res_52657;
            ((float *) mem_53369.mem)[i_52974] = res_52662;
            
            int32_t scanacc_tmp_53529 = res_52657;
            
            scanacc_52971 = scanacc_tmp_53529;
        }
        discard_52977 = scanacc_52971;
        memmove(mem_53355.mem + i_53002 * N_52346 * 4, mem_53376.mem + 0,
                N_52346 * sizeof(float));
        memmove(mem_53360.mem + i_53002 * N_52346 * 4, mem_53379.mem + 0,
                N_52346 * sizeof(int32_t));
        for (int32_t write_iter_52978 = 0; write_iter_52978 < N_52346;
             write_iter_52978++) {
            float write_iv_52981 = ((float *) mem_53369.mem)[write_iter_52978];
            int32_t write_iv_52982 =
                    ((int32_t *) mem_53366.mem)[write_iter_52978];
            bool res_52677;
            
            res_52677 = futrts_isnan32(write_iv_52981);
            
            bool res_52678 = !res_52677;
            int32_t res_52679;
            
            if (res_52678) {
                int32_t res_52680 = write_iv_52982 - 1;
                
                res_52679 = res_52680;
            } else {
                res_52679 = -1;
            }
            
            bool less_than_zzero_52984 = slt32(res_52679, 0);
            bool greater_than_sizze_52985 = sle32(N_52346, res_52679);
            bool outside_bounds_dim_52986 = less_than_zzero_52984 ||
                 greater_than_sizze_52985;
            
            memmove(mem_53384.mem + 0, mem_53360.mem + i_53002 * N_52346 * 4,
                    N_52346 * sizeof(int32_t));
            
            struct memblock write_out_mem_53391;
            
            write_out_mem_53391.references = NULL;
            if (outside_bounds_dim_52986) {
                if (memblock_set(ctx, &write_out_mem_53391, &mem_53384,
                                 "mem_53384") != 0)
                    return 1;
            } else {
                struct memblock mem_53387;
                
                mem_53387.references = NULL;
                if (memblock_alloc(ctx, &mem_53387, 4, "mem_53387"))
                    return 1;
                
                int32_t x_53535;
                
                for (int32_t i_53534 = 0; i_53534 < 1; i_53534++) {
                    x_53535 = write_iter_52978 + sext_i32_i32(i_53534);
                    ((int32_t *) mem_53387.mem)[i_53534] = x_53535;
                }
                
                struct memblock mem_53390;
                
                mem_53390.references = NULL;
                if (memblock_alloc(ctx, &mem_53390, bytes_53364, "mem_53390"))
                    return 1;
                memmove(mem_53390.mem + 0, mem_53360.mem + i_53002 * N_52346 *
                        4, N_52346 * sizeof(int32_t));
                memmove(mem_53390.mem + res_52679 * 4, mem_53387.mem + 0,
                        sizeof(int32_t));
                if (memblock_unref(ctx, &mem_53387, "mem_53387") != 0)
                    return 1;
                if (memblock_set(ctx, &write_out_mem_53391, &mem_53390,
                                 "mem_53390") != 0)
                    return 1;
                if (memblock_unref(ctx, &mem_53390, "mem_53390") != 0)
                    return 1;
                if (memblock_unref(ctx, &mem_53387, "mem_53387") != 0)
                    return 1;
            }
            memmove(mem_53360.mem + i_53002 * N_52346 * 4,
                    write_out_mem_53391.mem + 0, N_52346 * sizeof(int32_t));
            if (memblock_unref(ctx, &write_out_mem_53391,
                               "write_out_mem_53391") != 0)
                return 1;
            memmove(mem_53394.mem + 0, mem_53355.mem + i_53002 * N_52346 * 4,
                    N_52346 * sizeof(float));
            
            struct memblock write_out_mem_53398;
            
            write_out_mem_53398.references = NULL;
            if (outside_bounds_dim_52986) {
                if (memblock_set(ctx, &write_out_mem_53398, &mem_53394,
                                 "mem_53394") != 0)
                    return 1;
            } else {
                struct memblock mem_53397;
                
                mem_53397.references = NULL;
                if (memblock_alloc(ctx, &mem_53397, bytes_53364, "mem_53397"))
                    return 1;
                memmove(mem_53397.mem + 0, mem_53355.mem + i_53002 * N_52346 *
                        4, N_52346 * sizeof(float));
                memmove(mem_53397.mem + res_52679 * 4, mem_53369.mem +
                        write_iter_52978 * 4, sizeof(float));
                if (memblock_set(ctx, &write_out_mem_53398, &mem_53397,
                                 "mem_53397") != 0)
                    return 1;
                if (memblock_unref(ctx, &mem_53397, "mem_53397") != 0)
                    return 1;
            }
            memmove(mem_53355.mem + i_53002 * N_52346 * 4,
                    write_out_mem_53398.mem + 0, N_52346 * sizeof(float));
            if (memblock_unref(ctx, &write_out_mem_53398,
                               "write_out_mem_53398") != 0)
                return 1;
            if (memblock_unref(ctx, &write_out_mem_53398,
                               "write_out_mem_53398") != 0)
                return 1;
            if (memblock_unref(ctx, &write_out_mem_53391,
                               "write_out_mem_53391") != 0)
                return 1;
        }
        memmove(mem_53350.mem + i_53002 * 4, mem_53366.mem + i_52624 * 4,
                sizeof(int32_t));
    }
    if (memblock_unref(ctx, &mem_53366, "mem_53366") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53369, "mem_53369") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53376, "mem_53376") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53379, "mem_53379") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53384, "mem_53384") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53394, "mem_53394") != 0)
        return 1;
    
    struct memblock mem_53406;
    
    mem_53406.references = NULL;
    if (memblock_alloc(ctx, &mem_53406, bytes_53348, "mem_53406"))
        return 1;
    
    struct memblock mem_53409;
    
    mem_53409.references = NULL;
    if (memblock_alloc(ctx, &mem_53409, bytes_53348, "mem_53409"))
        return 1;
    
    struct memblock mem_53412;
    
    mem_53412.references = NULL;
    if (memblock_alloc(ctx, &mem_53412, bytes_53348, "mem_53412"))
        return 1;
    for (int32_t i_53016 = 0; i_53016 < m_52347; i_53016++) {
        int32_t res_52701;
        int32_t redout_53006 = 0;
        
        for (int32_t i_53007 = 0; i_53007 < n_52351; i_53007++) {
            float x_52705 = ((float *) images_mem_53199.mem)[i_53016 * N_52348 +
                                                             i_53007];
            bool res_52706;
            
            res_52706 = futrts_isnan32(x_52705);
            
            bool cond_52707 = !res_52706;
            int32_t res_52708;
            
            if (cond_52707) {
                res_52708 = 1;
            } else {
                res_52708 = 0;
            }
            
            int32_t res_52704 = res_52708 + redout_53006;
            int32_t redout_tmp_53539 = res_52704;
            
            redout_53006 = redout_tmp_53539;
        }
        res_52701 = redout_53006;
        
        float res_52709;
        float redout_53008 = 0.0F;
        
        for (int32_t i_53009 = 0; i_53009 < n_52351; i_53009++) {
            float y_error_elem_52714 = ((float *) mem_53355.mem)[i_53016 *
                                                                 N_52346 +
                                                                 i_53009];
            bool cond_52715 = slt32(i_53009, res_52701);
            float res_52716;
            
            if (cond_52715) {
                res_52716 = y_error_elem_52714;
            } else {
                res_52716 = 0.0F;
            }
            
            float res_52717 = res_52716 * res_52716;
            float res_52712 = res_52717 + redout_53008;
            float redout_tmp_53540 = res_52712;
            
            redout_53008 = redout_tmp_53540;
        }
        res_52709 = redout_53008;
        
        int32_t r32_arg_52718 = res_52701 - k2p2_52368;
        float res_52719 = sitofp_i32_f32(r32_arg_52718);
        float sqrt_arg_52720 = res_52709 / res_52719;
        float res_52721;
        
        res_52721 = futrts_sqrt32(sqrt_arg_52720);
        
        float res_52722 = sitofp_i32_f32(res_52701);
        float t32_arg_52723 = hfrac_52353 * res_52722;
        int32_t res_52724 = fptosi_f32_i32(t32_arg_52723);
        
        ((int32_t *) mem_53406.mem)[i_53016] = res_52724;
        ((int32_t *) mem_53409.mem)[i_53016] = res_52701;
        ((float *) mem_53412.mem)[i_53016] = res_52721;
    }
    
    int32_t res_52728;
    int32_t redout_53020 = 0;
    
    for (int32_t i_53021 = 0; i_53021 < m_52347; i_53021++) {
        int32_t x_52732 = ((int32_t *) mem_53406.mem)[i_53021];
        int32_t res_52731 = smax32(x_52732, redout_53020);
        int32_t redout_tmp_53541 = res_52731;
        
        redout_53020 = redout_tmp_53541;
    }
    res_52728 = redout_53020;
    
    struct memblock mem_53421;
    
    mem_53421.references = NULL;
    if (memblock_alloc(ctx, &mem_53421, bytes_53348, "mem_53421"))
        return 1;
    for (int32_t i_53026 = 0; i_53026 < m_52347; i_53026++) {
        int32_t x_52736 = ((int32_t *) mem_53409.mem)[i_53026];
        int32_t x_52737 = ((int32_t *) mem_53406.mem)[i_53026];
        float res_52738;
        float redout_53022 = 0.0F;
        
        for (int32_t i_53023 = 0; i_53023 < res_52728; i_53023++) {
            bool cond_52743 = slt32(i_53023, x_52737);
            float res_52744;
            
            if (cond_52743) {
                int32_t x_52745 = x_52736 + i_53023;
                int32_t x_52746 = x_52745 - x_52737;
                int32_t i_52747 = 1 + x_52746;
                float res_52748 = ((float *) mem_53355.mem)[i_53026 * N_52346 +
                                                            i_52747];
                
                res_52744 = res_52748;
            } else {
                res_52744 = 0.0F;
            }
            
            float res_52741 = res_52744 + redout_53022;
            float redout_tmp_53543 = res_52741;
            
            redout_53022 = redout_tmp_53543;
        }
        res_52738 = redout_53022;
        ((float *) mem_53421.mem)[i_53026] = res_52738;
    }
    
    int32_t iota_arg_52750 = N_52346 - n_52351;
    bool bounds_invalid_upwards_52751 = slt32(iota_arg_52750, 0);
    bool eq_x_zz_52752 = 0 == iota_arg_52750;
    bool not_p_52753 = !bounds_invalid_upwards_52751;
    bool p_and_eq_x_y_52754 = eq_x_zz_52752 && not_p_52753;
    bool dim_zzero_52755 = bounds_invalid_upwards_52751 || p_and_eq_x_y_52754;
    bool both_empty_52756 = eq_x_zz_52752 && dim_zzero_52755;
    bool eq_x_y_52757 = iota_arg_52750 == 0;
    bool p_and_eq_x_y_52758 = bounds_invalid_upwards_52751 && eq_x_y_52757;
    bool dim_match_52759 = not_p_52753 || p_and_eq_x_y_52758;
    bool empty_or_match_52760 = both_empty_52756 || dim_match_52759;
    bool empty_or_match_cert_52761;
    
    if (!empty_or_match_52760) {
        ctx->error = msgprintf("Error at %s:\n%s%s%s%d%s%s\n",
                               "bfast-irreg.fut:133:1-314:84 -> bfast-irreg.fut:242:22-31 -> /futlib/array.fut:61:1-62:12",
                               "Function return value does not match shape of type ",
                               "*", "[", iota_arg_52750, "]", "intrinsics.i32");
        if (memblock_unref(ctx, &mem_53421, "mem_53421") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53412, "mem_53412") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53409, "mem_53409") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53406, "mem_53406") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53394, "mem_53394") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53384, "mem_53384") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53379, "mem_53379") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53376, "mem_53376") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53369, "mem_53369") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53366, "mem_53366") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53360, "mem_53360") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53355, "mem_53355") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53350, "mem_53350") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53337, "mem_53337") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53322, "mem_53322") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53307, "mem_53307") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53296, "mem_53296") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53290, "mem_53290") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53286, "mem_53286") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53252, "mem_53252") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53235, "mem_53235") != 0)
            return 1;
        if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_53230,
                           "lifted_1_zlzb_arg_mem_53230") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53495, "out_mem_53495") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53493, "out_mem_53493") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53490, "out_mem_53490") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53487, "out_mem_53487") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53484, "out_mem_53484") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53481, "out_mem_53481") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53477, "out_mem_53477") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53473, "out_mem_53473") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53470, "out_mem_53470") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53468, "out_mem_53468") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53466, "out_mem_53466") != 0)
            return 1;
        return 1;
    }
    
    int32_t x_52763 = 1 + n_52351;
    bool index_certs_52764;
    
    if (!x_52625) {
        ctx->error = msgprintf("Error at %s:\n%s%d%s%d%s\n",
                               "bfast-irreg.fut:133:1-314:84 -> bfast-irreg.fut:238:15-242:32 -> bfast-irreg.fut:240:63-81",
                               "Index [", i_52624,
                               "] out of bounds for array of shape [", N_52346,
                               "].");
        if (memblock_unref(ctx, &mem_53421, "mem_53421") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53412, "mem_53412") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53409, "mem_53409") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53406, "mem_53406") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53394, "mem_53394") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53384, "mem_53384") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53379, "mem_53379") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53376, "mem_53376") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53369, "mem_53369") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53366, "mem_53366") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53360, "mem_53360") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53355, "mem_53355") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53350, "mem_53350") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53337, "mem_53337") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53322, "mem_53322") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53307, "mem_53307") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53296, "mem_53296") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53290, "mem_53290") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53286, "mem_53286") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53252, "mem_53252") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53235, "mem_53235") != 0)
            return 1;
        if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_53230,
                           "lifted_1_zlzb_arg_mem_53230") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53495, "out_mem_53495") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53493, "out_mem_53493") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53490, "out_mem_53490") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53487, "out_mem_53487") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53484, "out_mem_53484") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53481, "out_mem_53481") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53477, "out_mem_53477") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53473, "out_mem_53473") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53470, "out_mem_53470") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53468, "out_mem_53468") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53466, "out_mem_53466") != 0)
            return 1;
        return 1;
    }
    
    int32_t r32_arg_52765 = ((int32_t *) mappingindices_mem_53198.mem)[i_52624];
    float res_52766 = sitofp_i32_f32(r32_arg_52765);
    int64_t binop_x_53425 = sext_i32_i64(iota_arg_52750);
    int64_t bytes_53424 = 4 * binop_x_53425;
    struct memblock mem_53426;
    
    mem_53426.references = NULL;
    if (memblock_alloc(ctx, &mem_53426, bytes_53424, "mem_53426"))
        return 1;
    for (int32_t i_53030 = 0; i_53030 < iota_arg_52750; i_53030++) {
        int32_t t_52769 = x_52763 + i_53030;
        int32_t i_52770 = t_52769 - 1;
        int32_t time_52771 =
                ((int32_t *) mappingindices_mem_53198.mem)[i_52770];
        float res_52772 = sitofp_i32_f32(time_52771);
        float logplus_arg_52773 = res_52772 / res_52766;
        bool cond_52774 = 2.7182817F < logplus_arg_52773;
        float res_52775;
        
        if (cond_52774) {
            float res_52776;
            
            res_52776 = futrts_log32(logplus_arg_52773);
            res_52775 = res_52776;
        } else {
            res_52775 = 1.0F;
        }
        
        float res_52777;
        
        res_52777 = futrts_sqrt32(res_52775);
        
        float res_52778 = lam_52354 * res_52777;
        
        ((float *) mem_53426.mem)[i_53030] = res_52778;
    }
    
    struct memblock mem_53431;
    
    mem_53431.references = NULL;
    if (memblock_alloc(ctx, &mem_53431, bytes_53348, "mem_53431"))
        return 1;
    
    struct memblock mem_53434;
    
    mem_53434.references = NULL;
    if (memblock_alloc(ctx, &mem_53434, bytes_53348, "mem_53434"))
        return 1;
    
    struct memblock mem_53439;
    
    mem_53439.references = NULL;
    if (memblock_alloc(ctx, &mem_53439, bytes_53424, "mem_53439"))
        return 1;
    for (int32_t i_53046 = 0; i_53046 < m_52347; i_53046++) {
        int32_t x_52781 = ((int32_t *) mem_53350.mem)[i_53046];
        int32_t x_52782 = ((int32_t *) mem_53409.mem)[i_53046];
        float x_52783 = ((float *) mem_53412.mem)[i_53046];
        int32_t x_52784 = ((int32_t *) mem_53406.mem)[i_53046];
        float x_52785 = ((float *) mem_53421.mem)[i_53046];
        int32_t y_52788 = x_52781 - x_52782;
        float res_52789 = sitofp_i32_f32(x_52782);
        float res_52790;
        
        res_52790 = futrts_sqrt32(res_52789);
        
        float y_52791 = x_52783 * res_52790;
        float discard_53037;
        float scanacc_53033 = 0.0F;
        
        for (int32_t i_53035 = 0; i_53035 < iota_arg_52750; i_53035++) {
            bool cond_52808 = sle32(y_52788, i_53035);
            float res_52809;
            
            if (cond_52808) {
                res_52809 = 0.0F;
            } else {
                bool cond_52810 = i_53035 == 0;
                float res_52811;
                
                if (cond_52810) {
                    res_52811 = x_52785;
                } else {
                    int32_t x_52812 = x_52782 - x_52784;
                    int32_t i_52813 = x_52812 + i_53035;
                    float negate_arg_52814 = ((float *) mem_53355.mem)[i_53046 *
                                                                       N_52346 +
                                                                       i_52813];
                    float x_52815 = 0.0F - negate_arg_52814;
                    int32_t i_52816 = x_52782 + i_53035;
                    float y_52817 = ((float *) mem_53355.mem)[i_53046 *
                                                              N_52346 +
                                                              i_52816];
                    float res_52818 = x_52815 + y_52817;
                    
                    res_52811 = res_52818;
                }
                res_52809 = res_52811;
            }
            
            float res_52806 = res_52809 + scanacc_53033;
            
            ((float *) mem_53439.mem)[i_53035] = res_52806;
            
            float scanacc_tmp_53547 = res_52806;
            
            scanacc_53033 = scanacc_tmp_53547;
        }
        discard_53037 = scanacc_53033;
        
        bool acc0_52824;
        int32_t acc0_52825;
        float acc0_52826;
        bool redout_53038;
        int32_t redout_53039;
        float redout_53040;
        
        redout_53038 = 0;
        redout_53039 = -1;
        redout_53040 = 0.0F;
        for (int32_t i_53041 = 0; i_53041 < iota_arg_52750; i_53041++) {
            float x_52841 = ((float *) mem_53439.mem)[i_53041];
            float x_52842 = ((float *) mem_53426.mem)[i_53041];
            int32_t x_52843 = i_53041;
            float res_52845 = x_52841 / y_52791;
            bool cond_52846 = slt32(i_53041, y_52788);
            bool res_52847;
            
            res_52847 = futrts_isnan32(res_52845);
            
            bool res_52848 = !res_52847;
            bool x_52849 = cond_52846 && res_52848;
            float res_52850 = (float) fabs(res_52845);
            bool res_52851 = x_52842 < res_52850;
            bool x_52852 = x_52849 && res_52851;
            float res_52853;
            
            if (cond_52846) {
                res_52853 = res_52845;
            } else {
                res_52853 = 0.0F;
            }
            
            bool res_52833;
            int32_t res_52834;
            
            if (redout_53038) {
                res_52833 = redout_53038;
                res_52834 = redout_53039;
            } else {
                bool x_52836 = !x_52852;
                bool y_52837 = x_52836 && redout_53038;
                bool res_52838 = y_52837 || x_52852;
                int32_t res_52839;
                
                if (x_52852) {
                    res_52839 = x_52843;
                } else {
                    res_52839 = redout_53039;
                }
                res_52833 = res_52838;
                res_52834 = res_52839;
            }
            
            float res_52840 = res_52853 + redout_53040;
            bool redout_tmp_53549 = res_52833;
            int32_t redout_tmp_53550 = res_52834;
            float redout_tmp_53551;
            
            redout_tmp_53551 = res_52840;
            redout_53038 = redout_tmp_53549;
            redout_53039 = redout_tmp_53550;
            redout_53040 = redout_tmp_53551;
        }
        acc0_52824 = redout_53038;
        acc0_52825 = redout_53039;
        acc0_52826 = redout_53040;
        
        int32_t res_52860;
        
        if (acc0_52824) {
            res_52860 = acc0_52825;
        } else {
            res_52860 = -1;
        }
        
        bool cond_52862 = !acc0_52824;
        int32_t fst_breakzq_52863;
        
        if (cond_52862) {
            fst_breakzq_52863 = -1;
        } else {
            bool cond_52864 = slt32(res_52860, y_52788);
            int32_t res_52865;
            
            if (cond_52864) {
                int32_t i_52866 = x_52782 + res_52860;
                int32_t x_52867 = ((int32_t *) mem_53360.mem)[i_53046 *
                                                              N_52346 +
                                                              i_52866];
                int32_t res_52868 = x_52867 - n_52351;
                
                res_52865 = res_52868;
            } else {
                res_52865 = -1;
            }
            
            int32_t x_52869 = res_52865 - 1;
            int32_t x_52870 = sdiv32(x_52869, 2);
            int32_t x_52871 = 2 * x_52870;
            int32_t res_52872 = 1 + x_52871;
            
            fst_breakzq_52863 = res_52872;
        }
        
        bool cond_52873 = sle32(x_52782, 5);
        bool res_52874 = sle32(y_52788, 5);
        bool x_52875 = !cond_52873;
        bool y_52876 = res_52874 && x_52875;
        bool cond_52877 = cond_52873 || y_52876;
        int32_t fst_breakzq_52878;
        
        if (cond_52877) {
            fst_breakzq_52878 = -2;
        } else {
            fst_breakzq_52878 = fst_breakzq_52863;
        }
        ((int32_t *) mem_53431.mem)[i_53046] = fst_breakzq_52878;
        ((float *) mem_53434.mem)[i_53046] = acc0_52826;
    }
    if (memblock_unref(ctx, &mem_53406, "mem_53406") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53409, "mem_53409") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53412, "mem_53412") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53421, "mem_53421") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53426, "mem_53426") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53439, "mem_53439") != 0)
        return 1;
    out_arrsizze_53467 = m_52347;
    out_arrsizze_53469 = m_52347;
    out_arrsizze_53471 = N_52346;
    out_arrsizze_53472 = k2p2zq_52370;
    out_arrsizze_53474 = m_52347;
    out_arrsizze_53475 = k2p2zq_52370;
    out_arrsizze_53476 = k2p2zq_52370;
    out_arrsizze_53478 = m_52347;
    out_arrsizze_53479 = k2p2zq_52370;
    out_arrsizze_53480 = j_m_i_52504;
    out_arrsizze_53482 = m_52347;
    out_arrsizze_53483 = k2p2zq_52370;
    out_arrsizze_53485 = m_52347;
    out_arrsizze_53486 = k2p2zq_52370;
    out_arrsizze_53488 = m_52347;
    out_arrsizze_53489 = N_52346;
    out_arrsizze_53491 = m_52347;
    out_arrsizze_53492 = N_52346;
    out_arrsizze_53494 = m_52347;
    out_arrsizze_53496 = m_52347;
    out_arrsizze_53497 = N_52346;
    if (memblock_set(ctx, &out_mem_53466, &mem_53431, "mem_53431") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_53468, &mem_53434, "mem_53434") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_53470, &mem_53235, "mem_53235") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_53473, &mem_53252, "mem_53252") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_53477, &mem_53286, "mem_53286") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_53481, &mem_53307, "mem_53307") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_53484, &mem_53322, "mem_53322") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_53487, &mem_53337, "mem_53337") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_53490, &mem_53355, "mem_53355") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_53493, &mem_53350, "mem_53350") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_53495, &mem_53360, "mem_53360") != 0)
        return 1;
    (*out_mem_p_53552).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_53552, &out_mem_53466, "out_mem_53466") !=
        0)
        return 1;
    *out_out_arrsizze_53553 = out_arrsizze_53467;
    (*out_mem_p_53554).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_53554, &out_mem_53468, "out_mem_53468") !=
        0)
        return 1;
    *out_out_arrsizze_53555 = out_arrsizze_53469;
    (*out_mem_p_53556).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_53556, &out_mem_53470, "out_mem_53470") !=
        0)
        return 1;
    *out_out_arrsizze_53557 = out_arrsizze_53471;
    *out_out_arrsizze_53558 = out_arrsizze_53472;
    (*out_mem_p_53559).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_53559, &out_mem_53473, "out_mem_53473") !=
        0)
        return 1;
    *out_out_arrsizze_53560 = out_arrsizze_53474;
    *out_out_arrsizze_53561 = out_arrsizze_53475;
    *out_out_arrsizze_53562 = out_arrsizze_53476;
    (*out_mem_p_53563).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_53563, &out_mem_53477, "out_mem_53477") !=
        0)
        return 1;
    *out_out_arrsizze_53564 = out_arrsizze_53478;
    *out_out_arrsizze_53565 = out_arrsizze_53479;
    *out_out_arrsizze_53566 = out_arrsizze_53480;
    (*out_mem_p_53567).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_53567, &out_mem_53481, "out_mem_53481") !=
        0)
        return 1;
    *out_out_arrsizze_53568 = out_arrsizze_53482;
    *out_out_arrsizze_53569 = out_arrsizze_53483;
    (*out_mem_p_53570).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_53570, &out_mem_53484, "out_mem_53484") !=
        0)
        return 1;
    *out_out_arrsizze_53571 = out_arrsizze_53485;
    *out_out_arrsizze_53572 = out_arrsizze_53486;
    (*out_mem_p_53573).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_53573, &out_mem_53487, "out_mem_53487") !=
        0)
        return 1;
    *out_out_arrsizze_53574 = out_arrsizze_53488;
    *out_out_arrsizze_53575 = out_arrsizze_53489;
    (*out_mem_p_53576).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_53576, &out_mem_53490, "out_mem_53490") !=
        0)
        return 1;
    *out_out_arrsizze_53577 = out_arrsizze_53491;
    *out_out_arrsizze_53578 = out_arrsizze_53492;
    (*out_mem_p_53579).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_53579, &out_mem_53493, "out_mem_53493") !=
        0)
        return 1;
    *out_out_arrsizze_53580 = out_arrsizze_53494;
    (*out_mem_p_53581).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_53581, &out_mem_53495, "out_mem_53495") !=
        0)
        return 1;
    *out_out_arrsizze_53582 = out_arrsizze_53496;
    *out_out_arrsizze_53583 = out_arrsizze_53497;
    if (memblock_unref(ctx, &mem_53439, "mem_53439") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53434, "mem_53434") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53431, "mem_53431") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53426, "mem_53426") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53421, "mem_53421") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53412, "mem_53412") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53409, "mem_53409") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53406, "mem_53406") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53394, "mem_53394") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53384, "mem_53384") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53379, "mem_53379") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53376, "mem_53376") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53369, "mem_53369") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53366, "mem_53366") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53360, "mem_53360") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53355, "mem_53355") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53350, "mem_53350") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53337, "mem_53337") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53322, "mem_53322") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53307, "mem_53307") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53296, "mem_53296") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53290, "mem_53290") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53286, "mem_53286") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53252, "mem_53252") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53235, "mem_53235") != 0)
        return 1;
    if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_53230,
                       "lifted_1_zlzb_arg_mem_53230") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_53495, "out_mem_53495") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_53493, "out_mem_53493") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_53490, "out_mem_53490") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_53487, "out_mem_53487") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_53484, "out_mem_53484") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_53481, "out_mem_53481") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_53477, "out_mem_53477") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_53473, "out_mem_53473") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_53470, "out_mem_53470") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_53468, "out_mem_53468") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_53466, "out_mem_53466") != 0)
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
                       struct futhark_i32_2d **out10, const int32_t in0, const
                       int32_t in1, const int32_t in2, const float in3, const
                       float in4, const float in5, const
                       struct futhark_i32_1d *in6, const
                       struct futhark_f32_2d *in7)
{
    struct memblock mappingindices_mem_53198;
    
    mappingindices_mem_53198.references = NULL;
    
    struct memblock images_mem_53199;
    
    images_mem_53199.references = NULL;
    
    int32_t N_52346;
    int32_t m_52347;
    int32_t N_52348;
    int32_t trend_52349;
    int32_t k_52350;
    int32_t n_52351;
    float freq_52352;
    float hfrac_52353;
    float lam_52354;
    struct memblock out_mem_53466;
    
    out_mem_53466.references = NULL;
    
    int32_t out_arrsizze_53467;
    struct memblock out_mem_53468;
    
    out_mem_53468.references = NULL;
    
    int32_t out_arrsizze_53469;
    struct memblock out_mem_53470;
    
    out_mem_53470.references = NULL;
    
    int32_t out_arrsizze_53471;
    int32_t out_arrsizze_53472;
    struct memblock out_mem_53473;
    
    out_mem_53473.references = NULL;
    
    int32_t out_arrsizze_53474;
    int32_t out_arrsizze_53475;
    int32_t out_arrsizze_53476;
    struct memblock out_mem_53477;
    
    out_mem_53477.references = NULL;
    
    int32_t out_arrsizze_53478;
    int32_t out_arrsizze_53479;
    int32_t out_arrsizze_53480;
    struct memblock out_mem_53481;
    
    out_mem_53481.references = NULL;
    
    int32_t out_arrsizze_53482;
    int32_t out_arrsizze_53483;
    struct memblock out_mem_53484;
    
    out_mem_53484.references = NULL;
    
    int32_t out_arrsizze_53485;
    int32_t out_arrsizze_53486;
    struct memblock out_mem_53487;
    
    out_mem_53487.references = NULL;
    
    int32_t out_arrsizze_53488;
    int32_t out_arrsizze_53489;
    struct memblock out_mem_53490;
    
    out_mem_53490.references = NULL;
    
    int32_t out_arrsizze_53491;
    int32_t out_arrsizze_53492;
    struct memblock out_mem_53493;
    
    out_mem_53493.references = NULL;
    
    int32_t out_arrsizze_53494;
    struct memblock out_mem_53495;
    
    out_mem_53495.references = NULL;
    
    int32_t out_arrsizze_53496;
    int32_t out_arrsizze_53497;
    
    lock_lock(&ctx->lock);
    trend_52349 = in0;
    k_52350 = in1;
    n_52351 = in2;
    freq_52352 = in3;
    hfrac_52353 = in4;
    lam_52354 = in5;
    mappingindices_mem_53198 = in6->mem;
    N_52346 = in6->shape[0];
    images_mem_53199 = in7->mem;
    m_52347 = in7->shape[0];
    N_52348 = in7->shape[1];
    
    int ret = futrts_main(ctx, &out_mem_53466, &out_arrsizze_53467,
                          &out_mem_53468, &out_arrsizze_53469, &out_mem_53470,
                          &out_arrsizze_53471, &out_arrsizze_53472,
                          &out_mem_53473, &out_arrsizze_53474,
                          &out_arrsizze_53475, &out_arrsizze_53476,
                          &out_mem_53477, &out_arrsizze_53478,
                          &out_arrsizze_53479, &out_arrsizze_53480,
                          &out_mem_53481, &out_arrsizze_53482,
                          &out_arrsizze_53483, &out_mem_53484,
                          &out_arrsizze_53485, &out_arrsizze_53486,
                          &out_mem_53487, &out_arrsizze_53488,
                          &out_arrsizze_53489, &out_mem_53490,
                          &out_arrsizze_53491, &out_arrsizze_53492,
                          &out_mem_53493, &out_arrsizze_53494, &out_mem_53495,
                          &out_arrsizze_53496, &out_arrsizze_53497,
                          mappingindices_mem_53198, images_mem_53199, N_52346,
                          m_52347, N_52348, trend_52349, k_52350, n_52351,
                          freq_52352, hfrac_52353, lam_52354);
    
    if (ret == 0) {
        assert((*out0 =
                (struct futhark_i32_1d *) malloc(sizeof(struct futhark_i32_1d))) !=
            NULL);
        (*out0)->mem = out_mem_53466;
        (*out0)->shape[0] = out_arrsizze_53467;
        assert((*out1 =
                (struct futhark_f32_1d *) malloc(sizeof(struct futhark_f32_1d))) !=
            NULL);
        (*out1)->mem = out_mem_53468;
        (*out1)->shape[0] = out_arrsizze_53469;
        assert((*out2 =
                (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) !=
            NULL);
        (*out2)->mem = out_mem_53470;
        (*out2)->shape[0] = out_arrsizze_53471;
        (*out2)->shape[1] = out_arrsizze_53472;
        assert((*out3 =
                (struct futhark_f32_3d *) malloc(sizeof(struct futhark_f32_3d))) !=
            NULL);
        (*out3)->mem = out_mem_53473;
        (*out3)->shape[0] = out_arrsizze_53474;
        (*out3)->shape[1] = out_arrsizze_53475;
        (*out3)->shape[2] = out_arrsizze_53476;
        assert((*out4 =
                (struct futhark_f32_3d *) malloc(sizeof(struct futhark_f32_3d))) !=
            NULL);
        (*out4)->mem = out_mem_53477;
        (*out4)->shape[0] = out_arrsizze_53478;
        (*out4)->shape[1] = out_arrsizze_53479;
        (*out4)->shape[2] = out_arrsizze_53480;
        assert((*out5 =
                (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) !=
            NULL);
        (*out5)->mem = out_mem_53481;
        (*out5)->shape[0] = out_arrsizze_53482;
        (*out5)->shape[1] = out_arrsizze_53483;
        assert((*out6 =
                (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) !=
            NULL);
        (*out6)->mem = out_mem_53484;
        (*out6)->shape[0] = out_arrsizze_53485;
        (*out6)->shape[1] = out_arrsizze_53486;
        assert((*out7 =
                (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) !=
            NULL);
        (*out7)->mem = out_mem_53487;
        (*out7)->shape[0] = out_arrsizze_53488;
        (*out7)->shape[1] = out_arrsizze_53489;
        assert((*out8 =
                (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) !=
            NULL);
        (*out8)->mem = out_mem_53490;
        (*out8)->shape[0] = out_arrsizze_53491;
        (*out8)->shape[1] = out_arrsizze_53492;
        assert((*out9 =
                (struct futhark_i32_1d *) malloc(sizeof(struct futhark_i32_1d))) !=
            NULL);
        (*out9)->mem = out_mem_53493;
        (*out9)->shape[0] = out_arrsizze_53494;
        assert((*out10 =
                (struct futhark_i32_2d *) malloc(sizeof(struct futhark_i32_2d))) !=
            NULL);
        (*out10)->mem = out_mem_53495;
        (*out10)->shape[0] = out_arrsizze_53496;
        (*out10)->shape[1] = out_arrsizze_53497;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
