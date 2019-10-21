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
                       struct futhark_f32_3d **out1,
                       struct futhark_f32_3d **out2,
                       struct futhark_f32_2d **out3,
                       struct futhark_f32_2d **out4,
                       struct futhark_f32_2d **out5,
                       struct futhark_i32_1d **out6,
                       struct futhark_f32_2d **out7,
                       struct futhark_i32_2d **out8, const int32_t in0, const
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
    
    int32_t read_value_66459;
    
    if (read_scalar(&i32_info, &read_value_66459) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 0,
              i32_info.type_name, strerror(errno));
    
    int32_t read_value_66460;
    
    if (read_scalar(&i32_info, &read_value_66460) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 1,
              i32_info.type_name, strerror(errno));
    
    int32_t read_value_66461;
    
    if (read_scalar(&i32_info, &read_value_66461) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 2,
              i32_info.type_name, strerror(errno));
    
    float read_value_66462;
    
    if (read_scalar(&f32_info, &read_value_66462) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 3,
              f32_info.type_name, strerror(errno));
    
    float read_value_66463;
    
    if (read_scalar(&f32_info, &read_value_66463) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 4,
              f32_info.type_name, strerror(errno));
    
    float read_value_66464;
    
    if (read_scalar(&f32_info, &read_value_66464) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 5,
              f32_info.type_name, strerror(errno));
    
    struct futhark_i32_1d *read_value_66465;
    int64_t read_shape_66466[1];
    int32_t *read_arr_66467 = NULL;
    
    errno = 0;
    if (read_array(&i32_info, (void **) &read_arr_66467, read_shape_66466, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 6, "[]",
              i32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_66468;
    int64_t read_shape_66469[2];
    float *read_arr_66470 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_66470, read_shape_66469, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 7, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *result_66471;
    struct futhark_f32_3d *result_66472;
    struct futhark_f32_3d *result_66473;
    struct futhark_f32_2d *result_66474;
    struct futhark_f32_2d *result_66475;
    struct futhark_f32_2d *result_66476;
    struct futhark_i32_1d *result_66477;
    struct futhark_f32_2d *result_66478;
    struct futhark_i32_2d *result_66479;
    
    if (perform_warmup) {
        time_runs = 0;
        
        int r;
        
        ;
        ;
        ;
        ;
        ;
        ;
        assert((read_value_66465 = futhark_new_i32_1d(ctx, read_arr_66467,
                                                      read_shape_66466[0])) !=
            0);
        assert((read_value_66468 = futhark_new_f32_2d(ctx, read_arr_66470,
                                                      read_shape_66469[0],
                                                      read_shape_66469[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_main(ctx, &result_66471, &result_66472, &result_66473,
                               &result_66474, &result_66475, &result_66476,
                               &result_66477, &result_66478, &result_66479,
                               read_value_66459, read_value_66460,
                               read_value_66461, read_value_66462,
                               read_value_66463, read_value_66464,
                               read_value_66465, read_value_66468);
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
        assert(futhark_free_i32_1d(ctx, read_value_66465) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_66468) == 0);
        assert(futhark_free_f32_2d(ctx, result_66471) == 0);
        assert(futhark_free_f32_3d(ctx, result_66472) == 0);
        assert(futhark_free_f32_3d(ctx, result_66473) == 0);
        assert(futhark_free_f32_2d(ctx, result_66474) == 0);
        assert(futhark_free_f32_2d(ctx, result_66475) == 0);
        assert(futhark_free_f32_2d(ctx, result_66476) == 0);
        assert(futhark_free_i32_1d(ctx, result_66477) == 0);
        assert(futhark_free_f32_2d(ctx, result_66478) == 0);
        assert(futhark_free_i32_2d(ctx, result_66479) == 0);
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
        assert((read_value_66465 = futhark_new_i32_1d(ctx, read_arr_66467,
                                                      read_shape_66466[0])) !=
            0);
        assert((read_value_66468 = futhark_new_f32_2d(ctx, read_arr_66470,
                                                      read_shape_66469[0],
                                                      read_shape_66469[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_main(ctx, &result_66471, &result_66472, &result_66473,
                               &result_66474, &result_66475, &result_66476,
                               &result_66477, &result_66478, &result_66479,
                               read_value_66459, read_value_66460,
                               read_value_66461, read_value_66462,
                               read_value_66463, read_value_66464,
                               read_value_66465, read_value_66468);
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
        assert(futhark_free_i32_1d(ctx, read_value_66465) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_66468) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_2d(ctx, result_66471) == 0);
            assert(futhark_free_f32_3d(ctx, result_66472) == 0);
            assert(futhark_free_f32_3d(ctx, result_66473) == 0);
            assert(futhark_free_f32_2d(ctx, result_66474) == 0);
            assert(futhark_free_f32_2d(ctx, result_66475) == 0);
            assert(futhark_free_f32_2d(ctx, result_66476) == 0);
            assert(futhark_free_i32_1d(ctx, result_66477) == 0);
            assert(futhark_free_f32_2d(ctx, result_66478) == 0);
            assert(futhark_free_i32_2d(ctx, result_66479) == 0);
        }
    }
    ;
    ;
    ;
    ;
    ;
    ;
    free(read_arr_66467);
    free(read_arr_66470);
    if (binary_output)
        set_binary_mode(stdout);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_2d(ctx,
                                                                result_66471)[0] *
                            futhark_shape_f32_2d(ctx, result_66471)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_2d(ctx, result_66471, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_2d(ctx, result_66471), 2);
        free(arr);
    }
    printf("\n");
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_3d(ctx,
                                                                result_66472)[0] *
                            futhark_shape_f32_3d(ctx, result_66472)[1] *
                            futhark_shape_f32_3d(ctx, result_66472)[2]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_3d(ctx, result_66472, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_3d(ctx, result_66472), 3);
        free(arr);
    }
    printf("\n");
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_3d(ctx,
                                                                result_66473)[0] *
                            futhark_shape_f32_3d(ctx, result_66473)[1] *
                            futhark_shape_f32_3d(ctx, result_66473)[2]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_3d(ctx, result_66473, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_3d(ctx, result_66473), 3);
        free(arr);
    }
    printf("\n");
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_2d(ctx,
                                                                result_66474)[0] *
                            futhark_shape_f32_2d(ctx, result_66474)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_2d(ctx, result_66474, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_2d(ctx, result_66474), 2);
        free(arr);
    }
    printf("\n");
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_2d(ctx,
                                                                result_66475)[0] *
                            futhark_shape_f32_2d(ctx, result_66475)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_2d(ctx, result_66475, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_2d(ctx, result_66475), 2);
        free(arr);
    }
    printf("\n");
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_2d(ctx,
                                                                result_66476)[0] *
                            futhark_shape_f32_2d(ctx, result_66476)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_2d(ctx, result_66476, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_2d(ctx, result_66476), 2);
        free(arr);
    }
    printf("\n");
    {
        int32_t *arr = calloc(sizeof(int32_t), futhark_shape_i32_1d(ctx,
                                                                    result_66477)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_i32_1d(ctx, result_66477, arr) == 0);
        write_array(stdout, binary_output, &i32_info, arr,
                    futhark_shape_i32_1d(ctx, result_66477), 1);
        free(arr);
    }
    printf("\n");
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_2d(ctx,
                                                                result_66478)[0] *
                            futhark_shape_f32_2d(ctx, result_66478)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_2d(ctx, result_66478, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_2d(ctx, result_66478), 2);
        free(arr);
    }
    printf("\n");
    {
        int32_t *arr = calloc(sizeof(int32_t), futhark_shape_i32_2d(ctx,
                                                                    result_66479)[0] *
                              futhark_shape_i32_2d(ctx, result_66479)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_i32_2d(ctx, result_66479, arr) == 0);
        write_array(stdout, binary_output, &i32_info, arr,
                    futhark_shape_i32_2d(ctx, result_66479), 2);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_2d(ctx, result_66471) == 0);
    assert(futhark_free_f32_3d(ctx, result_66472) == 0);
    assert(futhark_free_f32_3d(ctx, result_66473) == 0);
    assert(futhark_free_f32_2d(ctx, result_66474) == 0);
    assert(futhark_free_f32_2d(ctx, result_66475) == 0);
    assert(futhark_free_f32_2d(ctx, result_66476) == 0);
    assert(futhark_free_i32_1d(ctx, result_66477) == 0);
    assert(futhark_free_f32_2d(ctx, result_66478) == 0);
    assert(futhark_free_i32_2d(ctx, result_66479) == 0);
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
                       struct memblock *out_mem_p_66431,
                       int32_t *out_out_arrsizze_66432,
                       int32_t *out_out_arrsizze_66433,
                       struct memblock *out_mem_p_66434,
                       int32_t *out_out_arrsizze_66435,
                       int32_t *out_out_arrsizze_66436,
                       int32_t *out_out_arrsizze_66437,
                       struct memblock *out_mem_p_66438,
                       int32_t *out_out_arrsizze_66439,
                       int32_t *out_out_arrsizze_66440,
                       int32_t *out_out_arrsizze_66441,
                       struct memblock *out_mem_p_66442,
                       int32_t *out_out_arrsizze_66443,
                       int32_t *out_out_arrsizze_66444,
                       struct memblock *out_mem_p_66445,
                       int32_t *out_out_arrsizze_66446,
                       int32_t *out_out_arrsizze_66447,
                       struct memblock *out_mem_p_66448,
                       int32_t *out_out_arrsizze_66449,
                       int32_t *out_out_arrsizze_66450,
                       struct memblock *out_mem_p_66451,
                       int32_t *out_out_arrsizze_66452,
                       struct memblock *out_mem_p_66453,
                       int32_t *out_out_arrsizze_66454,
                       int32_t *out_out_arrsizze_66455,
                       struct memblock *out_mem_p_66456,
                       int32_t *out_out_arrsizze_66457,
                       int32_t *out_out_arrsizze_66458,
                       struct memblock mappingindices_mem_66137,
                       struct memblock images_mem_66138, int32_t N_65548,
                       int32_t m_65549, int32_t N_65550, int32_t trend_65551,
                       int32_t k_65552, int32_t n_65553, float freq_65554,
                       float hfrac_65555, float lam_65556);
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
                       struct memblock *out_mem_p_66431,
                       int32_t *out_out_arrsizze_66432,
                       int32_t *out_out_arrsizze_66433,
                       struct memblock *out_mem_p_66434,
                       int32_t *out_out_arrsizze_66435,
                       int32_t *out_out_arrsizze_66436,
                       int32_t *out_out_arrsizze_66437,
                       struct memblock *out_mem_p_66438,
                       int32_t *out_out_arrsizze_66439,
                       int32_t *out_out_arrsizze_66440,
                       int32_t *out_out_arrsizze_66441,
                       struct memblock *out_mem_p_66442,
                       int32_t *out_out_arrsizze_66443,
                       int32_t *out_out_arrsizze_66444,
                       struct memblock *out_mem_p_66445,
                       int32_t *out_out_arrsizze_66446,
                       int32_t *out_out_arrsizze_66447,
                       struct memblock *out_mem_p_66448,
                       int32_t *out_out_arrsizze_66449,
                       int32_t *out_out_arrsizze_66450,
                       struct memblock *out_mem_p_66451,
                       int32_t *out_out_arrsizze_66452,
                       struct memblock *out_mem_p_66453,
                       int32_t *out_out_arrsizze_66454,
                       int32_t *out_out_arrsizze_66455,
                       struct memblock *out_mem_p_66456,
                       int32_t *out_out_arrsizze_66457,
                       int32_t *out_out_arrsizze_66458,
                       struct memblock mappingindices_mem_66137,
                       struct memblock images_mem_66138, int32_t N_65548,
                       int32_t m_65549, int32_t N_65550, int32_t trend_65551,
                       int32_t k_65552, int32_t n_65553, float freq_65554,
                       float hfrac_65555, float lam_65556)
{
    struct memblock out_mem_66365;
    
    out_mem_66365.references = NULL;
    
    int32_t out_arrsizze_66366;
    int32_t out_arrsizze_66367;
    struct memblock out_mem_66368;
    
    out_mem_66368.references = NULL;
    
    int32_t out_arrsizze_66369;
    int32_t out_arrsizze_66370;
    int32_t out_arrsizze_66371;
    struct memblock out_mem_66372;
    
    out_mem_66372.references = NULL;
    
    int32_t out_arrsizze_66373;
    int32_t out_arrsizze_66374;
    int32_t out_arrsizze_66375;
    struct memblock out_mem_66376;
    
    out_mem_66376.references = NULL;
    
    int32_t out_arrsizze_66377;
    int32_t out_arrsizze_66378;
    struct memblock out_mem_66379;
    
    out_mem_66379.references = NULL;
    
    int32_t out_arrsizze_66380;
    int32_t out_arrsizze_66381;
    struct memblock out_mem_66382;
    
    out_mem_66382.references = NULL;
    
    int32_t out_arrsizze_66383;
    int32_t out_arrsizze_66384;
    struct memblock out_mem_66385;
    
    out_mem_66385.references = NULL;
    
    int32_t out_arrsizze_66386;
    struct memblock out_mem_66387;
    
    out_mem_66387.references = NULL;
    
    int32_t out_arrsizze_66388;
    int32_t out_arrsizze_66389;
    struct memblock out_mem_66390;
    
    out_mem_66390.references = NULL;
    
    int32_t out_arrsizze_66391;
    int32_t out_arrsizze_66392;
    bool dim_zzero_65559 = 0 == m_65549;
    bool dim_zzero_65560 = 0 == N_65550;
    bool old_empty_65561 = dim_zzero_65559 || dim_zzero_65560;
    bool dim_zzero_65562 = 0 == N_65548;
    bool new_empty_65563 = dim_zzero_65559 || dim_zzero_65562;
    bool both_empty_65564 = old_empty_65561 && new_empty_65563;
    bool dim_match_65565 = N_65548 == N_65550;
    bool empty_or_match_65566 = both_empty_65564 || dim_match_65565;
    bool empty_or_match_cert_65567;
    
    if (!empty_or_match_65566) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "bfast-irreg.fut:133:1-316:68",
                               "function arguments of wrong shape");
        if (memblock_unref(ctx, &out_mem_66390, "out_mem_66390") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66387, "out_mem_66387") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66385, "out_mem_66385") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66382, "out_mem_66382") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66379, "out_mem_66379") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66376, "out_mem_66376") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66372, "out_mem_66372") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66368, "out_mem_66368") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66365, "out_mem_66365") != 0)
            return 1;
        return 1;
    }
    
    int32_t x_65569 = 2 * k_65552;
    int32_t k2p2_65570 = 2 + x_65569;
    bool cond_65571 = slt32(0, trend_65551);
    int32_t k2p2zq_65572;
    
    if (cond_65571) {
        k2p2zq_65572 = k2p2_65570;
    } else {
        int32_t res_65573 = k2p2_65570 - 1;
        
        k2p2zq_65572 = res_65573;
    }
    
    int64_t binop_x_66140 = sext_i32_i64(k2p2zq_65572);
    int64_t binop_y_66141 = sext_i32_i64(N_65548);
    int64_t binop_x_66142 = binop_x_66140 * binop_y_66141;
    int64_t bytes_66139 = 4 * binop_x_66142;
    int64_t binop_x_66155 = sext_i32_i64(k2p2zq_65572);
    int64_t binop_y_66156 = sext_i32_i64(N_65548);
    int64_t binop_x_66157 = binop_x_66155 * binop_y_66156;
    int64_t bytes_66154 = 4 * binop_x_66157;
    struct memblock lifted_1_zlzb_arg_mem_66169;
    
    lifted_1_zlzb_arg_mem_66169.references = NULL;
    if (cond_65571) {
        bool bounds_invalid_upwards_65575 = slt32(k2p2zq_65572, 0);
        bool eq_x_zz_65576 = 0 == k2p2zq_65572;
        bool not_p_65577 = !bounds_invalid_upwards_65575;
        bool p_and_eq_x_y_65578 = eq_x_zz_65576 && not_p_65577;
        bool dim_zzero_65579 = bounds_invalid_upwards_65575 ||
             p_and_eq_x_y_65578;
        bool both_empty_65580 = eq_x_zz_65576 && dim_zzero_65579;
        bool empty_or_match_65584 = not_p_65577 || both_empty_65580;
        bool empty_or_match_cert_65585;
        
        if (!empty_or_match_65584) {
            ctx->error = msgprintf("Error at %s:\n%s%s%s%d%s%s\n",
                                   "bfast-irreg.fut:133:1-316:68 -> bfast-irreg.fut:144:16-55 -> bfast-irreg.fut:64:10-18 -> /futlib/array.fut:61:1-62:12",
                                   "Function return value does not match shape of type ",
                                   "*", "[", k2p2zq_65572, "]",
                                   "intrinsics.i32");
            if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_66169,
                               "lifted_1_zlzb_arg_mem_66169") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_66390, "out_mem_66390") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_66387, "out_mem_66387") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_66385, "out_mem_66385") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_66382, "out_mem_66382") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_66379, "out_mem_66379") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_66376, "out_mem_66376") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_66372, "out_mem_66372") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_66368, "out_mem_66368") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_66365, "out_mem_66365") != 0)
                return 1;
            return 1;
        }
        
        struct memblock mem_66143;
        
        mem_66143.references = NULL;
        if (memblock_alloc(ctx, &mem_66143, bytes_66139, "mem_66143"))
            return 1;
        for (int32_t i_65892 = 0; i_65892 < k2p2zq_65572; i_65892++) {
            bool cond_65589 = i_65892 == 0;
            bool cond_65590 = i_65892 == 1;
            int32_t r32_arg_65591 = sdiv32(i_65892, 2);
            int32_t x_65592 = smod32(i_65892, 2);
            float res_65593 = sitofp_i32_f32(r32_arg_65591);
            bool cond_65594 = x_65592 == 0;
            float x_65595 = 6.2831855F * res_65593;
            
            for (int32_t i_65888 = 0; i_65888 < N_65548; i_65888++) {
                int32_t x_65597 =
                        ((int32_t *) mappingindices_mem_66137.mem)[i_65888];
                float res_65598;
                
                if (cond_65589) {
                    res_65598 = 1.0F;
                } else {
                    float res_65599;
                    
                    if (cond_65590) {
                        float res_65600 = sitofp_i32_f32(x_65597);
                        
                        res_65599 = res_65600;
                    } else {
                        float res_65601 = sitofp_i32_f32(x_65597);
                        float x_65602 = x_65595 * res_65601;
                        float angle_65603 = x_65602 / freq_65554;
                        float res_65604;
                        
                        if (cond_65594) {
                            float res_65605;
                            
                            res_65605 = futrts_sin32(angle_65603);
                            res_65604 = res_65605;
                        } else {
                            float res_65606;
                            
                            res_65606 = futrts_cos32(angle_65603);
                            res_65604 = res_65606;
                        }
                        res_65599 = res_65604;
                    }
                    res_65598 = res_65599;
                }
                ((float *) mem_66143.mem)[i_65892 * N_65548 + i_65888] =
                    res_65598;
            }
        }
        if (memblock_set(ctx, &lifted_1_zlzb_arg_mem_66169, &mem_66143,
                         "mem_66143") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_66143, "mem_66143") != 0)
            return 1;
    } else {
        bool bounds_invalid_upwards_65607 = slt32(k2p2zq_65572, 0);
        bool eq_x_zz_65608 = 0 == k2p2zq_65572;
        bool not_p_65609 = !bounds_invalid_upwards_65607;
        bool p_and_eq_x_y_65610 = eq_x_zz_65608 && not_p_65609;
        bool dim_zzero_65611 = bounds_invalid_upwards_65607 ||
             p_and_eq_x_y_65610;
        bool both_empty_65612 = eq_x_zz_65608 && dim_zzero_65611;
        bool empty_or_match_65616 = not_p_65609 || both_empty_65612;
        bool empty_or_match_cert_65617;
        
        if (!empty_or_match_65616) {
            ctx->error = msgprintf("Error at %s:\n%s%s%s%d%s%s\n",
                                   "bfast-irreg.fut:133:1-316:68 -> bfast-irreg.fut:145:16-55 -> bfast-irreg.fut:76:10-20 -> /futlib/array.fut:61:1-62:12",
                                   "Function return value does not match shape of type ",
                                   "*", "[", k2p2zq_65572, "]",
                                   "intrinsics.i32");
            if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_66169,
                               "lifted_1_zlzb_arg_mem_66169") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_66390, "out_mem_66390") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_66387, "out_mem_66387") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_66385, "out_mem_66385") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_66382, "out_mem_66382") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_66379, "out_mem_66379") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_66376, "out_mem_66376") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_66372, "out_mem_66372") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_66368, "out_mem_66368") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_66365, "out_mem_66365") != 0)
                return 1;
            return 1;
        }
        
        struct memblock mem_66158;
        
        mem_66158.references = NULL;
        if (memblock_alloc(ctx, &mem_66158, bytes_66154, "mem_66158"))
            return 1;
        for (int32_t i_65900 = 0; i_65900 < k2p2zq_65572; i_65900++) {
            bool cond_65621 = i_65900 == 0;
            int32_t i_65622 = 1 + i_65900;
            int32_t r32_arg_65623 = sdiv32(i_65622, 2);
            int32_t x_65624 = smod32(i_65622, 2);
            float res_65625 = sitofp_i32_f32(r32_arg_65623);
            bool cond_65626 = x_65624 == 0;
            float x_65627 = 6.2831855F * res_65625;
            
            for (int32_t i_65896 = 0; i_65896 < N_65548; i_65896++) {
                int32_t x_65629 =
                        ((int32_t *) mappingindices_mem_66137.mem)[i_65896];
                float res_65630;
                
                if (cond_65621) {
                    res_65630 = 1.0F;
                } else {
                    float res_65631 = sitofp_i32_f32(x_65629);
                    float x_65632 = x_65627 * res_65631;
                    float angle_65633 = x_65632 / freq_65554;
                    float res_65634;
                    
                    if (cond_65626) {
                        float res_65635;
                        
                        res_65635 = futrts_sin32(angle_65633);
                        res_65634 = res_65635;
                    } else {
                        float res_65636;
                        
                        res_65636 = futrts_cos32(angle_65633);
                        res_65634 = res_65636;
                    }
                    res_65630 = res_65634;
                }
                ((float *) mem_66158.mem)[i_65900 * N_65548 + i_65896] =
                    res_65630;
            }
        }
        if (memblock_set(ctx, &lifted_1_zlzb_arg_mem_66169, &mem_66158,
                         "mem_66158") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_66158, "mem_66158") != 0)
            return 1;
    }
    
    int32_t x_65638 = N_65548 * N_65548;
    int32_t y_65639 = 2 * N_65548;
    int32_t x_65640 = x_65638 + y_65639;
    int32_t x_65641 = 1 + x_65640;
    int32_t y_65642 = 1 + N_65548;
    int32_t x_65643 = sdiv32(x_65641, y_65642);
    int32_t x_65644 = x_65643 - N_65548;
    int32_t lifted_1_zlzb_arg_65645 = x_65644 - 1;
    float res_65646 = sitofp_i32_f32(lifted_1_zlzb_arg_65645);
    int64_t binop_x_66171 = sext_i32_i64(N_65548);
    int64_t binop_y_66172 = sext_i32_i64(k2p2zq_65572);
    int64_t binop_x_66173 = binop_x_66171 * binop_y_66172;
    int64_t bytes_66170 = 4 * binop_x_66173;
    struct memblock mem_66174;
    
    mem_66174.references = NULL;
    if (memblock_alloc(ctx, &mem_66174, bytes_66170, "mem_66174"))
        return 1;
    for (int32_t i_65908 = 0; i_65908 < N_65548; i_65908++) {
        for (int32_t i_65904 = 0; i_65904 < k2p2zq_65572; i_65904++) {
            float x_65651 =
                  ((float *) lifted_1_zlzb_arg_mem_66169.mem)[i_65904 *
                                                              N_65548 +
                                                              i_65908];
            float res_65652 = res_65646 + x_65651;
            
            ((float *) mem_66174.mem)[i_65908 * k2p2zq_65572 + i_65904] =
                res_65652;
        }
    }
    
    int32_t m_65655 = k2p2zq_65572 - 1;
    bool empty_slice_65662 = n_65553 == 0;
    int32_t m_65663 = n_65553 - 1;
    bool zzero_leq_i_p_m_t_s_65664 = sle32(0, m_65663);
    bool i_p_m_t_s_leq_w_65665 = slt32(m_65663, N_65548);
    bool i_lte_j_65666 = sle32(0, n_65553);
    bool y_65667 = zzero_leq_i_p_m_t_s_65664 && i_p_m_t_s_leq_w_65665;
    bool y_65668 = i_lte_j_65666 && y_65667;
    bool ok_or_empty_65669 = empty_slice_65662 || y_65668;
    bool index_certs_65671;
    
    if (!ok_or_empty_65669) {
        ctx->error = msgprintf("Error at %s:\n%s%d%s%s%s%d%s%d%s%d%s\n",
                               "bfast-irreg.fut:133:1-316:68 -> bfast-irreg.fut:154:15-21",
                               "Index [", 0, ", ", "", ":", n_65553,
                               "] out of bounds for array of shape [",
                               k2p2zq_65572, "][", N_65548, "].");
        if (memblock_unref(ctx, &mem_66174, "mem_66174") != 0)
            return 1;
        if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_66169,
                           "lifted_1_zlzb_arg_mem_66169") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66390, "out_mem_66390") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66387, "out_mem_66387") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66385, "out_mem_66385") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66382, "out_mem_66382") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66379, "out_mem_66379") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66376, "out_mem_66376") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66372, "out_mem_66372") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66368, "out_mem_66368") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66365, "out_mem_66365") != 0)
            return 1;
        return 1;
    }
    
    bool index_certs_65673;
    
    if (!ok_or_empty_65669) {
        ctx->error = msgprintf("Error at %s:\n%s%s%s%d%s%d%s%d%s%d%s\n",
                               "bfast-irreg.fut:133:1-316:68 -> bfast-irreg.fut:155:15-22",
                               "Index [", "", ":", n_65553, ", ", 0,
                               "] out of bounds for array of shape [", N_65548,
                               "][", k2p2zq_65572, "].");
        if (memblock_unref(ctx, &mem_66174, "mem_66174") != 0)
            return 1;
        if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_66169,
                           "lifted_1_zlzb_arg_mem_66169") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66390, "out_mem_66390") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66387, "out_mem_66387") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66385, "out_mem_66385") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66382, "out_mem_66382") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66379, "out_mem_66379") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66376, "out_mem_66376") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66372, "out_mem_66372") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66368, "out_mem_66368") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66365, "out_mem_66365") != 0)
            return 1;
        return 1;
    }
    
    bool index_certs_65684;
    
    if (!ok_or_empty_65669) {
        ctx->error = msgprintf("Error at %s:\n%s%d%s%s%s%d%s%d%s%d%s\n",
                               "bfast-irreg.fut:133:1-316:68 -> bfast-irreg.fut:156:15-26",
                               "Index [", 0, ", ", "", ":", n_65553,
                               "] out of bounds for array of shape [", m_65549,
                               "][", N_65548, "].");
        if (memblock_unref(ctx, &mem_66174, "mem_66174") != 0)
            return 1;
        if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_66169,
                           "lifted_1_zlzb_arg_mem_66169") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66390, "out_mem_66390") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66387, "out_mem_66387") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66385, "out_mem_66385") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66382, "out_mem_66382") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66379, "out_mem_66379") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66376, "out_mem_66376") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66372, "out_mem_66372") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66368, "out_mem_66368") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66365, "out_mem_66365") != 0)
            return 1;
        return 1;
    }
    
    int64_t binop_x_66186 = sext_i32_i64(m_65549);
    int64_t binop_x_66188 = binop_y_66172 * binop_x_66186;
    int64_t binop_x_66190 = binop_y_66172 * binop_x_66188;
    int64_t bytes_66185 = 4 * binop_x_66190;
    struct memblock mem_66191;
    
    mem_66191.references = NULL;
    if (memblock_alloc(ctx, &mem_66191, bytes_66185, "mem_66191"))
        return 1;
    for (int32_t i_65922 = 0; i_65922 < m_65549; i_65922++) {
        for (int32_t i_65918 = 0; i_65918 < k2p2zq_65572; i_65918++) {
            for (int32_t i_65914 = 0; i_65914 < k2p2zq_65572; i_65914++) {
                float res_65693;
                float redout_65910 = 0.0F;
                
                for (int32_t i_65911 = 0; i_65911 < n_65553; i_65911++) {
                    float x_65697 = ((float *) images_mem_66138.mem)[i_65922 *
                                                                     N_65550 +
                                                                     i_65911];
                    float x_65698 =
                          ((float *) lifted_1_zlzb_arg_mem_66169.mem)[i_65918 *
                                                                      N_65548 +
                                                                      i_65911];
                    float x_65699 = ((float *) mem_66174.mem)[i_65911 *
                                                              k2p2zq_65572 +
                                                              i_65914];
                    float x_65700 = x_65698 * x_65699;
                    bool res_65701;
                    
                    res_65701 = futrts_isnan32(x_65697);
                    
                    float y_65702;
                    
                    if (res_65701) {
                        y_65702 = 0.0F;
                    } else {
                        y_65702 = 1.0F;
                    }
                    
                    float res_65703 = x_65700 * y_65702;
                    float res_65696 = res_65703 + redout_65910;
                    float redout_tmp_66402 = res_65696;
                    
                    redout_65910 = redout_tmp_66402;
                }
                res_65693 = redout_65910;
                ((float *) mem_66191.mem)[i_65922 * (k2p2zq_65572 *
                                                     k2p2zq_65572) + i_65918 *
                                          k2p2zq_65572 + i_65914] = res_65693;
            }
        }
    }
    
    int32_t j_65705 = 2 * k2p2zq_65572;
    int32_t j_m_i_65706 = j_65705 - k2p2zq_65572;
    int32_t nm_65709 = k2p2zq_65572 * j_65705;
    bool empty_slice_65722 = j_m_i_65706 == 0;
    int32_t m_65723 = j_m_i_65706 - 1;
    int32_t i_p_m_t_s_65724 = k2p2zq_65572 + m_65723;
    bool zzero_leq_i_p_m_t_s_65725 = sle32(0, i_p_m_t_s_65724);
    bool ok_or_empty_65732 = empty_slice_65722 || zzero_leq_i_p_m_t_s_65725;
    bool index_certs_65734;
    
    if (!ok_or_empty_65732) {
        ctx->error = msgprintf("Error at %s:\n%s%d%s%d%s%d%s%d%s%d%s%d%s\n",
                               "bfast-irreg.fut:133:1-316:68 -> bfast-irreg.fut:168:14-29 -> bfast-irreg.fut:109:8-37",
                               "Index [", 0, ":", k2p2zq_65572, ", ",
                               k2p2zq_65572, ":", j_65705,
                               "] out of bounds for array of shape [",
                               k2p2zq_65572, "][", j_65705, "].");
        if (memblock_unref(ctx, &mem_66191, "mem_66191") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_66174, "mem_66174") != 0)
            return 1;
        if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_66169,
                           "lifted_1_zlzb_arg_mem_66169") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66390, "out_mem_66390") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66387, "out_mem_66387") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66385, "out_mem_66385") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66382, "out_mem_66382") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66379, "out_mem_66379") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66376, "out_mem_66376") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66372, "out_mem_66372") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66368, "out_mem_66368") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66365, "out_mem_66365") != 0)
            return 1;
        return 1;
    }
    
    int64_t binop_y_66223 = sext_i32_i64(j_m_i_65706);
    int64_t binop_x_66224 = binop_x_66188 * binop_y_66223;
    int64_t bytes_66219 = 4 * binop_x_66224;
    struct memblock mem_66225;
    
    mem_66225.references = NULL;
    if (memblock_alloc(ctx, &mem_66225, bytes_66219, "mem_66225"))
        return 1;
    
    int64_t binop_x_66228 = sext_i32_i64(nm_65709);
    int64_t bytes_66227 = 4 * binop_x_66228;
    struct memblock mem_66229;
    
    mem_66229.references = NULL;
    if (memblock_alloc(ctx, &mem_66229, bytes_66227, "mem_66229"))
        return 1;
    
    struct memblock mem_66235;
    
    mem_66235.references = NULL;
    if (memblock_alloc(ctx, &mem_66235, bytes_66227, "mem_66235"))
        return 1;
    for (int32_t i_65944 = 0; i_65944 < m_65549; i_65944++) {
        for (int32_t i_65926 = 0; i_65926 < nm_65709; i_65926++) {
            int32_t res_65739 = sdiv32(i_65926, j_65705);
            int32_t res_65740 = smod32(i_65926, j_65705);
            bool cond_65741 = slt32(res_65740, k2p2zq_65572);
            float res_65742;
            
            if (cond_65741) {
                float res_65743 = ((float *) mem_66191.mem)[i_65944 *
                                                            (k2p2zq_65572 *
                                                             k2p2zq_65572) +
                                                            res_65739 *
                                                            k2p2zq_65572 +
                                                            res_65740];
                
                res_65742 = res_65743;
            } else {
                int32_t y_65744 = k2p2zq_65572 + res_65739;
                bool cond_65745 = res_65740 == y_65744;
                float res_65746;
                
                if (cond_65745) {
                    res_65746 = 1.0F;
                } else {
                    res_65746 = 0.0F;
                }
                res_65742 = res_65746;
            }
            ((float *) mem_66229.mem)[i_65926] = res_65742;
        }
        for (int32_t i_65749 = 0; i_65749 < k2p2zq_65572; i_65749++) {
            float v1_65754 = ((float *) mem_66229.mem)[i_65749];
            bool cond_65755 = v1_65754 == 0.0F;
            
            for (int32_t i_65930 = 0; i_65930 < nm_65709; i_65930++) {
                int32_t res_65758 = sdiv32(i_65930, j_65705);
                int32_t res_65759 = smod32(i_65930, j_65705);
                float res_65760;
                
                if (cond_65755) {
                    int32_t x_65761 = j_65705 * res_65758;
                    int32_t i_65762 = res_65759 + x_65761;
                    float res_65763 = ((float *) mem_66229.mem)[i_65762];
                    
                    res_65760 = res_65763;
                } else {
                    float x_65764 = ((float *) mem_66229.mem)[res_65759];
                    float x_65765 = x_65764 / v1_65754;
                    bool cond_65766 = slt32(res_65758, m_65655);
                    float res_65767;
                    
                    if (cond_65766) {
                        int32_t x_65768 = 1 + res_65758;
                        int32_t x_65769 = j_65705 * x_65768;
                        int32_t i_65770 = res_65759 + x_65769;
                        float x_65771 = ((float *) mem_66229.mem)[i_65770];
                        int32_t i_65772 = i_65749 + x_65769;
                        float x_65773 = ((float *) mem_66229.mem)[i_65772];
                        float y_65774 = x_65765 * x_65773;
                        float res_65775 = x_65771 - y_65774;
                        
                        res_65767 = res_65775;
                    } else {
                        res_65767 = x_65765;
                    }
                    res_65760 = res_65767;
                }
                ((float *) mem_66235.mem)[i_65930] = res_65760;
            }
            for (int32_t write_iter_65932 = 0; write_iter_65932 < nm_65709;
                 write_iter_65932++) {
                bool less_than_zzero_65936 = slt32(write_iter_65932, 0);
                bool greater_than_sizze_65937 = sle32(nm_65709,
                                                      write_iter_65932);
                bool outside_bounds_dim_65938 = less_than_zzero_65936 ||
                     greater_than_sizze_65937;
                
                if (!outside_bounds_dim_65938) {
                    memmove(mem_66229.mem + write_iter_65932 * 4,
                            mem_66235.mem + write_iter_65932 * 4,
                            sizeof(float));
                }
            }
        }
        for (int32_t i_66408 = 0; i_66408 < k2p2zq_65572; i_66408++) {
            for (int32_t i_66409 = 0; i_66409 < j_m_i_65706; i_66409++) {
                ((float *) mem_66225.mem)[i_65944 * (j_m_i_65706 *
                                                     k2p2zq_65572) + (i_66408 *
                                                                      j_m_i_65706 +
                                                                      i_66409)] =
                    ((float *) mem_66229.mem)[k2p2zq_65572 + (i_66408 *
                                                              j_65705 +
                                                              i_66409)];
            }
        }
    }
    if (memblock_unref(ctx, &mem_66229, "mem_66229") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_66235, "mem_66235") != 0)
        return 1;
    
    int64_t bytes_66242 = 4 * binop_x_66188;
    struct memblock mem_66246;
    
    mem_66246.references = NULL;
    if (memblock_alloc(ctx, &mem_66246, bytes_66242, "mem_66246"))
        return 1;
    for (int32_t i_65954 = 0; i_65954 < m_65549; i_65954++) {
        for (int32_t i_65950 = 0; i_65950 < k2p2zq_65572; i_65950++) {
            float res_65786;
            float redout_65946 = 0.0F;
            
            for (int32_t i_65947 = 0; i_65947 < n_65553; i_65947++) {
                float x_65790 =
                      ((float *) lifted_1_zlzb_arg_mem_66169.mem)[i_65950 *
                                                                  N_65548 +
                                                                  i_65947];
                float x_65791 = ((float *) images_mem_66138.mem)[i_65954 *
                                                                 N_65550 +
                                                                 i_65947];
                bool res_65792;
                
                res_65792 = futrts_isnan32(x_65791);
                
                float res_65793;
                
                if (res_65792) {
                    res_65793 = 0.0F;
                } else {
                    float res_65794 = x_65790 * x_65791;
                    
                    res_65793 = res_65794;
                }
                
                float res_65789 = res_65793 + redout_65946;
                float redout_tmp_66412 = res_65789;
                
                redout_65946 = redout_tmp_66412;
            }
            res_65786 = redout_65946;
            ((float *) mem_66246.mem)[i_65954 * k2p2zq_65572 + i_65950] =
                res_65786;
        }
    }
    
    struct memblock mem_66261;
    
    mem_66261.references = NULL;
    if (memblock_alloc(ctx, &mem_66261, bytes_66242, "mem_66261"))
        return 1;
    for (int32_t i_65964 = 0; i_65964 < m_65549; i_65964++) {
        for (int32_t i_65960 = 0; i_65960 < k2p2zq_65572; i_65960++) {
            float res_65806;
            float redout_65956 = 0.0F;
            
            for (int32_t i_65957 = 0; i_65957 < j_m_i_65706; i_65957++) {
                float x_65810 = ((float *) mem_66246.mem)[i_65964 *
                                                          k2p2zq_65572 +
                                                          i_65957];
                float x_65811 = ((float *) mem_66225.mem)[i_65964 *
                                                          (j_m_i_65706 *
                                                           k2p2zq_65572) +
                                                          i_65960 *
                                                          j_m_i_65706 +
                                                          i_65957];
                float res_65812 = x_65810 * x_65811;
                float res_65809 = res_65812 + redout_65956;
                float redout_tmp_66415 = res_65809;
                
                redout_65956 = redout_tmp_66415;
            }
            res_65806 = redout_65956;
            ((float *) mem_66261.mem)[i_65964 * k2p2zq_65572 + i_65960] =
                res_65806;
        }
    }
    
    int64_t binop_x_66275 = binop_x_66171 * binop_x_66186;
    int64_t bytes_66272 = 4 * binop_x_66275;
    struct memblock mem_66276;
    
    mem_66276.references = NULL;
    if (memblock_alloc(ctx, &mem_66276, bytes_66272, "mem_66276"))
        return 1;
    for (int32_t i_65974 = 0; i_65974 < m_65549; i_65974++) {
        for (int32_t i_65970 = 0; i_65970 < N_65548; i_65970++) {
            float res_65818;
            float redout_65966 = 0.0F;
            
            for (int32_t i_65967 = 0; i_65967 < k2p2zq_65572; i_65967++) {
                float x_65822 = ((float *) mem_66261.mem)[i_65974 *
                                                          k2p2zq_65572 +
                                                          i_65967];
                float x_65823 = ((float *) mem_66174.mem)[i_65970 *
                                                          k2p2zq_65572 +
                                                          i_65967];
                float res_65824 = x_65822 * x_65823;
                float res_65821 = res_65824 + redout_65966;
                float redout_tmp_66418 = res_65821;
                
                redout_65966 = redout_tmp_66418;
            }
            res_65818 = redout_65966;
            ((float *) mem_66276.mem)[i_65974 * N_65548 + i_65970] = res_65818;
        }
    }
    if (memblock_unref(ctx, &mem_66174, "mem_66174") != 0)
        return 1;
    
    int32_t i_65826 = N_65548 - 1;
    bool x_65827 = sle32(0, i_65826);
    bool index_certs_65830;
    
    if (!x_65827) {
        ctx->error = msgprintf("Error at %s:\n%s%d%s%d%s\n",
                               "bfast-irreg.fut:133:1-316:68 -> bfast-irreg.fut:189:5-198:25 -> /futlib/soacs.fut:51:3-37 -> /futlib/soacs.fut:51:19-23 -> bfast-irreg.fut:194:30-91 -> bfast-irreg.fut:37:13-20 -> /futlib/array.fut:18:29-34",
                               "Index [", i_65826,
                               "] out of bounds for array of shape [", N_65548,
                               "].");
        if (memblock_unref(ctx, &mem_66276, "mem_66276") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_66261, "mem_66261") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_66246, "mem_66246") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_66235, "mem_66235") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_66229, "mem_66229") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_66225, "mem_66225") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_66191, "mem_66191") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_66174, "mem_66174") != 0)
            return 1;
        if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_66169,
                           "lifted_1_zlzb_arg_mem_66169") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66390, "out_mem_66390") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66387, "out_mem_66387") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66385, "out_mem_66385") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66382, "out_mem_66382") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66379, "out_mem_66379") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66376, "out_mem_66376") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66372, "out_mem_66372") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66368, "out_mem_66368") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_66365, "out_mem_66365") != 0)
            return 1;
        return 1;
    }
    
    int64_t bytes_66287 = 4 * binop_x_66186;
    struct memblock mem_66289;
    
    mem_66289.references = NULL;
    if (memblock_alloc(ctx, &mem_66289, bytes_66287, "mem_66289"))
        return 1;
    
    struct memblock mem_66294;
    
    mem_66294.references = NULL;
    if (memblock_alloc(ctx, &mem_66294, bytes_66272, "mem_66294"))
        return 1;
    
    struct memblock mem_66299;
    
    mem_66299.references = NULL;
    if (memblock_alloc(ctx, &mem_66299, bytes_66272, "mem_66299"))
        return 1;
    
    int64_t bytes_66303 = 4 * binop_x_66171;
    struct memblock mem_66305;
    
    mem_66305.references = NULL;
    if (memblock_alloc(ctx, &mem_66305, bytes_66303, "mem_66305"))
        return 1;
    
    struct memblock mem_66308;
    
    mem_66308.references = NULL;
    if (memblock_alloc(ctx, &mem_66308, bytes_66303, "mem_66308"))
        return 1;
    
    struct memblock mem_66315;
    
    mem_66315.references = NULL;
    if (memblock_alloc(ctx, &mem_66315, bytes_66303, "mem_66315"))
        return 1;
    for (int32_t i_66419 = 0; i_66419 < N_65548; i_66419++) {
        ((float *) mem_66315.mem)[i_66419] = NAN;
    }
    
    struct memblock mem_66318;
    
    mem_66318.references = NULL;
    if (memblock_alloc(ctx, &mem_66318, bytes_66303, "mem_66318"))
        return 1;
    for (int32_t i_66420 = 0; i_66420 < N_65548; i_66420++) {
        ((int32_t *) mem_66318.mem)[i_66420] = 0;
    }
    
    struct memblock mem_66323;
    
    mem_66323.references = NULL;
    if (memblock_alloc(ctx, &mem_66323, bytes_66303, "mem_66323"))
        return 1;
    
    struct memblock mem_66333;
    
    mem_66333.references = NULL;
    if (memblock_alloc(ctx, &mem_66333, bytes_66303, "mem_66333"))
        return 1;
    for (int32_t i_66009 = 0; i_66009 < m_65549; i_66009++) {
        int32_t discard_65984;
        int32_t scanacc_65978 = 0;
        
        for (int32_t i_65981 = 0; i_65981 < N_65548; i_65981++) {
            float x_65860 = ((float *) images_mem_66138.mem)[i_66009 * N_65550 +
                                                             i_65981];
            float x_65861 = ((float *) mem_66276.mem)[i_66009 * N_65548 +
                                                      i_65981];
            bool res_65862;
            
            res_65862 = futrts_isnan32(x_65860);
            
            bool cond_65863 = !res_65862;
            float res_65864;
            
            if (cond_65863) {
                float res_65865 = x_65860 - x_65861;
                
                res_65864 = res_65865;
            } else {
                res_65864 = NAN;
            }
            
            bool res_65866;
            
            res_65866 = futrts_isnan32(res_65864);
            
            bool res_65867 = !res_65866;
            int32_t res_65868;
            
            if (res_65867) {
                res_65868 = 1;
            } else {
                res_65868 = 0;
            }
            
            int32_t res_65859 = res_65868 + scanacc_65978;
            
            ((int32_t *) mem_66305.mem)[i_65981] = res_65859;
            ((float *) mem_66308.mem)[i_65981] = res_65864;
            
            int32_t scanacc_tmp_66424 = res_65859;
            
            scanacc_65978 = scanacc_tmp_66424;
        }
        discard_65984 = scanacc_65978;
        memmove(mem_66294.mem + i_66009 * N_65548 * 4, mem_66315.mem + 0,
                N_65548 * sizeof(float));
        memmove(mem_66299.mem + i_66009 * N_65548 * 4, mem_66318.mem + 0,
                N_65548 * sizeof(int32_t));
        for (int32_t write_iter_65985 = 0; write_iter_65985 < N_65548;
             write_iter_65985++) {
            float write_iv_65988 = ((float *) mem_66308.mem)[write_iter_65985];
            int32_t write_iv_65989 =
                    ((int32_t *) mem_66305.mem)[write_iter_65985];
            bool res_65879;
            
            res_65879 = futrts_isnan32(write_iv_65988);
            
            bool res_65880 = !res_65879;
            int32_t res_65881;
            
            if (res_65880) {
                int32_t res_65882 = write_iv_65989 - 1;
                
                res_65881 = res_65882;
            } else {
                res_65881 = -1;
            }
            
            bool less_than_zzero_65991 = slt32(res_65881, 0);
            bool greater_than_sizze_65992 = sle32(N_65548, res_65881);
            bool outside_bounds_dim_65993 = less_than_zzero_65991 ||
                 greater_than_sizze_65992;
            
            memmove(mem_66323.mem + 0, mem_66299.mem + i_66009 * N_65548 * 4,
                    N_65548 * sizeof(int32_t));
            
            struct memblock write_out_mem_66330;
            
            write_out_mem_66330.references = NULL;
            if (outside_bounds_dim_65993) {
                if (memblock_set(ctx, &write_out_mem_66330, &mem_66323,
                                 "mem_66323") != 0)
                    return 1;
            } else {
                struct memblock mem_66326;
                
                mem_66326.references = NULL;
                if (memblock_alloc(ctx, &mem_66326, 4, "mem_66326"))
                    return 1;
                
                int32_t x_66430;
                
                for (int32_t i_66429 = 0; i_66429 < 1; i_66429++) {
                    x_66430 = write_iter_65985 + sext_i32_i32(i_66429);
                    ((int32_t *) mem_66326.mem)[i_66429] = x_66430;
                }
                
                struct memblock mem_66329;
                
                mem_66329.references = NULL;
                if (memblock_alloc(ctx, &mem_66329, bytes_66303, "mem_66329"))
                    return 1;
                memmove(mem_66329.mem + 0, mem_66299.mem + i_66009 * N_65548 *
                        4, N_65548 * sizeof(int32_t));
                memmove(mem_66329.mem + res_65881 * 4, mem_66326.mem + 0,
                        sizeof(int32_t));
                if (memblock_unref(ctx, &mem_66326, "mem_66326") != 0)
                    return 1;
                if (memblock_set(ctx, &write_out_mem_66330, &mem_66329,
                                 "mem_66329") != 0)
                    return 1;
                if (memblock_unref(ctx, &mem_66329, "mem_66329") != 0)
                    return 1;
                if (memblock_unref(ctx, &mem_66326, "mem_66326") != 0)
                    return 1;
            }
            memmove(mem_66299.mem + i_66009 * N_65548 * 4,
                    write_out_mem_66330.mem + 0, N_65548 * sizeof(int32_t));
            if (memblock_unref(ctx, &write_out_mem_66330,
                               "write_out_mem_66330") != 0)
                return 1;
            memmove(mem_66333.mem + 0, mem_66294.mem + i_66009 * N_65548 * 4,
                    N_65548 * sizeof(float));
            
            struct memblock write_out_mem_66337;
            
            write_out_mem_66337.references = NULL;
            if (outside_bounds_dim_65993) {
                if (memblock_set(ctx, &write_out_mem_66337, &mem_66333,
                                 "mem_66333") != 0)
                    return 1;
            } else {
                struct memblock mem_66336;
                
                mem_66336.references = NULL;
                if (memblock_alloc(ctx, &mem_66336, bytes_66303, "mem_66336"))
                    return 1;
                memmove(mem_66336.mem + 0, mem_66294.mem + i_66009 * N_65548 *
                        4, N_65548 * sizeof(float));
                memmove(mem_66336.mem + res_65881 * 4, mem_66308.mem +
                        write_iter_65985 * 4, sizeof(float));
                if (memblock_set(ctx, &write_out_mem_66337, &mem_66336,
                                 "mem_66336") != 0)
                    return 1;
                if (memblock_unref(ctx, &mem_66336, "mem_66336") != 0)
                    return 1;
            }
            memmove(mem_66294.mem + i_66009 * N_65548 * 4,
                    write_out_mem_66337.mem + 0, N_65548 * sizeof(float));
            if (memblock_unref(ctx, &write_out_mem_66337,
                               "write_out_mem_66337") != 0)
                return 1;
            if (memblock_unref(ctx, &write_out_mem_66337,
                               "write_out_mem_66337") != 0)
                return 1;
            if (memblock_unref(ctx, &write_out_mem_66330,
                               "write_out_mem_66330") != 0)
                return 1;
        }
        memmove(mem_66289.mem + i_66009 * 4, mem_66305.mem + i_65826 * 4,
                sizeof(int32_t));
    }
    if (memblock_unref(ctx, &mem_66305, "mem_66305") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_66308, "mem_66308") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_66315, "mem_66315") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_66318, "mem_66318") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_66323, "mem_66323") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_66333, "mem_66333") != 0)
        return 1;
    out_arrsizze_66366 = k2p2zq_65572;
    out_arrsizze_66367 = N_65548;
    out_arrsizze_66369 = m_65549;
    out_arrsizze_66370 = k2p2zq_65572;
    out_arrsizze_66371 = k2p2zq_65572;
    out_arrsizze_66373 = m_65549;
    out_arrsizze_66374 = k2p2zq_65572;
    out_arrsizze_66375 = j_m_i_65706;
    out_arrsizze_66377 = m_65549;
    out_arrsizze_66378 = k2p2zq_65572;
    out_arrsizze_66380 = m_65549;
    out_arrsizze_66381 = k2p2zq_65572;
    out_arrsizze_66383 = m_65549;
    out_arrsizze_66384 = N_65548;
    out_arrsizze_66386 = m_65549;
    out_arrsizze_66388 = m_65549;
    out_arrsizze_66389 = N_65548;
    out_arrsizze_66391 = m_65549;
    out_arrsizze_66392 = N_65548;
    if (memblock_set(ctx, &out_mem_66365, &lifted_1_zlzb_arg_mem_66169,
                     "lifted_1_zlzb_arg_mem_66169") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_66368, &mem_66191, "mem_66191") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_66372, &mem_66225, "mem_66225") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_66376, &mem_66246, "mem_66246") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_66379, &mem_66261, "mem_66261") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_66382, &mem_66276, "mem_66276") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_66385, &mem_66289, "mem_66289") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_66387, &mem_66294, "mem_66294") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_66390, &mem_66299, "mem_66299") != 0)
        return 1;
    (*out_mem_p_66431).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_66431, &out_mem_66365, "out_mem_66365") !=
        0)
        return 1;
    *out_out_arrsizze_66432 = out_arrsizze_66366;
    *out_out_arrsizze_66433 = out_arrsizze_66367;
    (*out_mem_p_66434).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_66434, &out_mem_66368, "out_mem_66368") !=
        0)
        return 1;
    *out_out_arrsizze_66435 = out_arrsizze_66369;
    *out_out_arrsizze_66436 = out_arrsizze_66370;
    *out_out_arrsizze_66437 = out_arrsizze_66371;
    (*out_mem_p_66438).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_66438, &out_mem_66372, "out_mem_66372") !=
        0)
        return 1;
    *out_out_arrsizze_66439 = out_arrsizze_66373;
    *out_out_arrsizze_66440 = out_arrsizze_66374;
    *out_out_arrsizze_66441 = out_arrsizze_66375;
    (*out_mem_p_66442).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_66442, &out_mem_66376, "out_mem_66376") !=
        0)
        return 1;
    *out_out_arrsizze_66443 = out_arrsizze_66377;
    *out_out_arrsizze_66444 = out_arrsizze_66378;
    (*out_mem_p_66445).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_66445, &out_mem_66379, "out_mem_66379") !=
        0)
        return 1;
    *out_out_arrsizze_66446 = out_arrsizze_66380;
    *out_out_arrsizze_66447 = out_arrsizze_66381;
    (*out_mem_p_66448).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_66448, &out_mem_66382, "out_mem_66382") !=
        0)
        return 1;
    *out_out_arrsizze_66449 = out_arrsizze_66383;
    *out_out_arrsizze_66450 = out_arrsizze_66384;
    (*out_mem_p_66451).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_66451, &out_mem_66385, "out_mem_66385") !=
        0)
        return 1;
    *out_out_arrsizze_66452 = out_arrsizze_66386;
    (*out_mem_p_66453).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_66453, &out_mem_66387, "out_mem_66387") !=
        0)
        return 1;
    *out_out_arrsizze_66454 = out_arrsizze_66388;
    *out_out_arrsizze_66455 = out_arrsizze_66389;
    (*out_mem_p_66456).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_66456, &out_mem_66390, "out_mem_66390") !=
        0)
        return 1;
    *out_out_arrsizze_66457 = out_arrsizze_66391;
    *out_out_arrsizze_66458 = out_arrsizze_66392;
    if (memblock_unref(ctx, &mem_66333, "mem_66333") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_66323, "mem_66323") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_66318, "mem_66318") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_66315, "mem_66315") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_66308, "mem_66308") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_66305, "mem_66305") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_66299, "mem_66299") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_66294, "mem_66294") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_66289, "mem_66289") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_66276, "mem_66276") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_66261, "mem_66261") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_66246, "mem_66246") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_66235, "mem_66235") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_66229, "mem_66229") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_66225, "mem_66225") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_66191, "mem_66191") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_66174, "mem_66174") != 0)
        return 1;
    if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_66169,
                       "lifted_1_zlzb_arg_mem_66169") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_66390, "out_mem_66390") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_66387, "out_mem_66387") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_66385, "out_mem_66385") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_66382, "out_mem_66382") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_66379, "out_mem_66379") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_66376, "out_mem_66376") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_66372, "out_mem_66372") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_66368, "out_mem_66368") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_66365, "out_mem_66365") != 0)
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
                       struct futhark_f32_3d **out1,
                       struct futhark_f32_3d **out2,
                       struct futhark_f32_2d **out3,
                       struct futhark_f32_2d **out4,
                       struct futhark_f32_2d **out5,
                       struct futhark_i32_1d **out6,
                       struct futhark_f32_2d **out7,
                       struct futhark_i32_2d **out8, const int32_t in0, const
                       int32_t in1, const int32_t in2, const float in3, const
                       float in4, const float in5, const
                       struct futhark_i32_1d *in6, const
                       struct futhark_f32_2d *in7)
{
    struct memblock mappingindices_mem_66137;
    
    mappingindices_mem_66137.references = NULL;
    
    struct memblock images_mem_66138;
    
    images_mem_66138.references = NULL;
    
    int32_t N_65548;
    int32_t m_65549;
    int32_t N_65550;
    int32_t trend_65551;
    int32_t k_65552;
    int32_t n_65553;
    float freq_65554;
    float hfrac_65555;
    float lam_65556;
    struct memblock out_mem_66365;
    
    out_mem_66365.references = NULL;
    
    int32_t out_arrsizze_66366;
    int32_t out_arrsizze_66367;
    struct memblock out_mem_66368;
    
    out_mem_66368.references = NULL;
    
    int32_t out_arrsizze_66369;
    int32_t out_arrsizze_66370;
    int32_t out_arrsizze_66371;
    struct memblock out_mem_66372;
    
    out_mem_66372.references = NULL;
    
    int32_t out_arrsizze_66373;
    int32_t out_arrsizze_66374;
    int32_t out_arrsizze_66375;
    struct memblock out_mem_66376;
    
    out_mem_66376.references = NULL;
    
    int32_t out_arrsizze_66377;
    int32_t out_arrsizze_66378;
    struct memblock out_mem_66379;
    
    out_mem_66379.references = NULL;
    
    int32_t out_arrsizze_66380;
    int32_t out_arrsizze_66381;
    struct memblock out_mem_66382;
    
    out_mem_66382.references = NULL;
    
    int32_t out_arrsizze_66383;
    int32_t out_arrsizze_66384;
    struct memblock out_mem_66385;
    
    out_mem_66385.references = NULL;
    
    int32_t out_arrsizze_66386;
    struct memblock out_mem_66387;
    
    out_mem_66387.references = NULL;
    
    int32_t out_arrsizze_66388;
    int32_t out_arrsizze_66389;
    struct memblock out_mem_66390;
    
    out_mem_66390.references = NULL;
    
    int32_t out_arrsizze_66391;
    int32_t out_arrsizze_66392;
    
    lock_lock(&ctx->lock);
    trend_65551 = in0;
    k_65552 = in1;
    n_65553 = in2;
    freq_65554 = in3;
    hfrac_65555 = in4;
    lam_65556 = in5;
    mappingindices_mem_66137 = in6->mem;
    N_65548 = in6->shape[0];
    images_mem_66138 = in7->mem;
    m_65549 = in7->shape[0];
    N_65550 = in7->shape[1];
    
    int ret = futrts_main(ctx, &out_mem_66365, &out_arrsizze_66366,
                          &out_arrsizze_66367, &out_mem_66368,
                          &out_arrsizze_66369, &out_arrsizze_66370,
                          &out_arrsizze_66371, &out_mem_66372,
                          &out_arrsizze_66373, &out_arrsizze_66374,
                          &out_arrsizze_66375, &out_mem_66376,
                          &out_arrsizze_66377, &out_arrsizze_66378,
                          &out_mem_66379, &out_arrsizze_66380,
                          &out_arrsizze_66381, &out_mem_66382,
                          &out_arrsizze_66383, &out_arrsizze_66384,
                          &out_mem_66385, &out_arrsizze_66386, &out_mem_66387,
                          &out_arrsizze_66388, &out_arrsizze_66389,
                          &out_mem_66390, &out_arrsizze_66391,
                          &out_arrsizze_66392, mappingindices_mem_66137,
                          images_mem_66138, N_65548, m_65549, N_65550,
                          trend_65551, k_65552, n_65553, freq_65554,
                          hfrac_65555, lam_65556);
    
    if (ret == 0) {
        assert((*out0 =
                (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) !=
            NULL);
        (*out0)->mem = out_mem_66365;
        (*out0)->shape[0] = out_arrsizze_66366;
        (*out0)->shape[1] = out_arrsizze_66367;
        assert((*out1 =
                (struct futhark_f32_3d *) malloc(sizeof(struct futhark_f32_3d))) !=
            NULL);
        (*out1)->mem = out_mem_66368;
        (*out1)->shape[0] = out_arrsizze_66369;
        (*out1)->shape[1] = out_arrsizze_66370;
        (*out1)->shape[2] = out_arrsizze_66371;
        assert((*out2 =
                (struct futhark_f32_3d *) malloc(sizeof(struct futhark_f32_3d))) !=
            NULL);
        (*out2)->mem = out_mem_66372;
        (*out2)->shape[0] = out_arrsizze_66373;
        (*out2)->shape[1] = out_arrsizze_66374;
        (*out2)->shape[2] = out_arrsizze_66375;
        assert((*out3 =
                (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) !=
            NULL);
        (*out3)->mem = out_mem_66376;
        (*out3)->shape[0] = out_arrsizze_66377;
        (*out3)->shape[1] = out_arrsizze_66378;
        assert((*out4 =
                (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) !=
            NULL);
        (*out4)->mem = out_mem_66379;
        (*out4)->shape[0] = out_arrsizze_66380;
        (*out4)->shape[1] = out_arrsizze_66381;
        assert((*out5 =
                (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) !=
            NULL);
        (*out5)->mem = out_mem_66382;
        (*out5)->shape[0] = out_arrsizze_66383;
        (*out5)->shape[1] = out_arrsizze_66384;
        assert((*out6 =
                (struct futhark_i32_1d *) malloc(sizeof(struct futhark_i32_1d))) !=
            NULL);
        (*out6)->mem = out_mem_66385;
        (*out6)->shape[0] = out_arrsizze_66386;
        assert((*out7 =
                (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) !=
            NULL);
        (*out7)->mem = out_mem_66387;
        (*out7)->shape[0] = out_arrsizze_66388;
        (*out7)->shape[1] = out_arrsizze_66389;
        assert((*out8 =
                (struct futhark_i32_2d *) malloc(sizeof(struct futhark_i32_2d))) !=
            NULL);
        (*out8)->mem = out_mem_66390;
        (*out8)->shape[0] = out_arrsizze_66391;
        (*out8)->shape[1] = out_arrsizze_66392;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
