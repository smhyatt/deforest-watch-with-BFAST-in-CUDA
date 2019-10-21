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

int futhark_entry_main(struct futhark_context *ctx, bool *out0, bool *out1,
                       bool *out2, bool *out3, bool *out4, bool *out5,
                       bool *out6, bool *out7, bool *out8, const
                       struct futhark_f32_2d *in0, const
                       struct futhark_f32_3d *in1, const
                       struct futhark_f32_3d *in2, const
                       struct futhark_f32_2d *in3, const
                       struct futhark_f32_2d *in4, const
                       struct futhark_f32_2d *in5, const
                       struct futhark_i32_1d *in6, const
                       struct futhark_f32_2d *in7, const
                       struct futhark_i32_2d *in8, const
                       struct futhark_f32_2d *in9, const
                       struct futhark_f32_3d *in10, const
                       struct futhark_f32_3d *in11, const
                       struct futhark_f32_2d *in12, const
                       struct futhark_f32_2d *in13, const
                       struct futhark_f32_2d *in14, const
                       struct futhark_i32_1d *in15, const
                       struct futhark_f32_2d *in16, const
                       struct futhark_i32_2d *in17);

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
    
    struct futhark_f32_2d *read_value_26688;
    int64_t read_shape_26689[2];
    float *read_arr_26690 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_26690, read_shape_26689, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_3d *read_value_26691;
    int64_t read_shape_26692[3];
    float *read_arr_26693 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_26693, read_shape_26692, 3) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1,
              "[][][]", f32_info.type_name, strerror(errno));
    
    struct futhark_f32_3d *read_value_26694;
    int64_t read_shape_26695[3];
    float *read_arr_26696 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_26696, read_shape_26695, 3) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 2,
              "[][][]", f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_26697;
    int64_t read_shape_26698[2];
    float *read_arr_26699 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_26699, read_shape_26698, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 3, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_26700;
    int64_t read_shape_26701[2];
    float *read_arr_26702 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_26702, read_shape_26701, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 4, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_26703;
    int64_t read_shape_26704[2];
    float *read_arr_26705 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_26705, read_shape_26704, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 5, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_i32_1d *read_value_26706;
    int64_t read_shape_26707[1];
    int32_t *read_arr_26708 = NULL;
    
    errno = 0;
    if (read_array(&i32_info, (void **) &read_arr_26708, read_shape_26707, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 6, "[]",
              i32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_26709;
    int64_t read_shape_26710[2];
    float *read_arr_26711 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_26711, read_shape_26710, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 7, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_i32_2d *read_value_26712;
    int64_t read_shape_26713[2];
    int32_t *read_arr_26714 = NULL;
    
    errno = 0;
    if (read_array(&i32_info, (void **) &read_arr_26714, read_shape_26713, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 8, "[][]",
              i32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_26715;
    int64_t read_shape_26716[2];
    float *read_arr_26717 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_26717, read_shape_26716, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 9, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_3d *read_value_26718;
    int64_t read_shape_26719[3];
    float *read_arr_26720 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_26720, read_shape_26719, 3) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 10,
              "[][][]", f32_info.type_name, strerror(errno));
    
    struct futhark_f32_3d *read_value_26721;
    int64_t read_shape_26722[3];
    float *read_arr_26723 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_26723, read_shape_26722, 3) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 11,
              "[][][]", f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_26724;
    int64_t read_shape_26725[2];
    float *read_arr_26726 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_26726, read_shape_26725, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 12,
              "[][]", f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_26727;
    int64_t read_shape_26728[2];
    float *read_arr_26729 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_26729, read_shape_26728, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 13,
              "[][]", f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_26730;
    int64_t read_shape_26731[2];
    float *read_arr_26732 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_26732, read_shape_26731, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 14,
              "[][]", f32_info.type_name, strerror(errno));
    
    struct futhark_i32_1d *read_value_26733;
    int64_t read_shape_26734[1];
    int32_t *read_arr_26735 = NULL;
    
    errno = 0;
    if (read_array(&i32_info, (void **) &read_arr_26735, read_shape_26734, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 15, "[]",
              i32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_26736;
    int64_t read_shape_26737[2];
    float *read_arr_26738 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_26738, read_shape_26737, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 16,
              "[][]", f32_info.type_name, strerror(errno));
    
    struct futhark_i32_2d *read_value_26739;
    int64_t read_shape_26740[2];
    int32_t *read_arr_26741 = NULL;
    
    errno = 0;
    if (read_array(&i32_info, (void **) &read_arr_26741, read_shape_26740, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 17,
              "[][]", i32_info.type_name, strerror(errno));
    
    bool result_26742;
    bool result_26743;
    bool result_26744;
    bool result_26745;
    bool result_26746;
    bool result_26747;
    bool result_26748;
    bool result_26749;
    bool result_26750;
    
    if (perform_warmup) {
        time_runs = 0;
        
        int r;
        
        assert((read_value_26688 = futhark_new_f32_2d(ctx, read_arr_26690,
                                                      read_shape_26689[0],
                                                      read_shape_26689[1])) !=
            0);
        assert((read_value_26691 = futhark_new_f32_3d(ctx, read_arr_26693,
                                                      read_shape_26692[0],
                                                      read_shape_26692[1],
                                                      read_shape_26692[2])) !=
            0);
        assert((read_value_26694 = futhark_new_f32_3d(ctx, read_arr_26696,
                                                      read_shape_26695[0],
                                                      read_shape_26695[1],
                                                      read_shape_26695[2])) !=
            0);
        assert((read_value_26697 = futhark_new_f32_2d(ctx, read_arr_26699,
                                                      read_shape_26698[0],
                                                      read_shape_26698[1])) !=
            0);
        assert((read_value_26700 = futhark_new_f32_2d(ctx, read_arr_26702,
                                                      read_shape_26701[0],
                                                      read_shape_26701[1])) !=
            0);
        assert((read_value_26703 = futhark_new_f32_2d(ctx, read_arr_26705,
                                                      read_shape_26704[0],
                                                      read_shape_26704[1])) !=
            0);
        assert((read_value_26706 = futhark_new_i32_1d(ctx, read_arr_26708,
                                                      read_shape_26707[0])) !=
            0);
        assert((read_value_26709 = futhark_new_f32_2d(ctx, read_arr_26711,
                                                      read_shape_26710[0],
                                                      read_shape_26710[1])) !=
            0);
        assert((read_value_26712 = futhark_new_i32_2d(ctx, read_arr_26714,
                                                      read_shape_26713[0],
                                                      read_shape_26713[1])) !=
            0);
        assert((read_value_26715 = futhark_new_f32_2d(ctx, read_arr_26717,
                                                      read_shape_26716[0],
                                                      read_shape_26716[1])) !=
            0);
        assert((read_value_26718 = futhark_new_f32_3d(ctx, read_arr_26720,
                                                      read_shape_26719[0],
                                                      read_shape_26719[1],
                                                      read_shape_26719[2])) !=
            0);
        assert((read_value_26721 = futhark_new_f32_3d(ctx, read_arr_26723,
                                                      read_shape_26722[0],
                                                      read_shape_26722[1],
                                                      read_shape_26722[2])) !=
            0);
        assert((read_value_26724 = futhark_new_f32_2d(ctx, read_arr_26726,
                                                      read_shape_26725[0],
                                                      read_shape_26725[1])) !=
            0);
        assert((read_value_26727 = futhark_new_f32_2d(ctx, read_arr_26729,
                                                      read_shape_26728[0],
                                                      read_shape_26728[1])) !=
            0);
        assert((read_value_26730 = futhark_new_f32_2d(ctx, read_arr_26732,
                                                      read_shape_26731[0],
                                                      read_shape_26731[1])) !=
            0);
        assert((read_value_26733 = futhark_new_i32_1d(ctx, read_arr_26735,
                                                      read_shape_26734[0])) !=
            0);
        assert((read_value_26736 = futhark_new_f32_2d(ctx, read_arr_26738,
                                                      read_shape_26737[0],
                                                      read_shape_26737[1])) !=
            0);
        assert((read_value_26739 = futhark_new_i32_2d(ctx, read_arr_26741,
                                                      read_shape_26740[0],
                                                      read_shape_26740[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_main(ctx, &result_26742, &result_26743, &result_26744,
                               &result_26745, &result_26746, &result_26747,
                               &result_26748, &result_26749, &result_26750,
                               read_value_26688, read_value_26691,
                               read_value_26694, read_value_26697,
                               read_value_26700, read_value_26703,
                               read_value_26706, read_value_26709,
                               read_value_26712, read_value_26715,
                               read_value_26718, read_value_26721,
                               read_value_26724, read_value_26727,
                               read_value_26730, read_value_26733,
                               read_value_26736, read_value_26739);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_2d(ctx, read_value_26688) == 0);
        assert(futhark_free_f32_3d(ctx, read_value_26691) == 0);
        assert(futhark_free_f32_3d(ctx, read_value_26694) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_26697) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_26700) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_26703) == 0);
        assert(futhark_free_i32_1d(ctx, read_value_26706) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_26709) == 0);
        assert(futhark_free_i32_2d(ctx, read_value_26712) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_26715) == 0);
        assert(futhark_free_f32_3d(ctx, read_value_26718) == 0);
        assert(futhark_free_f32_3d(ctx, read_value_26721) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_26724) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_26727) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_26730) == 0);
        assert(futhark_free_i32_1d(ctx, read_value_26733) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_26736) == 0);
        assert(futhark_free_i32_2d(ctx, read_value_26739) == 0);
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
        
        assert((read_value_26688 = futhark_new_f32_2d(ctx, read_arr_26690,
                                                      read_shape_26689[0],
                                                      read_shape_26689[1])) !=
            0);
        assert((read_value_26691 = futhark_new_f32_3d(ctx, read_arr_26693,
                                                      read_shape_26692[0],
                                                      read_shape_26692[1],
                                                      read_shape_26692[2])) !=
            0);
        assert((read_value_26694 = futhark_new_f32_3d(ctx, read_arr_26696,
                                                      read_shape_26695[0],
                                                      read_shape_26695[1],
                                                      read_shape_26695[2])) !=
            0);
        assert((read_value_26697 = futhark_new_f32_2d(ctx, read_arr_26699,
                                                      read_shape_26698[0],
                                                      read_shape_26698[1])) !=
            0);
        assert((read_value_26700 = futhark_new_f32_2d(ctx, read_arr_26702,
                                                      read_shape_26701[0],
                                                      read_shape_26701[1])) !=
            0);
        assert((read_value_26703 = futhark_new_f32_2d(ctx, read_arr_26705,
                                                      read_shape_26704[0],
                                                      read_shape_26704[1])) !=
            0);
        assert((read_value_26706 = futhark_new_i32_1d(ctx, read_arr_26708,
                                                      read_shape_26707[0])) !=
            0);
        assert((read_value_26709 = futhark_new_f32_2d(ctx, read_arr_26711,
                                                      read_shape_26710[0],
                                                      read_shape_26710[1])) !=
            0);
        assert((read_value_26712 = futhark_new_i32_2d(ctx, read_arr_26714,
                                                      read_shape_26713[0],
                                                      read_shape_26713[1])) !=
            0);
        assert((read_value_26715 = futhark_new_f32_2d(ctx, read_arr_26717,
                                                      read_shape_26716[0],
                                                      read_shape_26716[1])) !=
            0);
        assert((read_value_26718 = futhark_new_f32_3d(ctx, read_arr_26720,
                                                      read_shape_26719[0],
                                                      read_shape_26719[1],
                                                      read_shape_26719[2])) !=
            0);
        assert((read_value_26721 = futhark_new_f32_3d(ctx, read_arr_26723,
                                                      read_shape_26722[0],
                                                      read_shape_26722[1],
                                                      read_shape_26722[2])) !=
            0);
        assert((read_value_26724 = futhark_new_f32_2d(ctx, read_arr_26726,
                                                      read_shape_26725[0],
                                                      read_shape_26725[1])) !=
            0);
        assert((read_value_26727 = futhark_new_f32_2d(ctx, read_arr_26729,
                                                      read_shape_26728[0],
                                                      read_shape_26728[1])) !=
            0);
        assert((read_value_26730 = futhark_new_f32_2d(ctx, read_arr_26732,
                                                      read_shape_26731[0],
                                                      read_shape_26731[1])) !=
            0);
        assert((read_value_26733 = futhark_new_i32_1d(ctx, read_arr_26735,
                                                      read_shape_26734[0])) !=
            0);
        assert((read_value_26736 = futhark_new_f32_2d(ctx, read_arr_26738,
                                                      read_shape_26737[0],
                                                      read_shape_26737[1])) !=
            0);
        assert((read_value_26739 = futhark_new_i32_2d(ctx, read_arr_26741,
                                                      read_shape_26740[0],
                                                      read_shape_26740[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_main(ctx, &result_26742, &result_26743, &result_26744,
                               &result_26745, &result_26746, &result_26747,
                               &result_26748, &result_26749, &result_26750,
                               read_value_26688, read_value_26691,
                               read_value_26694, read_value_26697,
                               read_value_26700, read_value_26703,
                               read_value_26706, read_value_26709,
                               read_value_26712, read_value_26715,
                               read_value_26718, read_value_26721,
                               read_value_26724, read_value_26727,
                               read_value_26730, read_value_26733,
                               read_value_26736, read_value_26739);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_2d(ctx, read_value_26688) == 0);
        assert(futhark_free_f32_3d(ctx, read_value_26691) == 0);
        assert(futhark_free_f32_3d(ctx, read_value_26694) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_26697) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_26700) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_26703) == 0);
        assert(futhark_free_i32_1d(ctx, read_value_26706) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_26709) == 0);
        assert(futhark_free_i32_2d(ctx, read_value_26712) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_26715) == 0);
        assert(futhark_free_f32_3d(ctx, read_value_26718) == 0);
        assert(futhark_free_f32_3d(ctx, read_value_26721) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_26724) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_26727) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_26730) == 0);
        assert(futhark_free_i32_1d(ctx, read_value_26733) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_26736) == 0);
        assert(futhark_free_i32_2d(ctx, read_value_26739) == 0);
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
        }
    }
    free(read_arr_26690);
    free(read_arr_26693);
    free(read_arr_26696);
    free(read_arr_26699);
    free(read_arr_26702);
    free(read_arr_26705);
    free(read_arr_26708);
    free(read_arr_26711);
    free(read_arr_26714);
    free(read_arr_26717);
    free(read_arr_26720);
    free(read_arr_26723);
    free(read_arr_26726);
    free(read_arr_26729);
    free(read_arr_26732);
    free(read_arr_26735);
    free(read_arr_26738);
    free(read_arr_26741);
    if (binary_output)
        set_binary_mode(stdout);
    write_scalar(stdout, binary_output, &bool_info, &result_26742);
    printf("\n");
    write_scalar(stdout, binary_output, &bool_info, &result_26743);
    printf("\n");
    write_scalar(stdout, binary_output, &bool_info, &result_26744);
    printf("\n");
    write_scalar(stdout, binary_output, &bool_info, &result_26745);
    printf("\n");
    write_scalar(stdout, binary_output, &bool_info, &result_26746);
    printf("\n");
    write_scalar(stdout, binary_output, &bool_info, &result_26747);
    printf("\n");
    write_scalar(stdout, binary_output, &bool_info, &result_26748);
    printf("\n");
    write_scalar(stdout, binary_output, &bool_info, &result_26749);
    printf("\n");
    write_scalar(stdout, binary_output, &bool_info, &result_26750);
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
static int futrts_main(struct futhark_context *ctx, bool *out_scalar_out_26679,
                       bool *out_scalar_out_26680, bool *out_scalar_out_26681,
                       bool *out_scalar_out_26682, bool *out_scalar_out_26683,
                       bool *out_scalar_out_26684, bool *out_scalar_out_26685,
                       bool *out_scalar_out_26686, bool *out_scalar_out_26687,
                       struct memblock X_mem_26633,
                       struct memblock Xsqr_mem_26634,
                       struct memblock Xinv_mem_26635,
                       struct memblock beta0_mem_26636,
                       struct memblock beta_mem_26637,
                       struct memblock y_preds_mem_26638,
                       struct memblock Nss_mem_26639,
                       struct memblock y_errors_mem_26640,
                       struct memblock val_indss_mem_26641,
                       struct memblock Xseq_mem_26642,
                       struct memblock Xsqrseq_mem_26643,
                       struct memblock Xinvseq_mem_26644,
                       struct memblock beta0seq_mem_26645,
                       struct memblock betaseq_mem_26646,
                       struct memblock y_predsseq_mem_26647,
                       struct memblock Nssseq_mem_26648,
                       struct memblock y_errorsseq_mem_26649,
                       struct memblock val_indssseq_mem_26650,
                       int32_t sizze_26195, int32_t sizze_26196,
                       int32_t sizze_26197, int32_t sizze_26198,
                       int32_t sizze_26199, int32_t sizze_26200,
                       int32_t sizze_26201, int32_t sizze_26202,
                       int32_t sizze_26203, int32_t sizze_26204,
                       int32_t sizze_26205, int32_t sizze_26206,
                       int32_t sizze_26207, int32_t sizze_26208,
                       int32_t sizze_26209, int32_t sizze_26210,
                       int32_t sizze_26211, int32_t sizze_26212,
                       int32_t sizze_26213, int32_t sizze_26214,
                       int32_t sizze_26215, int32_t sizze_26216,
                       int32_t sizze_26217, int32_t sizze_26218,
                       int32_t sizze_26219, int32_t sizze_26220,
                       int32_t sizze_26221, int32_t sizze_26222,
                       int32_t sizze_26223, int32_t sizze_26224,
                       int32_t sizze_26225, int32_t sizze_26226,
                       int32_t sizze_26227, int32_t sizze_26228,
                       int32_t sizze_26229, int32_t sizze_26230,
                       int32_t sizze_26231, int32_t sizze_26232);
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
static int futrts_main(struct futhark_context *ctx, bool *out_scalar_out_26679,
                       bool *out_scalar_out_26680, bool *out_scalar_out_26681,
                       bool *out_scalar_out_26682, bool *out_scalar_out_26683,
                       bool *out_scalar_out_26684, bool *out_scalar_out_26685,
                       bool *out_scalar_out_26686, bool *out_scalar_out_26687,
                       struct memblock X_mem_26633,
                       struct memblock Xsqr_mem_26634,
                       struct memblock Xinv_mem_26635,
                       struct memblock beta0_mem_26636,
                       struct memblock beta_mem_26637,
                       struct memblock y_preds_mem_26638,
                       struct memblock Nss_mem_26639,
                       struct memblock y_errors_mem_26640,
                       struct memblock val_indss_mem_26641,
                       struct memblock Xseq_mem_26642,
                       struct memblock Xsqrseq_mem_26643,
                       struct memblock Xinvseq_mem_26644,
                       struct memblock beta0seq_mem_26645,
                       struct memblock betaseq_mem_26646,
                       struct memblock y_predsseq_mem_26647,
                       struct memblock Nssseq_mem_26648,
                       struct memblock y_errorsseq_mem_26649,
                       struct memblock val_indssseq_mem_26650,
                       int32_t sizze_26195, int32_t sizze_26196,
                       int32_t sizze_26197, int32_t sizze_26198,
                       int32_t sizze_26199, int32_t sizze_26200,
                       int32_t sizze_26201, int32_t sizze_26202,
                       int32_t sizze_26203, int32_t sizze_26204,
                       int32_t sizze_26205, int32_t sizze_26206,
                       int32_t sizze_26207, int32_t sizze_26208,
                       int32_t sizze_26209, int32_t sizze_26210,
                       int32_t sizze_26211, int32_t sizze_26212,
                       int32_t sizze_26213, int32_t sizze_26214,
                       int32_t sizze_26215, int32_t sizze_26216,
                       int32_t sizze_26217, int32_t sizze_26218,
                       int32_t sizze_26219, int32_t sizze_26220,
                       int32_t sizze_26221, int32_t sizze_26222,
                       int32_t sizze_26223, int32_t sizze_26224,
                       int32_t sizze_26225, int32_t sizze_26226,
                       int32_t sizze_26227, int32_t sizze_26228,
                       int32_t sizze_26229, int32_t sizze_26230,
                       int32_t sizze_26231, int32_t sizze_26232)
{
    bool scalar_out_26651;
    bool scalar_out_26652;
    bool scalar_out_26653;
    bool scalar_out_26654;
    bool scalar_out_26655;
    bool scalar_out_26656;
    bool scalar_out_26657;
    bool scalar_out_26658;
    bool scalar_out_26659;
    bool dim_zzero_26251 = 0 == sizze_26214;
    bool dim_zzero_26252 = 0 == sizze_26215;
    bool old_empty_26253 = dim_zzero_26251 || dim_zzero_26252;
    bool dim_zzero_26254 = 0 == sizze_26195;
    bool new_empty_26255 = dim_zzero_26252 || dim_zzero_26254;
    bool both_empty_26256 = old_empty_26253 && new_empty_26255;
    bool dim_match_26257 = sizze_26195 == sizze_26214;
    bool empty_or_match_26258 = both_empty_26256 || dim_match_26257;
    bool empty_or_match_cert_26259;
    
    if (!empty_or_match_26258) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-100:4 -> validation-benchmark.fut:24:12-26:25",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_26260 = 0 == sizze_26196;
    bool both_empty_26261 = dim_zzero_26252 && dim_zzero_26260;
    bool dim_match_26262 = sizze_26196 == sizze_26215;
    bool empty_or_match_26263 = both_empty_26261 || dim_match_26262;
    bool empty_or_match_cert_26264;
    
    if (!empty_or_match_26263) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-100:4 -> validation-benchmark.fut:24:12-26:25 -> /futlib/soacs.fut:51:3-37 -> /futlib/soacs.fut:51:19-23 -> validation-benchmark.fut:25:18-62",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool res_26266;
    bool redout_26549 = 1;
    
    for (int32_t i_26550 = 0; i_26550 < sizze_26195; i_26550++) {
        bool res_26273;
        bool redout_26547 = 1;
        
        for (int32_t i_26548 = 0; i_26548 < sizze_26196; i_26548++) {
            float x_26277 = ((float *) X_mem_26633.mem)[i_26550 * sizze_26196 +
                                                        i_26548];
            float x_26278 = ((float *) Xseq_mem_26642.mem)[i_26550 *
                                                           sizze_26215 +
                                                           i_26548];
            float abs_arg_26279 = x_26277 - x_26278;
            float res_26280 = (float) fabs(abs_arg_26279);
            bool res_26281 = res_26280 < 1.0e-2F;
            bool x_26276 = res_26281 && redout_26547;
            bool redout_tmp_26661 = x_26276;
            
            redout_26547 = redout_tmp_26661;
        }
        res_26273 = redout_26547;
        
        bool x_26269 = res_26273 && redout_26549;
        bool redout_tmp_26660 = x_26269;
        
        redout_26549 = redout_tmp_26660;
    }
    res_26266 = redout_26549;
    
    bool dim_zzero_26282 = 0 == sizze_26216;
    bool dim_zzero_26283 = 0 == sizze_26217;
    bool dim_zzero_26284 = 0 == sizze_26218;
    bool y_26285 = dim_zzero_26283 || dim_zzero_26284;
    bool old_empty_26286 = dim_zzero_26282 || y_26285;
    bool dim_zzero_26287 = 0 == sizze_26197;
    bool new_empty_26288 = y_26285 || dim_zzero_26287;
    bool both_empty_26289 = old_empty_26286 && new_empty_26288;
    bool dim_match_26290 = sizze_26197 == sizze_26216;
    bool empty_or_match_26291 = both_empty_26289 || dim_match_26290;
    bool empty_or_match_cert_26292;
    
    if (!empty_or_match_26291) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-100:4 -> validation-benchmark.fut:31:15-35:39",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_26293 = 0 == sizze_26198;
    bool new_empty_26294 = dim_zzero_26284 || dim_zzero_26293;
    bool both_empty_26295 = y_26285 && new_empty_26294;
    bool dim_match_26296 = sizze_26198 == sizze_26217;
    bool empty_or_match_26297 = both_empty_26295 || dim_match_26296;
    bool empty_or_match_cert_26298;
    
    if (!empty_or_match_26297) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-100:4 -> validation-benchmark.fut:31:15-35:39 -> /futlib/soacs.fut:51:3-37 -> /futlib/soacs.fut:51:19-23 -> validation-benchmark.fut:32:25-34:35",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_26299 = 0 == sizze_26199;
    bool both_empty_26300 = dim_zzero_26284 && dim_zzero_26299;
    bool dim_match_26301 = sizze_26199 == sizze_26218;
    bool empty_or_match_26302 = both_empty_26300 || dim_match_26301;
    bool empty_or_match_cert_26303;
    
    if (!empty_or_match_26302) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-100:4 -> validation-benchmark.fut:31:15-35:39 -> /futlib/soacs.fut:51:3-37 -> /futlib/soacs.fut:51:19-23 -> validation-benchmark.fut:32:25-34:35 -> /futlib/soacs.fut:51:3-37 -> /futlib/soacs.fut:51:19-23 -> validation-benchmark.fut:33:31-80",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool res_26305;
    bool redout_26555 = 1;
    
    for (int32_t i_26556 = 0; i_26556 < sizze_26197; i_26556++) {
        bool res_26312;
        bool redout_26553 = 1;
        
        for (int32_t i_26554 = 0; i_26554 < sizze_26198; i_26554++) {
            bool res_26319;
            bool redout_26551 = 1;
            
            for (int32_t i_26552 = 0; i_26552 < sizze_26199; i_26552++) {
                float x_26323 = ((float *) Xsqr_mem_26634.mem)[i_26556 *
                                                               (sizze_26199 *
                                                                sizze_26198) +
                                                               i_26554 *
                                                               sizze_26199 +
                                                               i_26552];
                float x_26324 = ((float *) Xsqrseq_mem_26643.mem)[i_26556 *
                                                                  (sizze_26218 *
                                                                   sizze_26217) +
                                                                  i_26554 *
                                                                  sizze_26218 +
                                                                  i_26552];
                float abs_arg_26325 = x_26323 - x_26324;
                float res_26326 = (float) fabs(abs_arg_26325);
                bool res_26327 = res_26326 < 0.1F;
                bool x_26322 = res_26327 && redout_26551;
                bool redout_tmp_26664 = x_26322;
                
                redout_26551 = redout_tmp_26664;
            }
            res_26319 = redout_26551;
            
            bool x_26315 = res_26319 && redout_26553;
            bool redout_tmp_26663 = x_26315;
            
            redout_26553 = redout_tmp_26663;
        }
        res_26312 = redout_26553;
        
        bool x_26308 = res_26312 && redout_26555;
        bool redout_tmp_26662 = x_26308;
        
        redout_26555 = redout_tmp_26662;
    }
    res_26305 = redout_26555;
    
    bool dim_zzero_26328 = 0 == sizze_26219;
    bool dim_zzero_26329 = 0 == sizze_26220;
    bool dim_zzero_26330 = 0 == sizze_26221;
    bool y_26331 = dim_zzero_26329 || dim_zzero_26330;
    bool old_empty_26332 = dim_zzero_26328 || y_26331;
    bool dim_zzero_26333 = 0 == sizze_26200;
    bool new_empty_26334 = y_26331 || dim_zzero_26333;
    bool both_empty_26335 = old_empty_26332 && new_empty_26334;
    bool dim_match_26336 = sizze_26200 == sizze_26219;
    bool empty_or_match_26337 = both_empty_26335 || dim_match_26336;
    bool empty_or_match_cert_26338;
    
    if (!empty_or_match_26337) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-100:4 -> validation-benchmark.fut:42:15-46:39",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_26339 = 0 == sizze_26201;
    bool new_empty_26340 = dim_zzero_26330 || dim_zzero_26339;
    bool both_empty_26341 = y_26331 && new_empty_26340;
    bool dim_match_26342 = sizze_26201 == sizze_26220;
    bool empty_or_match_26343 = both_empty_26341 || dim_match_26342;
    bool empty_or_match_cert_26344;
    
    if (!empty_or_match_26343) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-100:4 -> validation-benchmark.fut:42:15-46:39 -> /futlib/soacs.fut:51:3-37 -> /futlib/soacs.fut:51:19-23 -> validation-benchmark.fut:43:25-45:35",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_26345 = 0 == sizze_26202;
    bool both_empty_26346 = dim_zzero_26330 && dim_zzero_26345;
    bool dim_match_26347 = sizze_26202 == sizze_26221;
    bool empty_or_match_26348 = both_empty_26346 || dim_match_26347;
    bool empty_or_match_cert_26349;
    
    if (!empty_or_match_26348) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-100:4 -> validation-benchmark.fut:42:15-46:39 -> /futlib/soacs.fut:51:3-37 -> /futlib/soacs.fut:51:19-23 -> validation-benchmark.fut:43:25-45:35 -> /futlib/soacs.fut:51:3-37 -> /futlib/soacs.fut:51:19-23 -> validation-benchmark.fut:44:31-86",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool res_26351;
    bool redout_26561 = 1;
    
    for (int32_t i_26562 = 0; i_26562 < sizze_26200; i_26562++) {
        bool res_26358;
        bool redout_26559 = 1;
        
        for (int32_t i_26560 = 0; i_26560 < sizze_26201; i_26560++) {
            bool res_26365;
            bool redout_26557 = 1;
            
            for (int32_t i_26558 = 0; i_26558 < sizze_26202; i_26558++) {
                float x_26369 = ((float *) Xinv_mem_26635.mem)[i_26562 *
                                                               (sizze_26202 *
                                                                sizze_26201) +
                                                               i_26560 *
                                                               sizze_26202 +
                                                               i_26558];
                float x_26370 = ((float *) Xinvseq_mem_26644.mem)[i_26562 *
                                                                  (sizze_26221 *
                                                                   sizze_26220) +
                                                                  i_26560 *
                                                                  sizze_26221 +
                                                                  i_26558];
                float abs_arg_26371 = x_26369 - x_26370;
                float res_26372 = (float) fabs(abs_arg_26371);
                bool res_26373 = res_26372 < 1.0e-7F;
                bool x_26368 = res_26373 && redout_26557;
                bool redout_tmp_26667 = x_26368;
                
                redout_26557 = redout_tmp_26667;
            }
            res_26365 = redout_26557;
            
            bool x_26361 = res_26365 && redout_26559;
            bool redout_tmp_26666 = x_26361;
            
            redout_26559 = redout_tmp_26666;
        }
        res_26358 = redout_26559;
        
        bool x_26354 = res_26358 && redout_26561;
        bool redout_tmp_26665 = x_26354;
        
        redout_26561 = redout_tmp_26665;
    }
    res_26351 = redout_26561;
    
    bool dim_zzero_26374 = 0 == sizze_26222;
    bool dim_zzero_26375 = 0 == sizze_26223;
    bool old_empty_26376 = dim_zzero_26374 || dim_zzero_26375;
    bool dim_zzero_26377 = 0 == sizze_26203;
    bool new_empty_26378 = dim_zzero_26375 || dim_zzero_26377;
    bool both_empty_26379 = old_empty_26376 && new_empty_26378;
    bool dim_match_26380 = sizze_26203 == sizze_26222;
    bool empty_or_match_26381 = both_empty_26379 || dim_match_26380;
    bool empty_or_match_cert_26382;
    
    if (!empty_or_match_26381) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-100:4 -> validation-benchmark.fut:53:16-55:33",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_26383 = 0 == sizze_26204;
    bool both_empty_26384 = dim_zzero_26375 && dim_zzero_26383;
    bool dim_match_26385 = sizze_26204 == sizze_26223;
    bool empty_or_match_26386 = both_empty_26384 || dim_match_26385;
    bool empty_or_match_cert_26387;
    
    if (!empty_or_match_26386) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-100:4 -> validation-benchmark.fut:53:16-55:33 -> /futlib/soacs.fut:51:3-37 -> /futlib/soacs.fut:51:19-23 -> validation-benchmark.fut:54:18-61",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool res_26389;
    bool redout_26565 = 1;
    
    for (int32_t i_26566 = 0; i_26566 < sizze_26203; i_26566++) {
        bool res_26396;
        bool redout_26563 = 1;
        
        for (int32_t i_26564 = 0; i_26564 < sizze_26204; i_26564++) {
            float x_26400 = ((float *) beta0_mem_26636.mem)[i_26566 *
                                                            sizze_26204 +
                                                            i_26564];
            float x_26401 = ((float *) beta0seq_mem_26645.mem)[i_26566 *
                                                               sizze_26223 +
                                                               i_26564];
            float abs_arg_26402 = x_26400 - x_26401;
            float res_26403 = (float) fabs(abs_arg_26402);
            bool res_26404 = res_26403 < 1.1F;
            bool x_26399 = res_26404 && redout_26563;
            bool redout_tmp_26669 = x_26399;
            
            redout_26563 = redout_tmp_26669;
        }
        res_26396 = redout_26563;
        
        bool x_26392 = res_26396 && redout_26565;
        bool redout_tmp_26668 = x_26392;
        
        redout_26565 = redout_tmp_26668;
    }
    res_26389 = redout_26565;
    
    bool dim_zzero_26405 = 0 == sizze_26224;
    bool dim_zzero_26406 = 0 == sizze_26225;
    bool old_empty_26407 = dim_zzero_26405 || dim_zzero_26406;
    bool dim_zzero_26408 = 0 == sizze_26205;
    bool new_empty_26409 = dim_zzero_26406 || dim_zzero_26408;
    bool both_empty_26410 = old_empty_26407 && new_empty_26409;
    bool dim_match_26411 = sizze_26205 == sizze_26224;
    bool empty_or_match_26412 = both_empty_26410 || dim_match_26411;
    bool empty_or_match_cert_26413;
    
    if (!empty_or_match_26412) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-100:4 -> validation-benchmark.fut:60:15-62:31",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_26414 = 0 == sizze_26206;
    bool both_empty_26415 = dim_zzero_26406 && dim_zzero_26414;
    bool dim_match_26416 = sizze_26206 == sizze_26225;
    bool empty_or_match_26417 = both_empty_26415 || dim_match_26416;
    bool empty_or_match_cert_26418;
    
    if (!empty_or_match_26417) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-100:4 -> validation-benchmark.fut:60:15-62:31 -> /futlib/soacs.fut:51:3-37 -> /futlib/soacs.fut:51:19-23 -> validation-benchmark.fut:61:18-61",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool res_26420;
    bool redout_26569 = 1;
    
    for (int32_t i_26570 = 0; i_26570 < sizze_26205; i_26570++) {
        bool res_26427;
        bool redout_26567 = 1;
        
        for (int32_t i_26568 = 0; i_26568 < sizze_26206; i_26568++) {
            float x_26431 = ((float *) beta_mem_26637.mem)[i_26570 *
                                                           sizze_26206 +
                                                           i_26568];
            float x_26432 = ((float *) betaseq_mem_26646.mem)[i_26570 *
                                                              sizze_26225 +
                                                              i_26568];
            float abs_arg_26433 = x_26431 - x_26432;
            float res_26434 = (float) fabs(abs_arg_26433);
            bool res_26435 = res_26434 < 0.1F;
            bool x_26430 = res_26435 && redout_26567;
            bool redout_tmp_26671 = x_26430;
            
            redout_26567 = redout_tmp_26671;
        }
        res_26427 = redout_26567;
        
        bool x_26423 = res_26427 && redout_26569;
        bool redout_tmp_26670 = x_26423;
        
        redout_26569 = redout_tmp_26670;
    }
    res_26420 = redout_26569;
    
    bool dim_zzero_26436 = 0 == sizze_26226;
    bool dim_zzero_26437 = 0 == sizze_26227;
    bool old_empty_26438 = dim_zzero_26436 || dim_zzero_26437;
    bool dim_zzero_26439 = 0 == sizze_26207;
    bool new_empty_26440 = dim_zzero_26437 || dim_zzero_26439;
    bool both_empty_26441 = old_empty_26438 && new_empty_26440;
    bool dim_match_26442 = sizze_26207 == sizze_26226;
    bool empty_or_match_26443 = both_empty_26441 || dim_match_26442;
    bool empty_or_match_cert_26444;
    
    if (!empty_or_match_26443) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-100:4 -> validation-benchmark.fut:67:18-69:37",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_26445 = 0 == sizze_26208;
    bool both_empty_26446 = dim_zzero_26437 && dim_zzero_26445;
    bool dim_match_26447 = sizze_26208 == sizze_26227;
    bool empty_or_match_26448 = both_empty_26446 || dim_match_26447;
    bool empty_or_match_cert_26449;
    
    if (!empty_or_match_26448) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-100:4 -> validation-benchmark.fut:67:18-69:37 -> /futlib/soacs.fut:51:3-37 -> /futlib/soacs.fut:51:19-23 -> validation-benchmark.fut:68:18-61",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool res_26451;
    bool redout_26573 = 1;
    
    for (int32_t i_26574 = 0; i_26574 < sizze_26207; i_26574++) {
        bool res_26458;
        bool redout_26571 = 1;
        
        for (int32_t i_26572 = 0; i_26572 < sizze_26208; i_26572++) {
            float x_26462 = ((float *) y_preds_mem_26638.mem)[i_26574 *
                                                              sizze_26208 +
                                                              i_26572];
            float x_26463 = ((float *) y_predsseq_mem_26647.mem)[i_26574 *
                                                                 sizze_26227 +
                                                                 i_26572];
            float abs_arg_26464 = x_26462 - x_26463;
            float res_26465 = (float) fabs(abs_arg_26464);
            bool res_26466 = res_26465 < 0.1F;
            bool x_26461 = res_26466 && redout_26571;
            bool redout_tmp_26673 = x_26461;
            
            redout_26571 = redout_tmp_26673;
        }
        res_26458 = redout_26571;
        
        bool x_26454 = res_26458 && redout_26573;
        bool redout_tmp_26672 = x_26454;
        
        redout_26573 = redout_tmp_26672;
    }
    res_26451 = redout_26573;
    
    bool dim_zzero_26467 = 0 == sizze_26228;
    bool dim_zzero_26468 = 0 == sizze_26209;
    bool both_empty_26469 = dim_zzero_26467 && dim_zzero_26468;
    bool dim_match_26470 = sizze_26209 == sizze_26228;
    bool empty_or_match_26471 = both_empty_26469 || dim_match_26470;
    bool empty_or_match_cert_26472;
    
    if (!empty_or_match_26471) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-100:4 -> validation-benchmark.fut:74:18-66",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool res_26474;
    bool redout_26575 = 1;
    
    for (int32_t i_26576 = 0; i_26576 < sizze_26209; i_26576++) {
        int32_t x_26478 = ((int32_t *) Nss_mem_26639.mem)[i_26576];
        int32_t x_26479 = ((int32_t *) Nssseq_mem_26648.mem)[i_26576];
        int32_t abs_arg_26480 = x_26478 - x_26479;
        int32_t res_26481 = abs(abs_arg_26480);
        bool res_26482 = slt32(res_26481, 1);
        bool x_26477 = res_26482 && redout_26575;
        bool redout_tmp_26674 = x_26477;
        
        redout_26575 = redout_tmp_26674;
    }
    res_26474 = redout_26575;
    
    bool dim_zzero_26483 = 0 == sizze_26229;
    bool dim_zzero_26484 = 0 == sizze_26230;
    bool old_empty_26485 = dim_zzero_26483 || dim_zzero_26484;
    bool dim_zzero_26486 = 0 == sizze_26210;
    bool new_empty_26487 = dim_zzero_26484 || dim_zzero_26486;
    bool both_empty_26488 = old_empty_26485 && new_empty_26487;
    bool dim_match_26489 = sizze_26210 == sizze_26229;
    bool empty_or_match_26490 = both_empty_26488 || dim_match_26489;
    bool empty_or_match_cert_26491;
    
    if (!empty_or_match_26490) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-100:4 -> validation-benchmark.fut:76:19-80:46",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_26492 = 0 == sizze_26211;
    bool both_empty_26493 = dim_zzero_26484 && dim_zzero_26492;
    bool dim_match_26494 = sizze_26211 == sizze_26230;
    bool empty_or_match_26495 = both_empty_26493 || dim_match_26494;
    bool empty_or_match_cert_26496;
    
    if (!empty_or_match_26495) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-100:4 -> validation-benchmark.fut:76:19-80:46 -> /futlib/soacs.fut:51:3-37 -> /futlib/soacs.fut:51:19-23 -> validation-benchmark.fut:77:25-79:73",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool res_26498;
    bool redout_26579 = 1;
    
    for (int32_t i_26580 = 0; i_26580 < sizze_26210; i_26580++) {
        bool res_26505;
        bool redout_26577 = 1;
        
        for (int32_t i_26578 = 0; i_26578 < sizze_26211; i_26578++) {
            float x_26509 = ((float *) y_errors_mem_26640.mem)[i_26580 *
                                                               sizze_26211 +
                                                               i_26578];
            float x_26510 = ((float *) y_errorsseq_mem_26649.mem)[i_26580 *
                                                                  sizze_26230 +
                                                                  i_26578];
            bool res_26511;
            
            res_26511 = futrts_isnan32(x_26509);
            
            float abs_arg_26512 = x_26509 - x_26510;
            float res_26513 = (float) fabs(abs_arg_26512);
            bool res_26514 = res_26513 < 0.1F;
            bool x_26515 = !res_26511;
            bool y_26516 = res_26514 && x_26515;
            bool res_26517 = res_26511 || y_26516;
            bool x_26508 = res_26517 && redout_26577;
            bool redout_tmp_26676 = x_26508;
            
            redout_26577 = redout_tmp_26676;
        }
        res_26505 = redout_26577;
        
        bool x_26501 = res_26505 && redout_26579;
        bool redout_tmp_26675 = x_26501;
        
        redout_26579 = redout_tmp_26675;
    }
    res_26498 = redout_26579;
    
    bool dim_zzero_26518 = 0 == sizze_26231;
    bool dim_zzero_26519 = 0 == sizze_26232;
    bool old_empty_26520 = dim_zzero_26518 || dim_zzero_26519;
    bool dim_zzero_26521 = 0 == sizze_26212;
    bool new_empty_26522 = dim_zzero_26519 || dim_zzero_26521;
    bool both_empty_26523 = old_empty_26520 && new_empty_26522;
    bool dim_match_26524 = sizze_26212 == sizze_26231;
    bool empty_or_match_26525 = both_empty_26523 || dim_match_26524;
    bool empty_or_match_cert_26526;
    
    if (!empty_or_match_26525) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-100:4 -> validation-benchmark.fut:84:20-86:48",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_26527 = 0 == sizze_26213;
    bool both_empty_26528 = dim_zzero_26519 && dim_zzero_26527;
    bool dim_match_26529 = sizze_26213 == sizze_26232;
    bool empty_or_match_26530 = both_empty_26528 || dim_match_26529;
    bool empty_or_match_cert_26531;
    
    if (!empty_or_match_26530) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "validation-benchmark.fut:8:1-100:4 -> validation-benchmark.fut:84:20-86:48 -> /futlib/soacs.fut:51:3-37 -> /futlib/soacs.fut:51:19-23 -> validation-benchmark.fut:85:25-53",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool res_26533;
    bool redout_26583 = 1;
    
    for (int32_t i_26584 = 0; i_26584 < sizze_26212; i_26584++) {
        bool res_26540;
        bool redout_26581 = 1;
        
        for (int32_t i_26582 = 0; i_26582 < sizze_26213; i_26582++) {
            int32_t x_26544 = ((int32_t *) val_indss_mem_26641.mem)[i_26584 *
                                                                    sizze_26213 +
                                                                    i_26582];
            int32_t x_26545 = ((int32_t *) val_indssseq_mem_26650.mem)[i_26584 *
                                                                       sizze_26232 +
                                                                       i_26582];
            bool res_26546 = x_26544 == x_26545;
            bool x_26543 = res_26546 && redout_26581;
            bool redout_tmp_26678 = x_26543;
            
            redout_26581 = redout_tmp_26678;
        }
        res_26540 = redout_26581;
        
        bool x_26536 = res_26540 && redout_26583;
        bool redout_tmp_26677 = x_26536;
        
        redout_26583 = redout_tmp_26677;
    }
    res_26533 = redout_26583;
    scalar_out_26651 = res_26266;
    scalar_out_26652 = res_26305;
    scalar_out_26653 = res_26351;
    scalar_out_26654 = res_26389;
    scalar_out_26655 = res_26420;
    scalar_out_26656 = res_26451;
    scalar_out_26657 = res_26474;
    scalar_out_26658 = res_26498;
    scalar_out_26659 = res_26533;
    *out_scalar_out_26679 = scalar_out_26651;
    *out_scalar_out_26680 = scalar_out_26652;
    *out_scalar_out_26681 = scalar_out_26653;
    *out_scalar_out_26682 = scalar_out_26654;
    *out_scalar_out_26683 = scalar_out_26655;
    *out_scalar_out_26684 = scalar_out_26656;
    *out_scalar_out_26685 = scalar_out_26657;
    *out_scalar_out_26686 = scalar_out_26658;
    *out_scalar_out_26687 = scalar_out_26659;
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
                       bool *out6, bool *out7, bool *out8, const
                       struct futhark_f32_2d *in0, const
                       struct futhark_f32_3d *in1, const
                       struct futhark_f32_3d *in2, const
                       struct futhark_f32_2d *in3, const
                       struct futhark_f32_2d *in4, const
                       struct futhark_f32_2d *in5, const
                       struct futhark_i32_1d *in6, const
                       struct futhark_f32_2d *in7, const
                       struct futhark_i32_2d *in8, const
                       struct futhark_f32_2d *in9, const
                       struct futhark_f32_3d *in10, const
                       struct futhark_f32_3d *in11, const
                       struct futhark_f32_2d *in12, const
                       struct futhark_f32_2d *in13, const
                       struct futhark_f32_2d *in14, const
                       struct futhark_i32_1d *in15, const
                       struct futhark_f32_2d *in16, const
                       struct futhark_i32_2d *in17)
{
    struct memblock X_mem_26633;
    
    X_mem_26633.references = NULL;
    
    struct memblock Xsqr_mem_26634;
    
    Xsqr_mem_26634.references = NULL;
    
    struct memblock Xinv_mem_26635;
    
    Xinv_mem_26635.references = NULL;
    
    struct memblock beta0_mem_26636;
    
    beta0_mem_26636.references = NULL;
    
    struct memblock beta_mem_26637;
    
    beta_mem_26637.references = NULL;
    
    struct memblock y_preds_mem_26638;
    
    y_preds_mem_26638.references = NULL;
    
    struct memblock Nss_mem_26639;
    
    Nss_mem_26639.references = NULL;
    
    struct memblock y_errors_mem_26640;
    
    y_errors_mem_26640.references = NULL;
    
    struct memblock val_indss_mem_26641;
    
    val_indss_mem_26641.references = NULL;
    
    struct memblock Xseq_mem_26642;
    
    Xseq_mem_26642.references = NULL;
    
    struct memblock Xsqrseq_mem_26643;
    
    Xsqrseq_mem_26643.references = NULL;
    
    struct memblock Xinvseq_mem_26644;
    
    Xinvseq_mem_26644.references = NULL;
    
    struct memblock beta0seq_mem_26645;
    
    beta0seq_mem_26645.references = NULL;
    
    struct memblock betaseq_mem_26646;
    
    betaseq_mem_26646.references = NULL;
    
    struct memblock y_predsseq_mem_26647;
    
    y_predsseq_mem_26647.references = NULL;
    
    struct memblock Nssseq_mem_26648;
    
    Nssseq_mem_26648.references = NULL;
    
    struct memblock y_errorsseq_mem_26649;
    
    y_errorsseq_mem_26649.references = NULL;
    
    struct memblock val_indssseq_mem_26650;
    
    val_indssseq_mem_26650.references = NULL;
    
    int32_t sizze_26195;
    int32_t sizze_26196;
    int32_t sizze_26197;
    int32_t sizze_26198;
    int32_t sizze_26199;
    int32_t sizze_26200;
    int32_t sizze_26201;
    int32_t sizze_26202;
    int32_t sizze_26203;
    int32_t sizze_26204;
    int32_t sizze_26205;
    int32_t sizze_26206;
    int32_t sizze_26207;
    int32_t sizze_26208;
    int32_t sizze_26209;
    int32_t sizze_26210;
    int32_t sizze_26211;
    int32_t sizze_26212;
    int32_t sizze_26213;
    int32_t sizze_26214;
    int32_t sizze_26215;
    int32_t sizze_26216;
    int32_t sizze_26217;
    int32_t sizze_26218;
    int32_t sizze_26219;
    int32_t sizze_26220;
    int32_t sizze_26221;
    int32_t sizze_26222;
    int32_t sizze_26223;
    int32_t sizze_26224;
    int32_t sizze_26225;
    int32_t sizze_26226;
    int32_t sizze_26227;
    int32_t sizze_26228;
    int32_t sizze_26229;
    int32_t sizze_26230;
    int32_t sizze_26231;
    int32_t sizze_26232;
    bool scalar_out_26651;
    bool scalar_out_26652;
    bool scalar_out_26653;
    bool scalar_out_26654;
    bool scalar_out_26655;
    bool scalar_out_26656;
    bool scalar_out_26657;
    bool scalar_out_26658;
    bool scalar_out_26659;
    
    lock_lock(&ctx->lock);
    X_mem_26633 = in0->mem;
    sizze_26195 = in0->shape[0];
    sizze_26196 = in0->shape[1];
    Xsqr_mem_26634 = in1->mem;
    sizze_26197 = in1->shape[0];
    sizze_26198 = in1->shape[1];
    sizze_26199 = in1->shape[2];
    Xinv_mem_26635 = in2->mem;
    sizze_26200 = in2->shape[0];
    sizze_26201 = in2->shape[1];
    sizze_26202 = in2->shape[2];
    beta0_mem_26636 = in3->mem;
    sizze_26203 = in3->shape[0];
    sizze_26204 = in3->shape[1];
    beta_mem_26637 = in4->mem;
    sizze_26205 = in4->shape[0];
    sizze_26206 = in4->shape[1];
    y_preds_mem_26638 = in5->mem;
    sizze_26207 = in5->shape[0];
    sizze_26208 = in5->shape[1];
    Nss_mem_26639 = in6->mem;
    sizze_26209 = in6->shape[0];
    y_errors_mem_26640 = in7->mem;
    sizze_26210 = in7->shape[0];
    sizze_26211 = in7->shape[1];
    val_indss_mem_26641 = in8->mem;
    sizze_26212 = in8->shape[0];
    sizze_26213 = in8->shape[1];
    Xseq_mem_26642 = in9->mem;
    sizze_26214 = in9->shape[0];
    sizze_26215 = in9->shape[1];
    Xsqrseq_mem_26643 = in10->mem;
    sizze_26216 = in10->shape[0];
    sizze_26217 = in10->shape[1];
    sizze_26218 = in10->shape[2];
    Xinvseq_mem_26644 = in11->mem;
    sizze_26219 = in11->shape[0];
    sizze_26220 = in11->shape[1];
    sizze_26221 = in11->shape[2];
    beta0seq_mem_26645 = in12->mem;
    sizze_26222 = in12->shape[0];
    sizze_26223 = in12->shape[1];
    betaseq_mem_26646 = in13->mem;
    sizze_26224 = in13->shape[0];
    sizze_26225 = in13->shape[1];
    y_predsseq_mem_26647 = in14->mem;
    sizze_26226 = in14->shape[0];
    sizze_26227 = in14->shape[1];
    Nssseq_mem_26648 = in15->mem;
    sizze_26228 = in15->shape[0];
    y_errorsseq_mem_26649 = in16->mem;
    sizze_26229 = in16->shape[0];
    sizze_26230 = in16->shape[1];
    val_indssseq_mem_26650 = in17->mem;
    sizze_26231 = in17->shape[0];
    sizze_26232 = in17->shape[1];
    
    int ret = futrts_main(ctx, &scalar_out_26651, &scalar_out_26652,
                          &scalar_out_26653, &scalar_out_26654,
                          &scalar_out_26655, &scalar_out_26656,
                          &scalar_out_26657, &scalar_out_26658,
                          &scalar_out_26659, X_mem_26633, Xsqr_mem_26634,
                          Xinv_mem_26635, beta0_mem_26636, beta_mem_26637,
                          y_preds_mem_26638, Nss_mem_26639, y_errors_mem_26640,
                          val_indss_mem_26641, Xseq_mem_26642,
                          Xsqrseq_mem_26643, Xinvseq_mem_26644,
                          beta0seq_mem_26645, betaseq_mem_26646,
                          y_predsseq_mem_26647, Nssseq_mem_26648,
                          y_errorsseq_mem_26649, val_indssseq_mem_26650,
                          sizze_26195, sizze_26196, sizze_26197, sizze_26198,
                          sizze_26199, sizze_26200, sizze_26201, sizze_26202,
                          sizze_26203, sizze_26204, sizze_26205, sizze_26206,
                          sizze_26207, sizze_26208, sizze_26209, sizze_26210,
                          sizze_26211, sizze_26212, sizze_26213, sizze_26214,
                          sizze_26215, sizze_26216, sizze_26217, sizze_26218,
                          sizze_26219, sizze_26220, sizze_26221, sizze_26222,
                          sizze_26223, sizze_26224, sizze_26225, sizze_26226,
                          sizze_26227, sizze_26228, sizze_26229, sizze_26230,
                          sizze_26231, sizze_26232);
    
    if (ret == 0) {
        *out0 = scalar_out_26651;
        *out1 = scalar_out_26652;
        *out2 = scalar_out_26653;
        *out3 = scalar_out_26654;
        *out4 = scalar_out_26655;
        *out5 = scalar_out_26656;
        *out6 = scalar_out_26657;
        *out7 = scalar_out_26658;
        *out8 = scalar_out_26659;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
